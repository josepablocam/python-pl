# Construct a very rough data/control flow-ish dependency graph
# References: http://web.cs.iastate.edu/~weile/cs513x/5.DependencySlicing.pdf

import ast
from copy import deepcopy
from enum import Enum

from astunparse import unparse
import networkx as nx
import matplotlib.pyplot as plt


class ControlFlowMarkers(Enum):
    """
    Used to keep track of control-flow structure for later purposes of lifting a slice
    into source code again.
    """
    TEST = 1
    TRUE_BRANCH = 2
    FALSE_BRANCH = 3


def safe_enlist(x):
    try:
        return list(x)
    except TypeError:
        return list([x])


class ExtractNestedReferences(ast.NodeVisitor):
    """
    Extract nested names and attributes references along with the member reference
    depth at which they were found (depth here references to AST depth, not member access depth).
    The result is a nested list associated with each name/attribute
    For example:
        a.b.c ->   [(a, 2), (a.b, 1),  (a.b.c, 0)]
        c + a.b -> [(c, 0)], [(a.b, 0),  (a, 1)]
        a + a.b -> [(a, 0)], [(a.b, 0),  (a, 1)]

    """
    def __init__(self):
        self.nodes_with_member_ast_depth = []
        self.depth = -1
        self.tmp = []

    def collect(self, node):
        copy_node = deepcopy(node)
        nd = (copy_node, self.depth)
        if self.depth < 0:
            # not inside an attribute, so just add now
            self.nodes_with_member_ast_depth.append([nd])
        else:
            # accumulate until done
            self.tmp.append(nd)

    def visit_Name(self, node):
        self.collect(node)

    def visit_Attribute(self, node):
        self.depth += 1
        self.collect(node)
        self.generic_visit(node)
        self.depth -= 1

        if self.depth < 0:
            self.nodes_with_member_ast_depth.append(self.tmp)
            self.tmp = []

    def run(self, node):
        self.visit(node)
        return self.nodes_with_member_ast_depth


class DependenciesConstructor(ast.NodeVisitor):
    """
    Produces a very rough data/cf dependency graph. It over-approximates significantly
    and is not sound.

    Functions/Classes are treated independently from any surrounding code, so there are
    no data dependencies across functions or between functions and expressions/statements outside.

    True/False branches in control flow are independent (i.e a value in the false branch
    cannot have a data dependency to an assignment in the true branch), but the dependencies
    from each branch are merged with the previous dependency information upon exit. So for example:

    1: x = 100
    2: if _:
    3:     x = x * 2
    4: else:
    5:     x = x + 4
    6: c = x + 3

    c (6) has dependencies on x (1), x(3) and x(5). We ignore implicit flows.
    x (3) has a dependency on x(1)
    x (5) has a dependency on x(1)

    We also overapproximate when accessing object attributes. For example:
    a.b.c = w + 2

    associates a, a.b, and a.b.c with this assignment for data dependency information. This is meant to address
    situations such as: a.b = 100; a.c() , this last use should depend on the previous assignment, as it may
    be that a.c() uses a.b in its computation. Given that we don't know either way, we can establish the over-approximate
    dependency by making sure that a.b = 100 updates the last statement id for both (a) and (a.b).

    An expression of the form:
    a.b.c + 2

    in turn, will depend on a.b.c (if available), else a.b (if available), or a (if available).
    If it does not find any of these previous assignments, it will allocate the most specific version (a.b.c),
    make this the dependency.

    For subscripting we always associate with the source. So a[..][...][..] = 1 updates the last statement id for (a).
    """
    def __init__(self, assume_standalone_calls_mutate=False):
        # We should construct a graph representation using data dependencies
        # wrap it in some existing graph library
        # we can then do
        self.id_counter = 0
        self.unknown_id = -1
        self.graph = nx.DiGraph()
        self.scope = [{}]
        self.context = []
        self.cf_dependences = []
        self.assume_standalone_calls_mutate = assume_standalone_calls_mutate

    def run(self, tree):
        if not isinstance(tree, ast.Module):
            tree = ast.parse(tree)
        self.visit(tree)
        return self.graph

    def lookup_node_ids(self, node):
        try:
            assert isinstance(node.ctx, ast.Load), 'Node lookups must be loads'
        except AttributeError:
            pass
        key = ast.dump(node)
        # look from deepest scope -> outwards
        for _map in reversed(self.scope):
            if key in _map:
                # always return a copy
                return set(_map[key])
        # TODO: don't like that this returns different type
        return self.unknown_id

    def allocate_id(self):
        _id = self.id_counter
        self.id_counter += 1
        return _id

    def has_entry(self, node):
        return self.lookup_node_ids(node) != self.unknown_id

    def push_scope(self, scope=None):
        if scope is None:
            scope = {}
        self.scope += [scope]

    def pop_scope(self):
        return self.scope.pop()

    def merge_scopes(self, scopes):
        merged = {}
        for s in scopes:
            for k, _ids in s.items():
                if not k in merged:
                    merged[k] = set([])
                merged[k].update(_ids)
        return merged

    def get_current_scope(self):
        return self.scope[-1]

    def push_context(self, node):
        self.context += [node]

    def pop_context(self):
        if self.context:
            return self.context.pop()
        else:
            return None

    def push_cf_dependence(self, _id):
        self.cf_dependences.append(_id)

    def pop_cf_dependence(self):
        return self.cf_dependences.pop()

    def create_node(self, node):
        _id = self.allocate_id()
        self.graph.add_node(_id)
        self.graph.node[_id]['ast'] = node
        self.graph.node[_id]['context'] = list(self.context
                                               ) if self.context else []
        return _id

    def update_node_id(self, node, _id, append=False):
        # make sure the node is now in context Load()
        load_node = deepcopy(node)
        try:
            load_node.ctx = ast.Load()
        except AttributeError:
            pass
        current_scope = self.get_current_scope()
        key = ast.dump(load_node)
        if isinstance(_id, int):
            _id = set([_id])
        if append:
            current_scope[key].update(_id)
        else:
            current_scope[key] = set(_id)

    def extract_reference_nodes(self, node):
        return ExtractNestedReferences().run(node)

    def extract_dependence_ids(self, reference_nodes):
        ids = set([])
        for ref_node in reference_nodes:
            if not self.has_entry(ref_node):
                # if we don't know where it came from
                # just allocate a new node in the graph
                _id = self.create_node(ref_node)
                self.update_node_id(ref_node, [_id], append=False)
            ref_ids = self.lookup_node_ids(ref_node)
            ids.update(ref_ids)
        return ids

    def stores(self, current_stmt_id, nodes, extract_references=True):
        """
        Update the source line for all references(nodes) to be the current_stmt_id
        """
        store_references = []
        for target in safe_enlist(nodes):
            if extract_references:
                references = self.extract_reference_nodes(target)
                references = [
                    ref for nested in references for ref, ast_depth in nested
                ]
            else:
                references = nodes
            for ref in references:
                self.update_node_id(ref, current_stmt_id, append=False)

    def loads(self, current_stmt_id, nodes):
        """
        Add dependency edge between all sources of references used in nodes and the current_stmt_id
        """
        clean_references = []
        for use in safe_enlist(nodes):
            load_references = self.extract_reference_nodes(use)
            for nested_refs in load_references:
                # sort in ascending order of ast depth, less deep references are taken if available
                # as these correspond to more precise member access
                sorted_refs = [
                    r for r, _ in sorted(nested_refs, key=lambda x: x[1])
                ]
                # if we don't find anything, we'll just add the most specific
                ref_to_add = sorted_refs[0]
                for ref in sorted_refs:
                    if self.has_entry(ref):
                        ref_to_add = ref
                        break
                clean_references.append(ref_to_add)

        # establish edges
        depends_on_ids = self.extract_dependence_ids(clean_references)

        # add in any implicit dependence if necessary
        if self.cf_dependences:
            depends_on_ids.add(self.cf_dependences[-1])

        edges = [(dep, current_stmt_id) for dep in depends_on_ids]
        self.graph.add_edges_from(edges)

    def visit_Import(self, node):
        import_id = self.create_node(node)
        for name in node.names:
            identifier = name.asname if name.asname else name.name
            # wrap the identifier in the appropriate node
            # so that later lookups when used in expressions work as expected
            key_node = ast.parse(identifier).body[0].value
            try:
                key_node.ctx = ast.Store()
            except AttributeError:
                pass
            self.stores(import_id, [key_node], extract_references=False)

    def visit_ImportFrom(self, node):
        self.visit_Import(node)

    def visit_Assign(self, node):
        assign_id = self.create_node(node)
        self.loads(assign_id, node.value)
        self.stores(assign_id, node.targets)

    def visit_AugAssign(self, node):
        assign_id = self.create_node(node)
        implicit_load_node = deepcopy(node.target)
        implicit_load_node.ctx = ast.Load()
        self.loads(assign_id, [node.value, implicit_load_node])
        self.stores(assign_id, node.target)

    def visit_Expr(self, node):
        expr_id = self.create_node(node)
        self.loads(expr_id, node.value)
        if isinstance(node, ast.Call) and self.assume_standalone_calls_mutate:
            self.stores(expr_id, node.value)

    def visit_Return(self, node):
        return_id = self.create_node(node)
        self.loads(return_id, node.value)

    def visit_FunctionDef(self, node):
        self.push_context(node)
        scope = self.scope
        self.scope = [{}]
        for stmt in node.body:
            self.visit(stmt)
        self.scope = scope

    def visit_ClassDef(self, node):
        self.push_context(node)
        scope = self.scope
        self.scope = [{}]
        for stmt in node.body:
            self.visit(stmt)
        self.scope = scope

    def visit_For(self, node):
        iter_id = self.create_node(node.iter)
        self.loads(iter_id, node.iter)

        self.push_cf_dependence(iter_id)
        self.push_scope()
        self.push_context((node, ControlFlowMarkers.TRUE_BRANCH))
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()

        # aded dependene between test and assignments in the body
        # this breaks the DAG
        self.loads(iter_id, node.iter)
        true_scope = self.pop_scope()

        self.push_scope()
        self.push_context((node, ControlFlowMarkers.FALSE_BRANCH))
        for stmt in node.orelse:
            self.visit(stmt)
        self.pop_context()
        false_scope = self.pop_scope()
        self.pop_cf_dependence()

        outer_scope = self.pop_scope()
        merged_scope = self.merge_scopes([
            outer_scope, true_scope, false_scope
        ])
        self.push_scope(merged_scope)

    def visit_While(self, node):
        self.push_context((node, ControlFlowMarkers.TEST))
        test_id = self.create_node(node.test)
        self.loads(test_id, node.test)
        self.pop_context()

        self.push_cf_dependence(test_id)
        self.push_scope()
        self.push_context((node, ControlFlowMarkers.TRUE_BRANCH))
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()

        # add dependence between test and assignments in the body
        # this breaks the DAG
        self.loads(test_id, node.test)
        true_scope = self.pop_scope()

        self.push_scope()
        self.push_context((node, ControlFlowMarkers.FALSE_BRANCH))
        for stmt in node.orelse:
            self.visit(stmt)
        self.pop_context()
        false_scope = self.pop_scope()
        self.pop_cf_dependence()

        outer_scope = self.pop_scope()
        merged_scope = self.merge_scopes([
            outer_scope, true_scope, false_scope
        ])
        self.push_scope(merged_scope)

    def visit_If(self, node):
        self.push_context((node, ControlFlowMarkers.TEST))
        test_id = self.create_node(node.test)
        self.loads(test_id, node.test)
        self.pop_context()

        self.push_cf_dependence(test_id)
        self.push_scope()
        self.push_context((node, ControlFlowMarkers.TRUE_BRANCH))
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()
        true_scope = self.pop_scope()

        self.push_scope()
        self.push_context((node, ControlFlowMarkers.FALSE_BRANCH))
        for stmt in node.orelse:
            self.visit(stmt)
        self.pop_context()
        false_scope = self.pop_scope()
        self.pop_cf_dependence()

        outer_scope = self.pop_scope()
        # if a set of references are assigned in both
        # branches, then future uses cannot depend on assignments
        # before the if-statement
        in_both = set(true_scope.keys()).intersection(false_scope.keys())
        # drop these from the outer scope
        for key in in_both:
            if key in outer_scope:
                outer_scope.pop(key)
        unified_scope = self.merge_scopes([
            outer_scope, true_scope, false_scope
        ])
        self.push_scope(unified_scope)

    def visit_With(self, node):
        for with_item in node.items:
            item_id = self.create_node(with_item)
            self.loads(item_id, with_item.context_expr)

        self.push_context(node)
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()


def build_graph(src, assume_standalone_calls_mutate=False):
    constructor = DependenciesConstructor(
        assume_standalone_calls_mutate=assume_standalone_calls_mutate
    )
    g = constructor.run(src)
    return constructor, g


def draw(g, dot_layout=True):
    fig, ax = plt.subplots(1)
    labels = nx.get_node_attributes(g, 'ast')
    labels = {k: unparse(v).strip() for k, v in labels.items()}
    # use better graphviz layout
    pos = nx.drawing.nx_pydot.graphviz_layout(g) if dot_layout else None
    nx.draw(g, labels=labels, node_size=100, ax=ax, pos=pos)
    plt.show()


def slice_graph(graph, seed, reverse):
    search_graph = graph
    if reverse:
        search_graph = search_graph.reverse(copy=False)
    slice_nodes = nx.dfs_preorder_nodes(search_graph, seed)
    return graph.subgraph(slice_nodes)


def get_node_ids(graph, predicate):
    ids = []
    for _id, node_attributes in graph.nodes.items():
        if predicate(node_attributes):
            ids.append(_id)
    return ids


def slices_from_ids(graph, ids, backwards):
    return [slice_graph(graph, _id, backwards) for _id in ids]


def slices_from_expr(graph, src, backwards):
    src_node = ast.parse(src).body[0]
    ids = get_node_ids(
        graph, lambda attrs: ast.dump(attrs['ast']) == ast.dump(src_node)
    )
    return slices_from_ids(graph, ids, backwards)
