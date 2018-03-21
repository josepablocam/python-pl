# Construct a very rough data dependency graph

import ast
from copy import deepcopy

from astunparse import unparse
import networkx as nx
import matplotlib.pyplot as plt



class ExtractNestedReferences(ast.NodeVisitor):
    """
    Extract nested names and attributes references.
    For example:
        a.b.c -> (a, a.b, a.b.c)
    """
    def __init__(self):
        self.nodes = []
        
    def visit_Name(self, node):
        self.nodes.append(deepcopy(node))
    
    def visit_Attribute(self, node):
        self.nodes.append(deepcopy(node))
        self.generic_visit(node)

    def run(self, node):
        self.visit(node)
        return self.nodes


class DataDependenciesConstructor(ast.NodeVisitor):
    """
    Produces a very rough data dependency graph. It over-approximates significantly
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
    
    associates a, a.b, and a.b.c with this assignment for data dependency information
    
    An expression of the form:
    a.b.c + 2
    
    in turn, will depend on the last assignment to a, a.b, and a.b.c (if any).
    In fact, if it doesn't find previous assignments to any of these, it will allocate
    new nodes (free) and make these the dependency.
    """
    
    def __init__(self):
        # We should construct a graph representation using data dependencies
        # wrap it in some existing graph library
        # we can then do
        self.id_counter = 0
        self.unknown_id = -1
        self.graph = nx.DiGraph()
        self.scope = [{}]
        self.context = []
        
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
        
    def get_current_context(self):
        return self.context if self.context else []
        
    def create_node(self, node):
        _id = self.allocate_id()
        self.graph.add_node(_id)
        self.graph.node[_id]['ast'] = node
        self.graph.node[_id]['context'] = list(self.get_current_context())
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

    def extract_dependence_ids(self, node):
        # extract relevant references 
        reference_nodes = self.extract_reference_nodes(node)
        ids = set([])
        for ref_node in reference_nodes:
            _ids = self.lookup_node_ids(ref_node)
            if _ids == self.unknown_id:
                # if we don't know where it came from
                # just allocate a new node in the graph
                _ids = (self.create_node(ref_node),)
                self.update_node_id(ref_node, _ids, append=False)
            ids.update(_ids)
        return ids

    def visit_Assign(self, node):
        # Store
        assign_id = self.create_node(node)
        # definition dependencies
        depends_on = self.extract_dependence_ids(node.value)
        edges = [(dep, assign_id) for dep in depends_on]
        self.graph.add_edges_from(edges)
        # update data dependency information for 
        # assignments to this statement
        for target in node.targets:
            store_references = self.extract_reference_nodes(target)
            for ref in store_references:
                self.update_node_id(ref, assign_id, append=False)
        
    def visit_Expr(self, node):
        # Load
        expr_id = self.create_node(node)
        depends_on = self.extract_dependence_ids(node.value)
        edges = [(dep, expr_id) for dep in depends_on]
        self.graph.add_edges_from(edges)

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
        self.push_scope()
        self.push_context((node, True))
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()
        true_scope = self.pop_scope()
        
        self.push_scope()
        self.push_context((node, False))
        for stmt in node.orelse:
            self.visit(stmt)
        self.pop_context()
        false_scope = self.pop_scope()
        
        outer_scope = self.pop_scope()
        merged_scope = self.merge_scopes([outer_scope, true_scope, false_scope])
        self.push_scope(merged_scope)
        
    def visit_While(self, node):
        self.push_scope()
        self.push_context((node, True))
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()
        true_scope = self.pop_scope()
        
        self.push_scope()
        self.push_context((node, False))
        for stmt in node.orelse:
            self.visit(stmt)
        self.pop_context()
        false_scope = self.pop_scope()
        
        outer_scope = self.pop_scope()
        merged_scope = self.merge_scopes([outer_scope, true_scope, false_scope])
        self.push_scope(merged_scope)
        
    def visit_If(self, node):
        self.push_scope()
        self.push_context((node, True))
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()
        true_scope = self.pop_scope()
        
        self.push_scope()
        self.push_context((node, False))
        for stmt in node.orelse:
            self.visit(stmt)
        self.pop_context()
        false_scope = self.pop_scope()
        
        outer_scope = self.pop_scope()
        unified_scope = self.merge_scopes([outer_scope, true_scope, false_scope])
        self.push_scope(unified_scope)

    def visit_With(self, node):
        self.push_context(node)
        for stmt in node.body:
            self.visit(stmt)
        self.pop_context()


def test(src):
    constructor = DataDependenciesConstructor()
    g = constructor.run(src)
    return constructor, g
    
def draw(g):
    labels = nx.get_node_attributes(g, 'ast')
    labels = {k:unparse(v).strip() for k, v in labels.items()}
    nx.draw(g, labels=labels, node_size=100)
    plt.show()

def backward_slice(graph, seed): 
    reversed_graph = graph.reverse(copy=True)
    slice_nodes = nx.dfs_preorder_nodes(reversed_graph, seed)
    return graph.subgraph(slice_nodes)

def get_node_ids(graph, predicate):
    ids = []
    for _id, node_attributes in graph.nodes.items():
        if predicate(node_attributes):
            ids.append(_id)
    return ids
    
def backward_slices_from_expr(graph, src):
    src_node = ast.parse(src).body[0]
    ids = get_node_ids(graph, lambda attrs: ast.dump(attrs['ast']) == ast.dump(src_node))
    return [backward_slice(graph, _id) for _id in ids]
        
    
    
    

        