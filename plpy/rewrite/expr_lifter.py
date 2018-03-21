# A simple rewriter to produce a source that eliminates
# (most) non-atomic sub-expressions
# Some constucts (such as for-statements) only rewrite
# the branches, but not the entire statement, as there may be dependencies
# between rewritten body and the test condition, which complicates things

import ast
from copy import deepcopy

from astunparse import unparse


# TODO:
# remove unnecessary deepcopy
# add tests
# add documentation


class ExpressionLifter(ast.NodeTransformer):
    """
    Convert python AST to lift nested expression such that
    any subexpression is now atomic (unless one of the ignored AST node types)
    """
    def __init__(self, sym_format_name=None):
        if sym_format_name is None:
            sym_format_name = '_var%d'

        self.variable_counter = 0
        self.sym_format_name = sym_format_name
        self.atom_types = (
            ast.Name,
            ast.Num,
            ast.Str,
            ast.Bytes,
            ast.NameConstant,
            ast.Ellipsis,
            ast.Constant,
            )
        # we ignore certain types where lifting
        # would change the semantics or complicate
        # with little benefit
        self.ignore_expr_types = (
            ast.BoolOp,
            ast.Lambda,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
        )
        self.ignore_stmt_types = (
            ast.Delete,
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Pass, ast.Break, ast.Continue,
        )


    def is_atomic(self, node):
        # TODO: need to modify this further
        return isinstance(node, self.atom_types) or node is None

    def is_ignorable_expr(self, node):
        return isinstance(node, self.ignore_expr_types)

    def is_ignorable_stmt(self, node):
        return isinstance(node, self.ignore_stmt_types)

    def ignore(self, node):
        return [], node

    def alloc_symbol_name(self):
        sym = self.sym_format_name % self.variable_counter
        self.variable_counter += 1
        return sym

    def alloc_assign_node(self, rhs_node, ctx=None):
        id_allocated = self.alloc_symbol_name()
        lhs_node = ast.Name(id=id_allocated, ctx=ast.Store())
        assign_node = ast.Assign([lhs_node], rhs_node)
        assign_node = ast.copy_location(assign_node, rhs_node)
        name_node = ast.Name(id=id_allocated, ctx=ctx if ctx else ast.Load())
        return assign_node, name_node

    def lift(self, node):
        if self.is_atomic(node):
            return self.ignore(node)
        recursive_nodes = self.visit(node)
        prev_assignment_nodes = recursive_nodes[0]
        rhs_node = recursive_nodes[1]
        curr_assignment, name_node = self.alloc_assign_node(rhs_node)
        assignment_nodes = prev_assignment_nodes + [curr_assignment]
        return assignment_nodes, name_node

    def lift_list(self, nodes):
        new_nodes = []
        assignments = []
        for node in nodes:
            if not self.is_atomic(node):
                lifted_nodes = self.lift(node)
                new_nodes.append(lifted_nodes[1])
                assignments.extend(lifted_nodes[0])
            else:
                new_nodes.append(node)
        return assignments, new_nodes

    def visit_top_level_list(self, nodes):
        new_nodes = []
        for expr in nodes:
            _nn = self.visit(expr)
            try:
                new_nodes.extend(_nn)
            except TypeError:
                new_nodes.append(_nn)
        return new_nodes

    def visit(self, node):
        if self.is_ignorable_stmt(node):
            return node
        elif self.is_ignorable_expr(node):
            return self.ignore(node)
        elif self.is_atomic(node):
            return self.ignore(node)
        else:
            return super().visit(node)

    # Top-level elements: return list or single element
    # Statements
    def visit_FunctionDef(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        return new_node

    def visit_AsyncFunctionDef(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        return new_node

    def visit_ClassDef(self, node):
        new_node = deepcopy(node)
        # new_node.bases = self.visit_top_level_list(node.exprs)
        # TODO:  what are the bases, keywords here
        new_node.body = self.visit_top_level_list(node.body)
        return new_node

    def visit_Return(self, node):
        new_node = deepcopy(node)
        assignments = []

        if not self.is_atomic(node.value):
            nodes = self.lift(node.value)
            new_node.value = nodes[1]
            assignments.extend(nodes[0])

        return assignments + [new_node]

    def visit_Assign(self, node):
        new_node = deepcopy(node)
        nodes = self.visit(node.value)
        assignment_nodes = nodes[0]
        new_value = nodes[1]
        new_node.value = new_value
        return assignment_nodes + [new_node]

    def visit_AugAssign(self, node):
        new_node = deepcopy(node)
        nodes = self.visit(node.value)
        assignment_nodes = nodes[0]
        # extra for augmented assigment
        more_nodes = self.lift(nodes[1])
        assignment_nodes.extend(more_nodes[0])
        new_node.value = more_nodes[1]
        return assignment_nodes + [new_node]

    def visit_AnnAssign(self, node):
        new_node = deepcopy(node)
        nodes = self.visit(node.value)
        assignment_nodes = nodes[0]
        new_node.value = nodes[1]
        return assignment_nodes + [new_node]

    def visit_For(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        new_node.orelse = self.visit_top_level_list(node.orelse)
        return new_node

    def visit_AsyncFor(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        new_node.orelse = self.visit_top_level_list(node.orelse)
        return new_node

    def visit_While(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        new_node.orelse = self.visit_top_level_list(node.orelse)
        return new_node

    def visit_If(self, node):
        new_node = deepcopy(node)
        assignment_nodes = []

        test_nodes = self.visit(node.test)
        assignment_nodes.extend(test_nodes[0])
        new_node.test = test_nodes[1]

        new_node.body = self.visit_top_level_list(node.body)
        new_node.orelse = self.visit_top_level_list(node.orelse)

        return assignment_nodes + [new_node]

    def visit_With(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        return new_node

    def visit_AsynchWith(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        return new_node

    def visit_Raise(self, node):
        new_node = deepcopy(node)
        assignment_nodes = []

        exc_nodes = self.visit(node.exc)
        assignment_nodes.extend(exc_nodes[0])
        new_node.exc = exc_nodes[1]

        cause_nodes = self.visit(node.cause)
        assignment_nodes.extend(cause_nodes[0])
        new_node.cause = cause_nodes[1]

        return assignment_nodes + [new_node]

    def visit_Try(self, node):
        new_node = deepcopy(node)
        new_node.body = self.visit_top_level_list(node.body)
        new_node.orelse = self.visit_top_level_list(node.orelse)
        new_node.finalbody = self.visit_top_level_list(node.finalbody)
        return new_node

    def visit_Assert(self, node):
        new_node = deepcopy(node)
        assignment_nodes = []

        test_nodes = self.visit(node.test)
        assignment_nodes.extend(test_nodes[0])
        new_node.test = test_nodes[1]

        return assignment_nodes + [new_node]

    # Expressions
    def visit_Expr(self, node):
        nodes = self.visit(node.value)
        # need to wrap the final node in expr
        other_nodes = nodes[0]
        value_node = nodes[1]
        expr_node = ast.Expr(value=value_node)
        return other_nodes + [expr_node]

    # Non-top level
    # Return tuples
    def visit_BinOp(self, node):
        new_node = deepcopy(node)
        assignments = []

        left_nodes = self.lift(node.left)
        assignments.extend(left_nodes[0])
        new_node.left = left_nodes[1]

        right_nodes = self.lift(node.right)
        assignments.extend(right_nodes[0])
        new_node.right = right_nodes[1]

        return assignments, new_node

    def visit_UnaryOp(self, node):
        new_node = deepcopy(node)
        assignments = []

        operand_nodes = self.lift(node.operand)
        assignments = operand_nodes[0]
        new_node.operand = operand_nodes[1]

        return assignments, new_node

    def visit_IfExp(self, node):
        new_node = deepcopy(node)
        assignments = []

        test_nodes = self.lift(node.test)
        new_node.test = test_nodes[1]
        assignments.extend(test_nodes[0])

        return assignments, new_node

    def visit_Compare(self, node):
        new_node = deepcopy(node)
        assignments = []

        left_nodes = self.lift(node.left)
        assignments.extend(left_nodes[0])
        new_node.left = left_nodes[1]

        comparators_nodes = self.lift_list(node.comparators)
        assignments.extend(comparators_nodes[0])
        new_node.comparators = comparators_nodes[1]

        return assignments, new_node

    def visit_Call(self, node):
        new_node = deepcopy(node)
        assignments = []

        func_nodes = self.lift(node.func)
        assignments.extend(func_nodes[0])
        new_node.func = func_nodes[1]

        arg_nodes = self.lift_list(node.args)
        assignments.extend(arg_nodes[0])
        new_node.args = arg_nodes[1]

        kw_nodes = self.lift_list(node.keywords)
        assignments.extend(kw_nodes[0])
        new_node.keywords = kw_nodes[1]

        return assignments, new_node

    def visit_keyword(self, node):
        new_node = deepcopy(node)
        assignments = []

        value_nodes = self.lift(node.value)
        assignments.extend(value_nodes[0])
        new_node.value = value_nodes[1]

        return assignments, new_node

    def visit_JoinedStr(self, node):
        new_node = deepcopy(node)
        assignments = []

        values_nodes = self.lift_list(node.values)
        assignments.extend(values_nodes[0])
        new_node.values = values_nodes[1]

        return assignments, new_node

    def visit_Attribute(self, node):
        new_node = deepcopy(node)
        assignments = []

        value_nodes = self.lift(node.value)
        assignments.extend(value_nodes[0])
        new_node.value = value_nodes[1]

        return assignments, new_node

    def visit_Subscript(self, node):
        new_node = deepcopy(node)
        assignments = []

        value_nodes = self.lift(node.value)
        assignments.extend(value_nodes[0])
        new_node.value = value_nodes[1]

        slice_nodes = self.visit(node.slice)
        assignments.extend(slice_nodes[0])
        new_node.slice = slice_nodes[1]

        return assignments, new_node

    def visit_Index(self, node):
        new_node = deepcopy(node)
        assignments = []

        value_nodes = self.lift(node.value)
        assignments.extend(value_nodes[0])
        new_node.value = value_nodes[1]

        return assignments, new_node

    def visit_Starred(self, node):
        new_node = deepcopy(node)
        assignments = []

        value_nodes = self.lift(node.value)
        assignments.extend(value_nodes[0])
        new_node.value = value_nodes[1]

        return assignments, new_node

    def visit_List(self, node):
        new_node = deepcopy(node)
        assignments = []

        elts_nodes = self.lift_list(node.elts)
        assignments.extend(elts_nodes[0])
        new_node.elts = elts_nodes[1]

        return assignments, new_node

    def visit_Tuple(self, node):
        new_node = deepcopy(node)
        assignments = []

        elts_nodes = self.lift_list(node.elts)
        assignments.extend(elts_nodes[0])
        new_node.elts = elts_nodes[1]

        return assignments, new_node

    def visit_Set(self, node):
        new_node = deepcopy(node)
        assignments = []

        elts_nodes = self.lift_list(node.elts)
        assignments.extend(elts_nodes[0])
        new_node.elts = elts_nodes[1]

        return assignments, new_node

    def visit_Dict(self, node):
        new_node = deepcopy(node)
        assignments = []

        keys_nodes = self.lift_list(node.keys)
        assignments.extend(keys_nodes[0])
        new_node.keys = keys_nodes[1]

        values_nodes = self.lift_list(node.values)
        assignments.extend(values_nodes[0])
        new_node.values = values_nodes[1]

        return assignments, new_node

    def visit_Slice(self, node):
        new_node = deepcopy(node)
        assignments = []

        lower_nodes = self.lift(node.lower)
        assignments.extend(lower_nodes[0])
        new_node.lower = lower_nodes[1]

        upper_nodes = self.lift(node.upper)
        assignments.extend(upper_nodes[0])
        new_node.upper = upper_nodes[1]

        step_nodes = self.lift(node.step)
        assignments.extend(step_nodes[0])
        new_node.step = step_nodes[1]

        return assignments, new_node

    def visit_ExtSlice(self, node):
        new_node = deepcopy(node)
        assignments = []

        dims_nodes = self.lift_list(node.dims)
        assignments.extend(dims_nodes[0])
        new_node.dims = dims_nodes[1]

        return assignments, new_node

    def visit_FormattedValue(self, node):
        new_node = deepcopy(node)
        assignments = []

        value_nodes = self.lift(node.value)
        assignments.extend(value_nodes[0])
        new_node.value = value_nodes[1]

        return assignments, new_node


def lift_expressions(tree):
    if not isinstance(tree, ast.Module):
        tree = ast.parse(tree)
    else:
        tree = deepcopy(tree)
    return ExpressionLifter().visit(tree)







