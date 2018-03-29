import ast
import astunparse
import sys
import textwrap

import numpy as np
import pytest

from plpy.analyze.dynamic_trace_events import *
from plpy.analyze import dynamic_data_dependencies as d3


class BasicTracer(object):
    def __init__(self, fun, trace_lines=False, trace_inside_call=True):
        self.fun = fun
        self.trace_lines = trace_lines
        self.trace_inside_call = trace_inside_call
        self.frame_acc = []
        self.result_acc = []
        self.orig_tracer = None

    def trace(self, frame, event, arg):
        if event == 'call' or (event == 'line' and self.trace_lines):
            self.frame_acc.append(frame)
            self.result_acc.append(self.fun(frame))
            if self.trace_inside_call:
                return self.trace
            else:
                return None

    def setup(self):
        self.orig_tracer = sys.gettrace()
        sys.settrace(self.trace)

    def shutdown(self):
        sys.settrace(self.orig_tracer)

    def clear(self):
        self.frame_acc = []
        self.result_acc = []

    def __enter__(self):
       self.setup()

    def __exit__(self, type, value, traceback):
        self.shutdown()


# test helpers
def test_to_ast_node():
    node = d3.to_ast_node('x + 2')
    assert astunparse.unparse(node).strip() == '(x + 2)'

def test_get_caller_frame():
    def f():
        return

    def g():
        return f()

    tracer = BasicTracer(lambda x: x)
    with tracer:
        g()

    g_frame = tracer.result_acc[0]
    f_frame = tracer.result_acc[1]
    assert d3.get_caller_frame(f_frame) == g_frame


def get_basic_function():
    def f():
        return
    tracer = BasicTracer(d3.get_function_obj)
    with tracer:
        f()
    return tracer.result_acc[0], f

def get_basic_method():
    class BasicClass(object):
        def f(self):
            return
    val = BasicClass()
    tracer = BasicTracer(d3.get_function_obj)
    with tracer:
        val.f()
    return tracer.result_acc[0], val.f

def get_basic_static_method():
    class BasicClass(object):
        @staticmethod
        def f():
            return
    tracer = BasicTracer(d3.get_function_obj)
    with tracer:
        BasicClass.f()
    return tracer.result_acc[0], BasicClass.f

def get_basic_nested_function():
    def g():
        def f():
            return
        f()
        return f
    tracer = BasicTracer(d3.get_function_obj)
    with tracer:
        res = g()
    return tracer.result_acc[1], res

@pytest.mark.parametrize('get_func', [get_basic_function, get_basic_method, get_basic_static_method, get_basic_nested_function])
def test_get_function_obj(get_func):
    fetched, fun = get_func()
    assert fetched == fun, 'Failed to retrieve appropriate function object'

def test_get_function_unqual_name():
    def f():
        return
    tracer = BasicTracer(d3.get_function_unqual_name)
    with tracer:
        f()
    assert tracer.result_acc[0] == 'f'

class BasicDummyClass(object):
    def m(self):
        return
    @staticmethod
    def s():
        return

def test_get_function_qual_name():
    tracer = BasicTracer(d3.get_function_qual_name)
    val = BasicDummyClass()
    with tracer:
        val.m()
        val.s()
    assert tracer.result_acc[0] == 'BasicDummyClass.m', 'Qualified method name'
    assert tracer.result_acc[1] == 'BasicDummyClass.s', 'Qualified static method name'

@pytest.mark.parametrize('node_str,full_expected,all_but_first_expected', [('a', ['a'], []), ('a.b.c', ['a', 'a.b', 'a.b.c'], ['a', 'a.b'])])
def test_extract_references(node_str, full_expected, all_but_first_expected):
    node = ast.parse(node_str)
    refs = set(d3.get_nested_references(node))
    refs_but_first = set(d3.get_nested_references(node, exclude_first=True))

    full_expected = set(full_expected)
    all_but_first_expected = set(all_but_first_expected)

    assert refs == full_expected, 'References do not match'
    assert refs_but_first == all_but_first_expected, 'References do not match'

def test_register_assignment_stubs():
    stubber = d3.AddMemoryUpdateStubs('_stub')
    src = "x = 1; y = f(); z += 1"
    expected = "x = 1; _stub(['x'], [x]); y = f(); _stub(['y'], [y]); z += 1; _stub(['z'], [z])"
    with_stubs = stubber.visit(ast.parse(src))
    assert ast.dump(with_stubs) == ast.dump(ast.parse(expected))

def test_is_stub_call():
    tracer = BasicTracer(d3.get_function_obj)
    with tracer:
        d3.memory_update_stub(['var'], [10])
    assert d3.is_stub_call(tracer.result_acc[0]), 'Calling a stub function should yield true'

    tracer.result_acc = []
    def f():
        return
    with tracer:
        f()
    assert not d3.is_stub_call(tracer.result_acc[0]), 'Calling non-stub function should yield false'

@pytest.mark.parametrize(
    '_input,expected',
    [
        ('a = 10',           []                   ),
        ('a = x * 10',       ['x']                ),
        ('a = a.b * 10',     ['a', 'a.b']         ),
        ('a = b[0] * 10',    ['b']                ),
        ('a = b[0][1] * 10', ['b']                ),
        ('a = b[1:2] * 10',  ['b']                ),
        ('a.b.c = x * 10',   ['x', 'a', 'a.b']    ),
        ('a[0][1] = c.d[0]', ['a', 'c', 'c.d']    ),
        ('a.b.c[0] = 10',    ['a', 'a.b', 'a.b.c']),
        ('x = {a:1, b.c:2}',   ['a', 'b', 'b.c'])
    ]
)
def test_get_load_references_from_line(_input, expected):
    tracer = d3.DynamicDataTracer()
    refs = tracer.get_load_references_from_line(_input)
    assert refs == set(expected), 'Load references do not match'

def test_function_defined_by_user():
    # make tracer think current file is user file
    tracer = d3.DynamicDataTracer()
    tracer.file_path = __file__
    helper = BasicTracer(tracer._defined_by_user)
    with helper:
        def f():
            return 2
        f()
    assert helper.result_acc[0], 'Function was defined by user in test file'

    helper.clear()
    with helper:
        # calling any function that defined by us in this file
        ast.parse('1')
    assert not helper.result_acc[0], 'Function was not defined by user in this test file'

def test_function_called_by_user():
    tracer = d3.DynamicDataTracer()
    tracer.file_path = __file__
    helper = BasicTracer(tracer._called_by_user, trace_inside_call=True)
    with helper:
        np.max([1, 2])
    assert helper.result_acc[0] and (not helper.result_acc[1]), 'First is call made by user, second is not (its call to np._amax in np source)'

def standardize_source(src):
    return astunparse.unparse(ast.parse(src))

def check_memory_update(event, updates):
    assert isinstance(event, MemoryUpdate)
    assert set(event.mem_locs.keys()) == set(updates)

def check_exec_line(event, line, refs_loaded):
    assert isinstance(event, ExecLine)
    assert standardize_source(event.line) == standardize_source(line)
    assert set(event.uses_mem_locs.keys()) == set(refs_loaded)

def check_enter_call(event, qualname, call_args, is_method):
    assert isinstance(event, EnterCall)
    print(event.details)
    assert event.details['qualname'] == qualname
    assert set(event.details['abstract_call_args'].keys()) == set(call_args)
    assert event.details['is_method'] == is_method

def check_exit_call(event, details=None):
    assert isinstance(event, ExitCall)

def make_event_check(fun, *args, **kwargs):
    return lambda x: fun(x, *args, **kwargs)

def basic_case_1():
        src = """
            def f(x, y):
                return x + y
            x = 10
            y = 20
            z = f(x, y)
        """

        expected_event_checks = [
            # x = 10
            make_event_check(check_exec_line, line='x = 10', refs_loaded=[]),
            make_event_check(check_memory_update, updates=['x']),
            # y = 20
            make_event_check(check_exec_line, line='y = 20', refs_loaded=[]),
            make_event_check(check_memory_update, updates=['y']),
            # z = f(x, y)
            make_event_check(check_exec_line, line='z = f(x, y)', refs_loaded=['f', 'x', 'y']),
            make_event_check(check_enter_call, qualname='f', call_args=['x', 'y'], is_method=False),
            make_event_check(check_exec_line, line='return x + y', refs_loaded=['x', 'y']),
            make_event_check(check_exit_call),
            make_event_check(check_memory_update, updates=['z']),
        ]
        return src, expected_event_checks

def basic_case_2():
        src = """
            class A(object):
                @staticmethod
                def f(x, y):
                    return x + y
            x = 10
            y = 20
            z = A.f(x, y)
        """

        expected_event_checks = [
            # x = 10
            make_event_check(check_exec_line, line='x = 10', refs_loaded=[]),
            make_event_check(check_memory_update, updates=['x']),
            # y = 20
            make_event_check(check_exec_line, line='y = 20', refs_loaded=[]),
            make_event_check(check_memory_update, updates=['y']),
            # z = A.f(x, y)
            make_event_check(check_exec_line, line='z = A.f(x, y)', refs_loaded=['x', 'y', 'A', 'A.f']),
            # note that is_method is False as staticmethods are indistinguishable from function's in Python 3.*
            # in particular, inspect.ismethod returns False
            make_event_check(check_enter_call, qualname='A.f', call_args=['x', 'y'], is_method=False),
            make_event_check(check_exec_line, line='return x + y', refs_loaded=['x', 'y']),
            make_event_check(check_exit_call),
            make_event_check(check_memory_update, updates=['z']),
        ]
        return src, expected_event_checks

def basic_case_3():
        src = """
            class A(object):
                def __init__(self, x):
                    self.v = x

                def f(self, x, y):
                    return x + y + self.v
            x = 10
            y = 20
            obj = A(10)
            z = obj.f(x, y)
            obj.v = 200
        """

        expected_event_checks = [
            # x = 10
            make_event_check(check_exec_line, line='x = 10', refs_loaded=[]),
            make_event_check(check_memory_update, updates=['x']),
            # y = 10
            make_event_check(check_exec_line, line='y = 20', refs_loaded=[]),
            make_event_check(check_memory_update, updates=['y']),
            # obj = A(10)
            make_event_check(check_exec_line, line='obj = A(10)', refs_loaded=['A']),
            # note that the constructor call is not yet a 'method' as there is no instance bount to it at the time of function entry
            make_event_check(check_enter_call, qualname='A', call_args=set(['self', 'x']), is_method=False),
            make_event_check(check_exec_line, line='self.v = x', refs_loaded=['self', 'x']),
            # self, and self.v
            make_event_check(check_memory_update, updates=['self', 'self.v']),
            make_event_check(check_exit_call),
            make_event_check(check_memory_update, updates=['obj']),
            # z = obj.f(x, y)
            make_event_check(check_exec_line, line='z = obj.f(x, y)', refs_loaded=['obj', 'obj.f', 'x', 'y']),
            # note that is_method is False as staticmethods are indistinguishable from function's in Python 3.*
            # in particular, inspect.ismethod returns False
            make_event_check(check_enter_call, qualname='A.f', call_args=set(['self', 'x', 'y']), is_method=True),
            make_event_check(check_exec_line, line='return x + y + self.v', refs_loaded=['x', 'y', 'self', 'self.v']),
            make_event_check(check_exit_call),
            make_event_check(check_memory_update, updates=['z']),
            # obj.v = 200
            make_event_check(check_exec_line, line='obj.v = 200', refs_loaded=['obj']),
            make_event_check(check_memory_update, updates=['obj', 'obj.v']),
        ]
        return src, expected_event_checks

@pytest.mark.parametrize('_input_fun', [basic_case_1, basic_case_2, basic_case_3])
def test_basic_programs(_input_fun):
    tracer = d3.DynamicDataTracer()
    src, expected_checks = _input_fun()
    src = textwrap.dedent(src)
    tracer.run(src)

    assert len(tracer.trace_events) == len(expected_checks), 'The event and checks are mismatched.'
    for event, check in zip(tracer.trace_events, expected_checks):
        check(event)

