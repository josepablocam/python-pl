import ast
import astunparse
import pytest
from plpy.analyze import dynamic_data_dependencies as d3
import sys


class BasicTracer(object):
    def __init__(self, fun, trace_lines=False):
        self.fun = fun
        self.trace_lines = trace_lines
        self.frame_acc = []
        self.result_acc = []
        self.orig_tracer = None
        
    def trace(self, frame, event, arg):
        if event == 'call' or (event == 'line' and self.trace_lines):
            self.frame_acc.append(frame)
            self.result_acc.append(self.fun(frame))
            return None
            
    def setup(self):
        self.orig_tracer = sys.gettrace()
        sys.settrace(self.trace)
        
    def shutdown(self):
        sys.settrace(self.orig_tracer)

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
    
@pytest.mark.parametrize('node_str,expected', [('a', 'a'), ('a.b.c', ['a', 'a.b', 'a.b.c'])])
def test_extract_references(node_str, expected):
    node = ast.parse(node_str)
    refs = d3.ExtractReferences().run(node)
    expected = set(expected)
    refs = set(refs)
    assert refs.symmetric_difference(expected) == set(), 'References do not match'
    

def test_register_assignment_stubs():
    stubber = d3.AddMemoryUpdateStubs('_stub')
    src = "x = 1; y = f(); z += 1"
    expected = "x = 1; _stub([x]); y = f(); _stub([y]); z += 1; _stub([z])"
    with_stubs = stubber.visit(ast.parse(src))
    assert ast.dump(with_stubs) == ast.dump(ast.parse(expected))
    
def test_is_stub_call():
    tracer = BasicTracer(d3.get_function_obj)
    with tracer:
        d3.memory_update_stub([10])
    assert d3.is_stub_call(tracer.result_acc[0]), 'Calling a stub function should yield true'
    
    tracer.result_acc = []
    def f():
        return
    with tracer:
        f()
    assert not d3.is_stub_call(tracer.result_acc[0]), 'Calling non-stub function should yield false'
        
    

# test loading memory locations in x = e, x.b = 100 => (x)
# check if a function is called by a user
# check if a function is defined by a user
# test small programs w/o external libs
# test small programs w/ external libs
# plan how to add control flow to this...

