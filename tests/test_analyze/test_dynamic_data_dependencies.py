import inspect
import astunparse
import pytest
from plpy.analyze import dynamic_data_dependencies as d3
import sys


class BasicTracer(object):
    def __init__(self, fun):
        self.fun = fun
        self.frame_acc = []
        self.result_acc = []
        self.orig_tracer = None
        
    def basic_trace_call(self, frame, event, arg):
        if event == 'call':
            self.frame_acc.append(frame)
            self.result_acc.append(self.fun(frame))
            return None
            
    def __enter__(self):
        self.orig_tracer = sys.gettrace()
        sys.settrace(self.basic_trace_call)
        
    def __exit__(self, type, value, traceback):
        sys.settrace(self.orig_tracer)


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
    

