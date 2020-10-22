import ast
import astunparse
import sys
import textwrap

import pytest

from plpy.analyze.dynamic_trace_events import *
from plpy.analyze import dynamic_tracer as dt


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
    node = dt.to_ast_node('x + 2')
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
    assert dt.get_caller_frame(f_frame) == g_frame


def get_basic_function():
    def f():
        return

    tracer = BasicTracer(dt.get_function_obj)
    with tracer:
        f()
    return tracer.result_acc[0], f


def get_basic_method():
    class BasicClass(object):
        def f(self):
            return

    val = BasicClass()
    tracer = BasicTracer(dt.get_function_obj)
    with tracer:
        val.f()
    return tracer.result_acc[0], val.f


def get_basic_static_method():
    class BasicClass(object):
        @staticmethod
        def f():
            return

    tracer = BasicTracer(dt.get_function_obj)
    with tracer:
        BasicClass.f()
    return tracer.result_acc[0], BasicClass.f


def get_basic_nested_function():
    def g():
        def f():
            return

        f()
        return f

    tracer = BasicTracer(dt.get_function_obj)
    with tracer:
        res = g()
    return tracer.result_acc[1], res


@pytest.mark.parametrize(
    'get_func', [
        get_basic_function, get_basic_method, get_basic_static_method,
        get_basic_nested_function
    ]
)
def test_get_function_obj(get_func):
    fetched, fun = get_func()
    assert fetched == fun, 'Failed to retrieve appropriate function object'


def test_get_co_name():
    def f():
        return

    tracer = BasicTracer(dt.get_co_name)
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
    tracer = BasicTracer(dt.get_function_qual_name)
    val = BasicDummyClass()
    with tracer:
        val.m()
        val.s()
    assert tracer.result_acc[0] == 'BasicDummyClass.m', 'Qualified method name'
    assert tracer.result_acc[
        1] == 'BasicDummyClass.s', 'Qualified static method name'


@pytest.mark.parametrize(
    'node_str,full_expected,all_but_first_expected',
    [('a', ['a'], []), ('a.b.c', ['a', 'a.b', 'a.b.c'], ['a', 'a.b'])]
)
def test_extract_references(node_str, full_expected, all_but_first_expected):
    node = ast.parse(node_str)
    refs = set(dt.get_nested_references(node))
    refs_but_first = set(dt.get_nested_references(node, exclude_first=True))

    full_expected = set(full_expected)
    all_but_first_expected = set(all_but_first_expected)

    assert refs == full_expected, 'References do not match'
    assert refs_but_first == all_but_first_expected, 'References do not match'


def test_register_assignment_stubs():
    stubber = dt.AddMemoryUpdateStubs('_stub')
    src = "x = 1; y = f(); z += 1; d[1] = 2"
    expected = "x = 1; _stub(['x']); y = f(); _stub(['y']); z += 1; _stub(['z']); d[1] = 2; _stub(['d', 'd[1]'])"
    with_stubs = stubber.visit(ast.parse(src))
    assert ast.dump(with_stubs) == ast.dump(ast.parse(expected))

    src_with_imports = """
    import numpy as np
    import sklearn.preprocessing
    from sklearn import linear_models
    from sklearn import linear_models as lm
    """
    expected_with_imports = """
    import numpy as np
    _stub(['np'])
    import sklearn.preprocessing
    _stub(['sklearn.preprocessing'])
    from sklearn import linear_models
    _stub(['linear_models'])
    from sklearn import linear_models as lm
    _stub(['lm'])
    """
    src_with_imports = textwrap.dedent(src_with_imports)
    expected_with_imports = textwrap.dedent(expected_with_imports)
    with_imports_with_stubs = stubber.visit(ast.parse(src_with_imports))
    assert ast.dump(with_imports_with_stubs) == ast.dump(
        ast.parse(expected_with_imports)
    )


def test_is_stub_call():
    tracer = BasicTracer(dt.get_function_obj)
    with tracer:
        dt.memory_update_stub(['var'])
    assert dt.is_stub_call(
        tracer.result_acc[0]
    ), 'Calling a stub function should yield true'

    tracer.result_acc = []

    def f():
        return

    with tracer:
        f()
    assert not dt.is_stub_call(
        tracer.result_acc[0]
    ), 'Calling non-stub function should yield false'


@pytest.mark.parametrize(
    '_input,expected', [
        ('a = 10', []),
        ('a = x * 10', ['x']),
        ('a = a.b * 10', ['a', 'a.b']),
        ('a = b[0] * 10', ['b', 'b[0]']),
        ('a = b[0][1] * 10', ['b', 'b[0]', 'b[0][1]']),
        ('a = b[1:2] * 10', ['b', 'b[1:2]']),
        ('a.b.c = x * 10', ['x', 'a', 'a.b']),
        ('a[0][1] = c.d[0]', ['a', 'a[0]', 'c', 'c.d', 'c.d[0]']),
        ('a.b.c[0] = 10', ['a', 'a.b', 'a.b.c']),
        ('x = {a:1, b.c:2}', ['a', 'b', 'b.c']),
    ]
)
def test_get_load_references_from_line(_input, expected):
    tracer = dt.DynamicDataTracer()
    refs = tracer.get_load_references_from_line(_input)
    assert refs == set(expected), 'Load references do not match'


def test_function_defined_by_user():
    # make tracer think current file is user file
    tracer = dt.DynamicDataTracer()
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
    assert not helper.result_acc[
        0], 'Function was not defined by user in this test file'


def test_function_called_by_user():
    tracer = dt.DynamicDataTracer()
    tracer.file_path = __file__
    helper = BasicTracer(tracer._called_by_user, trace_inside_call=True)
    import pandas as pd
    with helper:
        pd.DataFrame([(1, 2)])
    assert helper.result_acc[0] and (
        not helper.result_acc[1]
    ), 'First is call made by user, second is not (its call to np._amax in np source)'


def standardize_source(src):
    return astunparse.unparse(ast.parse(src))


def check_memory_update(event, updates):
    assert isinstance(event, MemoryUpdate)
    assert set(d.name for d in event.defs) == set(updates)


def check_exec_line(event, line, refs_loaded):
    assert isinstance(event, ExecLine)
    try:
        assert standardize_source(event.line) == standardize_source(line)
    except SyntaxError:
        # somethings such as with... can't parse as asingle line
        assert event.line.strip() == line.strip()
    uses = set(u.name for u in event.uses)
    assert uses == set(refs_loaded)


def check_enter_call(event, qualname, call_args, is_method):
    assert isinstance(event, EnterCall)
    assert event.details['qualname'] == qualname
    if call_args is not None:
        abstract_call_args = event.details['abstract_call_args']
        abstract_call_args_names = set(a.name for a in abstract_call_args)
        assert abstract_call_args_names == set(call_args)
    assert event.details['is_method'] == is_method


def check_exit_call(event, co_name):
    assert isinstance(event, ExitCall)
    assert event.details['co_name'] == co_name


def check_ignore(event):
    pass


def make_event_check(fun, *args, **kwargs):
    return lambda x: fun(x, *args, **kwargs)


# function call
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
        make_event_check(
            check_exec_line, line='z = f(x, y)', refs_loaded=['f', 'x', 'y']
        ),
        make_event_check(
            check_enter_call,
            qualname='f',
            call_args=['x', 'y'],
            is_method=False
        ),
        make_event_check(
            check_exec_line, line='return x + y', refs_loaded=['x', 'y']
        ),
        make_event_check(check_exit_call, co_name='f'),
        make_event_check(check_memory_update, updates=['z']),
    ]
    return src, expected_event_checks


# static method call
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
        make_event_check(
            check_exec_line,
            line='z = A.f(x, y)',
            refs_loaded=['x', 'y', 'A', 'A.f']
        ),
        # note that is_method is False as staticmethods are indistinguishable from function's in Python 3.*
        # in particular, inspect.ismethod returns False
        make_event_check(
            check_enter_call,
            qualname='A.f',
            call_args=['x', 'y'],
            is_method=False
        ),
        make_event_check(
            check_exec_line, line='return x + y', refs_loaded=['x', 'y']
        ),
        make_event_check(check_exit_call, co_name='f'),
        make_event_check(check_memory_update, updates=['z']),
    ]
    return src, expected_event_checks


# method call
def basic_case_3():
    src = """
            import numpy as np
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
            np.max([1,2,3])
            x = 2
        """

    expected_event_checks = [
        # import numpy as np
        make_event_check(
            check_exec_line, line='import numpy as np', refs_loaded=[]
        ),
        make_event_check(check_memory_update, updates=['np']),
        # x = 10
        make_event_check(check_exec_line, line='x = 10', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['x']),
        # y = 10
        make_event_check(check_exec_line, line='y = 20', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['y']),
        # obj = A(10)
        make_event_check(
            check_exec_line, line='obj = A(10)', refs_loaded=['A']
        ),
        # note that the constructor call is not yet a 'method' as there is no instance bount to it at the time of function entry
        make_event_check(
            check_enter_call,
            qualname='A',
            call_args=['self', 'x'],
            is_method=False
        ),
        make_event_check(
            check_exec_line, line='self.v = x', refs_loaded=['self', 'x']
        ),
        # self, and self.v
        make_event_check(check_memory_update, updates=['self', 'self.v']),
        make_event_check(check_exit_call, co_name='__init__'),
        make_event_check(check_memory_update, updates=['obj']),
        # z = obj.f(x, y)
        make_event_check(
            check_exec_line,
            line='z = obj.f(x, y)',
            refs_loaded=['obj', 'obj.f', 'x', 'y']
        ),
        # note that is_method is False as staticmethods are indistinguishable from function's in Python 3.*
        # in particular, inspect.ismethod returns False
        make_event_check(
            check_enter_call,
            qualname='A.f',
            call_args=['self', 'x', 'y'],
            is_method=True
        ),
        make_event_check(
            check_exec_line,
            line='return x + y + self.v',
            refs_loaded=['x', 'y', 'self', 'self.v']
        ),
        make_event_check(check_exit_call, co_name='f'),
        make_event_check(check_memory_update, updates=['z']),
        # obj.v = 200
        make_event_check(
            check_exec_line, line='obj.v = 200', refs_loaded=['obj']
        ),
        make_event_check(check_memory_update, updates=['obj', 'obj.v']),
        make_event_check(
            check_exec_line,
            line='np.max([1,2,3])',
            refs_loaded=['np', 'np.max']
        ),
        make_event_check(
            check_enter_call,
            qualname='amax',
            call_args=[],
            #call_args=['a', 'axis', 'out', 'keepdims'],
            is_method=False
        ),
        make_event_check(check_exit_call, co_name='amax'),
        make_event_check(check_exec_line, line='x = 2', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['x']),
    ]
    return src, expected_event_checks


# use of with (enter/exit)
def basic_case_4():
    src = """
            class A(object):
                def __init__(self, val):
                    self.val = val

                def __enter__(self):
                    return self

                def __exit__(self, type, value, traceback):
                    pass

            orig = 10
            with A(orig) as v:
                x = v.val
            max([x, 2])
            w = 10

            import pandas as pd
            df = pd.DataFrame([(1, 2), (3, 4)], columns=['c1', 'c2'])
            df.max()
        """

    expected_event_checks = [
        make_event_check(check_exec_line, line='orig = 10', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['orig']),
        make_event_check(
            check_exec_line,
            line='with A(orig) as v:',
            refs_loaded=['A', 'orig']
        ),
        # inside __init__
        make_event_check(
            check_exec_line,
            line='self.val = val',
            refs_loaded=['self', 'val']
        ),
        make_event_check(check_memory_update, updates=['self.val', 'self']),
        # inside __return__ as defined by user
        make_event_check(
            check_exec_line, line='return self', refs_loaded=['self']
        ),
        make_event_check(check_memory_update, updates=['v']),
        make_event_check(
            check_exec_line, line='x = v.val', refs_loaded=['v', 'v.val']
        ),
        make_event_check(check_memory_update, updates=['x']),
        make_event_check(
            check_exec_line, line='max([x, 2])', refs_loaded=['x', 'max']
        ),
        make_event_check(check_exec_line, line='w = 10', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['w']),
        make_event_check(
            check_exec_line, line='import pandas as pd', refs_loaded=[]
        ),
        make_event_check(check_memory_update, updates=['pd']),
        make_event_check(
            check_exec_line,
            line="df = pd.DataFrame([(1, 2), (3, 4)], columns=['c1', 'c2'])",
            refs_loaded=['pd', 'pd.DataFrame']
        ),
        # ignore call_args too many....
        make_event_check(
            check_enter_call,
            qualname='DataFrame',
            call_args=None,
            is_method=False
        ),
        make_event_check(check_exit_call, co_name='__init__'),
        make_event_check(check_memory_update, updates=['df']),
        make_event_check(
            check_exec_line, line='df.max()', refs_loaded=['df', 'df.max']
        ),
        make_event_check(
            check_enter_call,
            qualname='DataFrame.max',
            call_args=None,
            is_method=True
        ),
        # this matches the co_name for DataFrame.max
        make_event_check(check_exit_call, co_name='stat_func'),
    ]
    return src, expected_event_checks


# call to a C function
def basic_case_5():
    src = """
        import numpy as np
        v = [1,2,3]
        v_log = np.log(v)
        other = 100
        """

    expected_event_checks = [
        make_event_check(
            check_exec_line, line='import numpy as np', refs_loaded=[]
        ),
        make_event_check(check_memory_update, updates=['np']),
        make_event_check(check_exec_line, line='v = [1,2,3]', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['v']),
        make_event_check(
            check_exec_line,
            line='v_log = np.log(v)',
            refs_loaded=['np', 'np.log', 'v']
        ),
        # no enter/exit calls as C function
        make_event_check(check_memory_update, updates=['v_log']),
        make_event_check(check_exec_line, line='other = 100', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['other']),
    ]
    return src, expected_event_checks


# non-local variable in function and using a function closure
def basic_case_6():
    src = """
    _times = 2
    def mult(x):
        return x * _times
    mult(10)

    def with_closure():
        x = 100
        def times_100(y):
            return y * x
        return times_100

    f = with_closure()
    f(10)
    """

    expected_event_checks = [
        make_event_check(check_exec_line, line='_times = 2', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['_times']),
        make_event_check(
            check_exec_line, line='mult(10)', refs_loaded=['mult', '_times']
        ),  #_times loaded indirectly by mult call
        make_event_check(
            check_enter_call,
            qualname='mult',
            call_args=['x'],
            is_method=False
        ),
        make_event_check(
            check_exec_line,
            line='return x * _times',
            refs_loaded=['x', '_times']
        ),
        make_event_check(check_exit_call, co_name='mult'),
        make_event_check(
            check_exec_line,
            line='f = with_closure()',
            refs_loaded=['with_closure']
        ),
        make_event_check(
            check_enter_call,
            qualname='with_closure',
            call_args=[],
            is_method=False
        ),
        make_event_check(check_exec_line, line='x = 100', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['x']),
        make_event_check(
            check_exec_line,
            line='return times_100',
            refs_loaded=['times_100']
        ),
        make_event_check(check_exit_call, co_name='with_closure'),
        make_event_check(check_memory_update, updates=['f']),
        make_event_check(check_exec_line, line='f(10)', refs_loaded=['f']),
        make_event_check(
            check_enter_call,
            qualname='with_closure.<locals>.times_100',
            call_args=['y'],
            is_method=False
        ),
        make_event_check(
            check_exec_line, 'return y * x', refs_loaded=['x', 'y']
        ),
        make_event_check(check_exit_call, co_name='times_100'),
    ]
    return src, expected_event_checks


# call user code from non-user code
def basic_case_7():
    src = """
    import pandas as pd
    s = pd.Series([1, 2, 3])
    def g():
        return 2
    def f(x):
        g()
        return x * 2
    f(10)
    s.apply(f)
    s[0] = 100
    """

    expected_event_checks = [
        make_event_check(
            check_exec_line, line='import pandas as pd', refs_loaded=[]
        ),
        make_event_check(check_memory_update, updates=['pd']),
        make_event_check(
            check_exec_line,
            line='s = pd.Series([1,2,3])',
            refs_loaded=['pd', 'pd.Series']
        ),
        make_event_check(
            check_enter_call,
            qualname='Series',
            call_args=None,
            is_method=False
        ),
        make_event_check(check_exit_call, co_name='__init__'),
        make_event_check(check_memory_update, updates=['s']),
        make_event_check(
            check_exec_line, line='f(10)', refs_loaded=['f', 'g']
        ),  # g loaded indirectly by f
        make_event_check(
            check_enter_call, qualname='f', call_args=['x'], is_method=False
        ),
        make_event_check(check_exec_line, line='g()', refs_loaded=['g']),
        make_event_check(
            check_enter_call, qualname='g', call_args=[], is_method=False
        ),
        make_event_check(check_exec_line, line='return 2', refs_loaded=[]),
        make_event_check(check_exit_call, co_name='g'),
        make_event_check(
            check_exec_line, line='return x * 2', refs_loaded=['x']
        ),
        make_event_check(check_exit_call, co_name='f'),
        make_event_check(
            check_exec_line,
            line='s.apply(f)',
            refs_loaded=['s', 's.apply', 'f', 'g']
        ),  # g loaded indirectly by f
        # note that there is no entries for the f calls inside apply
        make_event_check(
            check_enter_call,
            qualname='Series.apply',
            call_args=None,
            is_method=True
        ),
        make_event_check(check_exit_call, co_name='apply'),
        make_event_check(
            check_exec_line, line='s[0] = 100', refs_loaded=['s']
        ),
        make_event_check(check_memory_update, updates=['s', 's[0]']),
    ]

    return src, expected_event_checks


# assignment that calls __setitem__, make sure we don't trigger that
def basic_case_8():
    src = """
    d = {1:2, 2:3}
    d[1] = 1000
    """

    expected_event_checks = [
        make_event_check(
            check_exec_line, line='d = {1:2, 2:3}', refs_loaded=[]
        ),
        make_event_check(check_memory_update, updates=['d']),
        make_event_check(
            check_exec_line, line='d[1] = 1000', refs_loaded=['d']
        ),
        make_event_check(check_memory_update, updates=['d', 'd[1]']),
    ]

    return src, expected_event_checks


# bounded loop
def basic_case_9():
    src = """
    class A(object):
        def __init__(self, v):
            self.v = 10

    x = 0
    while x < 10:
        for i in range(10):
            i
        x += 1

    a = A(1)
    a.v += 1

    [x for x in range(10)]
    {x for x in range(10)}
    """
    loop_bound = 2
    expected_event_checks = [
        make_event_check(check_exec_line, line='x = 0', refs_loaded=[]),
        make_event_check(check_memory_update, updates=['x']),
        # bounded to two
        make_event_check(check_exec_line, line='i', refs_loaded=['i']),
        make_event_check(check_exec_line, line='i', refs_loaded=['i']),
        make_event_check(check_exec_line, line='x += 1', refs_loaded=['x']),
        make_event_check(check_memory_update, updates=['x']),
        # bounded to two
        make_event_check(check_exec_line, line='i', refs_loaded=['i']),
        make_event_check(check_exec_line, line='i', refs_loaded=['i']),
        make_event_check(check_exec_line, line='x += 1', refs_loaded=['x']),
        make_event_check(check_memory_update, updates=['x']),
        # done with outer loop
        # outside
        make_event_check(check_exec_line, line='a = A(1)', refs_loaded=['A']),
        make_event_check(check_ignore),  # enter call
        make_event_check(check_ignore),  # line
        make_event_check(check_ignore),  # memory update
        make_event_check(check_ignore),  # exit call
        make_event_check(check_ignore),  # memory update
        make_event_check(
            check_exec_line, line='a.v += 1', refs_loaded=['a', 'a.v']
        ),
        make_event_check(check_memory_update, updates=['a', 'a.v']),
        make_event_check(
            check_exec_line,
            line='[x for x in range(10)]',
            refs_loaded=['x', 'range']
        ),
        make_event_check(
            check_exec_line,
            line='{x for x in range(10)}',
            refs_loaded=['x', 'range']
        ),
    ]
    return src, expected_event_checks, loop_bound


# use a global variables in user function that is called from third party function
def basic_case_10():
    src = """
        import pandas as pd
        df = pd.DataFrame([(1, 2), (3, 4)], columns=['c1', 'c2'])

        extra = 10
        other_extra = 100

        def g(y):
            return other_extra

        # dependency on global extra
        def add(x):
            return g(x) + extra

        df['c2'].apply(add)
    """

    # line, memupdate, line, call, return, memupdate, line, memupdate, line, memupdate
    expected_event_checks = [make_event_check(check_ignore)] * 10
    expected_event_checks += [
        # note that extra is also implicitly loaded as it is used by add. add is called inside apply, which we don't instrument, so
        # we add it at the ExecLine event
        # so the indirect references are: extra, g, other_extra
        make_event_check(
            check_exec_line,
            line="df['c2'].apply(add)",
            refs_loaded=[
                'df', "df['c2']", "df['c2'].apply", 'add', 'extra', 'g',
                'other_extra'
            ]
        )
    ]
    # call, return
    expected_event_checks += [make_event_check(check_ignore)] * 2
    return src, expected_event_checks


# an empty program should still work rather than crash
def basic_case_11():
    src = " "
    expected_event_checks = []
    return src, expected_event_checks


# use of comprehensions
def basic_case_12():
    src = """
    [a for a in range(3)]
    {a for a in range(4)}
    d = {'a':'b', 'c':'d'}
    { k:v for k, v in d.items()}
    """
    expected_event_checks = [
        make_event_check(
            check_exec_line,
            line='[a for a in range(3)]',
            refs_loaded=['range', 'a']
        ),
        make_event_check(
            check_exec_line,
            line='{a for a in range(4)}',
            refs_loaded=['range', 'a']
        ),
        make_event_check(check_ignore),
        make_event_check(check_ignore),
        make_event_check(
            check_exec_line,
            line="{k:v for k, v in d.items()}",
            refs_loaded=['k', 'v', 'd', 'd.items']
        ),
    ]
    return src, expected_event_checks


basic_cases = [
    basic_case_1,
    basic_case_2,
    basic_case_3,
    basic_case_4,
    basic_case_5,
    basic_case_6,
    basic_case_7,
    basic_case_8,
    basic_case_9,
    basic_case_10,
    basic_case_11,
    basic_case_12,
]


@pytest.mark.parametrize('_input_fun', basic_cases)
def test_basic_programs(_input_fun):
    test_inputs = _input_fun()
    if len(test_inputs) == 2:
        src, expected_checks = test_inputs
        loop_bound = None
    else:
        src, expected_checks, loop_bound = test_inputs

    tracer = dt.DynamicDataTracer(loop_bound=loop_bound)
    src = textwrap.dedent(src)
    tracer.run(src)

    print(list(map(str, tracer.trace_events)))
    assert len(
        tracer.trace_events
    ) == len(expected_checks), 'The event and checks are mismatched.'
    for event, check in zip(tracer.trace_events, expected_checks):
        check(event)
