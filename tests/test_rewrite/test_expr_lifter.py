import ast
import astunparse
import textwrap

import pytest

from plpy.rewrite.expr_lifter import ExpressionLifter, lift_expressions


def test_symbol_format():
    orig = ExpressionLifter(sym_format_name=None)
    assert orig.alloc_symbol_name() == '_var0'
    diff = ExpressionLifter(sym_format_name='other_var%d_')
    assert diff.alloc_symbol_name(
    ) == 'other_var0_', 'Symbol format should be taken from args'


def test_symbol_increases():
    lifter = ExpressionLifter(sym_format_name='%d')
    assert lifter.alloc_symbol_name() == '0'
    assert lifter.alloc_symbol_name() == '1', 'Symbol counter should increase'


def lift_atomic(expr):
    lifted = lift_expressions(expr)
    assert ast.dump(expr) == ast.dump(lifted)


def lift_non_atomic(orig, expected):
    lifted = lift_expressions(orig)
    lifted_src = astunparse.unparse(lifted)
    lifted = ast.parse(lifted)
    assert ast.dump(lifted) == ast.dump(expected)


def lift_twice(expr):
    lifted = lift_expressions(expr)
    relifted = lift_expressions(lifted)
    assert ast.dump(lifted) == ast.dump(relifted)


# name, atomic, non-atomic, expected
expr_test_cases = [
    # no case of slice that is atomic...we translate `:` syntax to slice() in a preprocessing step
    (
        'ExtSlice', '', 'x[1:2:g()]',
        '_var0 = g(); _var1 = slice(1, 2, _var0); x[_var1]'
    ),
    (
        'Slice', '', 'x[(1 + 2):3]',
        '_var0 = 1 + 2; _var1 = slice(_var0, 3, None); x[_var1]'
    ),
    (
        'Dict', '{1:10, 2:30}', '{(1+2): 3, 4: 2+g()}',
        '_var0 = 1+2; _var1 = g(); _var2 = 2+_var1; {_var0: 3, 4:_var2}'
    ),
    ('Set', '{1,2}', '{1, g()}', '_var0 = g(); {1, _var0}'),
    ('Tuple', '(1, 2)', '(1, g())', '_var0 = g(); (1, _var0)'),
    ('List', '[1, 2, 3]', '[1, g(), 3]', '_var0 = g(); [1, _var0, 3]'),
    ('Starred', '*x', '*f()', '_var0 = f(); *_var0'),
    ('Index', 'x[1]', 'x[1 + 2]', '_var0 = 1 + 2; _var1 = _var0; x[_var1]'),
    ('Subscript', 'x[1]', 'x[1][2]', '_var0 = x[1]; _var0[2]'),
    ('Attribute', 'a.b', 'a.b.c', '_var0 = a.b; _var0.c'),
    (
        'Call', 'f(1, 2, c=1)', 'f(h(), 2, c=g())',
        '_var0 = h(); _var1 = g(); f(_var0, 2, c=_var1)'
    ),
    (
        'Compare', '1 < c < 2', '1 * 5 < c < 2 + 4',
        '_var0 = 1 * 5; _var1 = 2 + 4; _var0 < c < _var1'
    ),
    (
        'IfExp', '1 if a else 2', '1 if a + 1 > 2 else 2',
        '_var0 = a + 1; _var1 = _var0 > 2; 1 if _var1 else 2'
    ),
    ('UnaryOp', '-b', '-(f() * 2)', '_var0 = f(); _var1 = _var0 * 2; -_var1'),
    ('BinOp', '1 + 2', '1 + g()', '_var0 = g(); 1 + _var0'),
]

stmt_test_cases = [
    (
        'Try', """
        try:
            pass
        except AttributeError:
            g()
        """, """
        try:
            g() + h()
        except AttributeError:
            z() * 2
        else:
            w + a.b
        finally:
            y(w())
        """, """
        try:
            _var0 = g()
            _var1 = h()
            _var0 + _var1
        except AttributeError:
            _var2 = z()
            _var2 * 2
        else:
            _var3 = a.b
            w + _var3
        finally:
            _var4 = w()
            y(_var4)
        """
    ),
    (
        'Raise', 'raise Ok', 'raise a.b.c()',
        '_var0 = a.b; _var1 = _var0.c; raise _var1()'
    ),
    (
        'If', """
        if x:
            2
        else:
            3
        """, """
        if x + 2 < 3:
            f() * g()
        elif c + d + w:
            z * 2 + h()
        else:
            w() + 2
        """, """
        _var0 = x + 2
        if _var0 < 3:
            _var1 = f()
            _var2 = g()
            _var1 * _var2
        else:
            _var3 = c + d
            if _var3 + w:
                _var4 = z * 2
                _var5 = h()
                _var4 + _var5
            else:
                _var6 = w()
                _var6 + 2
        """
    ),
    (
        'While', """
        while x:
            2
    """, """
        while x + g(h()):
            f(h())
    """, """
        while x + g(h()):
            _var0 = h()
            f(_var0)
    """
    ),
    (
        'For', """
        for y in x:
            2
    """, """
        for y in g(h()):
            f(g())
    """, """
        for y in g(h()):
            _var0 = g()
            f(_var0)
    """
    ), ('Assign', 'a = 1', 'a = g(h())', '_var0 = h(); a = g(_var0)'),
    (
        'Slicing Extra', """
    # nothing
    """, """
    m[:, 1]
    """, """
    _var0 = slice(None, None, None)
    _var1 = (_var0, 1)
    m[_var1]
    """
    )
]


@pytest.mark.parametrize(
    "expr_type,atomic_case,non_atomic_case,expected",
    expr_test_cases + stmt_test_cases
)
def test_expression(expr_type, atomic_case, non_atomic_case, expected):
    # remove any indendentation from using triple quotes
    atomic_case = textwrap.dedent(atomic_case)
    non_atomic_case = textwrap.dedent(non_atomic_case)
    expected = textwrap.dedent(expected)

    atomic_tree = ast.parse(atomic_case)
    non_atomic_tree = ast.parse(non_atomic_case)
    expected_tree = ast.parse(expected)

    lift_atomic(atomic_tree)
    lift_non_atomic(non_atomic_tree, expected_tree)
    lift_twice(non_atomic_tree)


program_cases = [(
    'sum_number_list', """
        def acc(acc_op, f, ls):
            v = 0
            for e in ls:
                v = acc_op(v, f(e))
            return v

        double = lambda x: x * 2
        negative = lambda x: -x
        acc_add = lambda x, y: x + y
        acc_prod = lambda x, y: x * y
        ls = [1, 2, 3 * 4, 6, 7, double(100) + negative(10)]
        result = (acc(acc_add, double, ls), acc(acc_prod, negative, ls))
    """
),
                 (
                     'http://0pointer.de/blog/projects/mandelbrot.html', """
        import math, colorsys

        dimensions = (800, 800)
        scale = 1.0/(dimensions[0]/3)
        center = (2.2, 1.5)       # Use this for Mandelbrot set
        #center = (1.5, 1.5)       # Use this for Julia set
        iterate_max = 100
        colors_max = 50

        # img = Image.new("RGB", dimensions)
        # d = ImageDraw.Draw(img)
        result = []

        # Calculate a tolerable palette
        palette = [0] * colors_max
        for i in range(colors_max):
            f = 1-abs((float(i)/colors_max-1)**15)
            #r, g, b = colorsys.hsv_to_rgb(.66+f/3, 1-f/2, f)
            r, g, b = (.66+f/3, 1-f/2, f)
            palette[i] = (int(r*255), int(g*255), int(b*255))

        # Calculate the mandelbrot sequence for the point c with start value z
        def iterate_mandelbrot(c, z = 0):
            for n in range(iterate_max + 1):
                z = z*z +c
                if abs(z) > 2:
                    return n
            return None

        # Draw our image
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                c = complex(x * scale - center[0], y * scale - center[1])

                n = iterate_mandelbrot(c)            # Use this for Mandelbrot set
                #n = iterate_mandelbrot(complex(0.3, 0.6), c)  # Use this for Julia set

                if n is None:
                    v = 1
                else:
                    v = n/100.0

                #d.point((x, y), fill = palette[int(v * (colors_max-1))])
                result.append((x, y, palette[int(v * (colors_max - 1))]))
    """
                 ),
                 (
                     'meta-lifter-> lift lifter and then use it', """
        %s\nresult = unparse(ExpressionLifter().run('2 + 3 + 4 + sum([10, 20, 30]) * 40'))
    """ % open('plpy/rewrite/expr_lifter.py', 'r').read()
                 )]


def execute_program_from_ast(tree, output_var='result'):
    compiled = compile(tree, filename='ast', mode='exec')
    _locals = {}
    exec(compiled, _locals)
    return _locals[output_var]


def compare_orig_and_lifted_execution(src):
    src = textwrap.dedent(src)
    orig_results = execute_program_from_ast(ast.parse(src))
    lifted = lift_expressions(src)
    # TODO
    # hack to get around issues with lineno etc. Need to work on that
    # at some point
    lifted_src = astunparse.unparse(lifted)
    lifted_results = execute_program_from_ast(ast.parse(lifted_src))
    assert orig_results == lifted_results


@pytest.mark.parametrize("program_name,program_src", program_cases)
def test_execution(program_name, program_src):
    compare_orig_and_lifted_execution(program_src)
