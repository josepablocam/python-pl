import pytest
import textwrap

from plpy.analyze import graph_builder as gb
from plpy.analyze.dynamic_trace_events import ExecLine, MemoryUpdate, Variable
from plpy.analyze.dynamic_tracer import DynamicDataTracer


def test_create_node():
    grapher = gb.DynamicTraceToGraph()
    unknown_node = grapher.create_and_add_node(grapher.unknown_id, None)
    for attr, val in unknown_node.items():
        if attr != 'src':
            assert val is None, 'Unknown node has no attributes except src'

    dummy_event = ExecLine(
        event_id=1,
        lineno=1,
        line='test code',
        uses=[Variable('val', 10, int.__name__)]
    )
    dummy_node = grapher.create_and_add_node(10, dummy_event)
    assert dummy_node['lineno'] == dummy_event.lineno
    assert dummy_node['src'] == dummy_event.line
    assert dummy_node['event'] == dummy_event


def count_nodes(graph, expected):
    assert len(graph.nodes) == expected


def count_edges(graph, expected):
    assert len(graph.edges) == expected


def get_node_id_by_line(graph, lineno):
    for node_id, data in graph.nodes(data=True):
        if data['lineno'] == lineno:
            return node_id
    return None


def exists_edge(graph, lineno_from, lineno_to):
    expected_edge = (lineno_from, lineno_to)
    expected_edge = (get_node_id_by_line(l) for l in expected_edge)
    assert tuple(expected_edge) in graph.edges


def test_basic_graph():
    # x = 1
    # y = 2
    # x + y
    basic_trace_events = [
        ExecLine(event_id=1, lineno=1, line='x = 1', uses=[]),
        MemoryUpdate(
            event_id=2, lineno=1, defs=[Variable('x', 1, int.__name__)]
        ),
        ExecLine(event_id=3, lineno=2, line='y = 1', uses=[]),
        MemoryUpdate(
            event_id=4, lineno=2, defs=[Variable('y', 2, int.__name__)]
        ),
        ExecLine(
            event_id=5,
            lineno=3,
            line='x + y',
            uses=[
                Variable('x', 1, int.__name__),
                Variable('y', 2, int.__name__)
            ]
        ),
    ]
    basic_tracer = DynamicDataTracer()
    basic_tracer.trace_events = basic_trace_events
    grapher = gb.DynamicTraceToGraph()
    basic_graph = grapher.run(basic_tracer)

    count_nodes(basic_graph, 3)
    count_edges(basic_graph, 2)


# FIXME: figuring out potential issues with using id on pandas dataframes
# see: https://stackoverflow.com/questions/49782139/pandas-dataframes-series-and-id-in-cpython
# For now using safe_id in dynamic_tracer.py, see function definition there for more info
src_case_1 = """
import pandas as pd
df = pd.DataFrame([(1, 2), (3, 4)], columns=['c1', 'c2'])
df.c1 = 100
df.c2 = 20
y = df.c1 + 10
"""

src_case_2 = """
d = [0, 1, 2, 3, 4]
d[1] = 10
d[0] = 20
x = d[1] + 2
y = d[0] + 3
"""

src_case_3 = """
m = [[0, 1, 2],[3, 4, 5],[6, 7, 8]]
m[0][1] = 100
m[0][2] = 20
x = m[0] * 2
y = m[0][1]
"""

src_case_4 = """
m = [[0, 1, 2],[3, 4, 5],[6, 7, 8]]
m[0][1] = 100
m[0][2] = 20
x = m[0] # note that we are just copying over, this is going to make the trace look like we assigned m[0] here, so y now depends on x
y = m[0][1]
"""

src_case_5 = """
import pandas as pd
df = pd.DataFrame([(1, 2), (3, 4)], columns=['c1', 'c2'])
df.c1 = 100
df.c2 = 20
df = df.copy() # new one
y = df.c1 + 10
"""

src_case_6 = """
import pandas as pd
df = pd.DataFrame([(1, 2), (3, 4)], columns=['c1', 'c2'])
df.c1 = 100
df.c2 = 20
df = df.copy() # new one
df.c1 = 1000
y = df.c1 + 10
"""

refinement_cases = [
    (
        src_case_1, gb.MemoryRefinementStrategy.INCLUDE_ALL, [(1, 2), (2, 3),
                                                              (3, 4), (3, 5),
                                                              (4, 5)]
    ),
    (
        src_case_1, gb.MemoryRefinementStrategy.IGNORE_BASE, [(1, 2), (2, 3),
                                                              (2, 4), (2, 5),
                                                              (3, 5)]
    ),
    (
        src_case_1, gb.MemoryRefinementStrategy.MOST_SPECIFIC, [(1, 2), (2, 3),
                                                                (2, 4), (2, 5),
                                                                (3, 5)]
    ),
    #####
    (
        src_case_2, gb.MemoryRefinementStrategy.INCLUDE_ALL, [(1, 2), (2, 3),
                                                              (2, 4), (3, 4),
                                                              (3, 5)]
    ),
    (
        src_case_2, gb.MemoryRefinementStrategy.IGNORE_BASE, [(1, 2), (1, 3),
                                                              (1, 4), (2, 4),
                                                              (1, 5), (3, 5)]
    ),
    (
        src_case_2, gb.MemoryRefinementStrategy.MOST_SPECIFIC, [(1, 2), (1, 3),
                                                                (1, 4), (2, 4),
                                                                (1, 5), (3, 5)]
    ),
    ####
    (
        src_case_3, gb.MemoryRefinementStrategy.INCLUDE_ALL, [(1, 2), (2, 3),
                                                              (3, 4), (2, 5),
                                                              (3, 5)]
    ),
    (
        src_case_3, gb.MemoryRefinementStrategy.IGNORE_BASE, [(1, 2), (1, 3),
                                                              (1, 4), (1, 5),
                                                              (2, 3), (2, 5),
                                                              (3, 4), (3, 5)]
    ),
    (
        src_case_3, gb.MemoryRefinementStrategy.MOST_SPECIFIC, [(1, 2), (1, 3),
                                                                (1, 4), (1, 5),
                                                                (2, 5)]
    ),
    ###
    (
        src_case_4, gb.MemoryRefinementStrategy.INCLUDE_ALL, [(1, 2), (2, 3),
                                                              (3, 4), (2, 5),
                                                              (3, 5), (4, 5)]
    ),
    ####
    # note that y = df.c1 + 10 is not dependent on the df.c1 = 100 at line 3, as we created an object copy
    (
        src_case_5, gb.MemoryRefinementStrategy.INCLUDE_ALL, [(1, 2), (2, 3),
                                                              (3, 4), (4, 5),
                                                              (5, 6)]
    ),
    (
        src_case_5, gb.MemoryRefinementStrategy.IGNORE_BASE, [(1, 2), (2, 3),
                                                              (2, 4), (2, 5),
                                                              (5, 6)]
    ),
    (
        src_case_5, gb.MemoryRefinementStrategy.MOST_SPECIFIC, [(1, 2), (2, 3),
                                                                (2, 4), (2, 5),
                                                                (5, 6)]
    ),
    ###
    (
        src_case_6, gb.MemoryRefinementStrategy.INCLUDE_ALL, [(1, 2), (2, 3),
                                                              (3, 4), (4, 5),
                                                              (5, 6), (6, 7)]
    ),
    (
        src_case_6, gb.MemoryRefinementStrategy.IGNORE_BASE, [(1, 2), (2, 3),
                                                              (2, 4), (2, 5),
                                                              (5, 6), (5, 7),
                                                              (6, 7)]
    ),
    (
        src_case_6, gb.MemoryRefinementStrategy.MOST_SPECIFIC, [(1, 2), (2, 3),
                                                                (2, 4), (2, 5),
                                                                (5, 6), (5, 7),
                                                                (6, 7)]
    ),
]


@pytest.mark.parametrize('src,strategy,expected', refinement_cases)
def test_memory_update_refinement_strategies(src, strategy, expected):
    src = textwrap.dedent(src)
    tracer = DynamicDataTracer()
    tracer.run(src)
    print(list(map(str, tracer.trace_events)))
    grapher = gb.DynamicTraceToGraph(
        memory_refinement=strategy, ignore_unknown=True
    )
    graph = grapher.run(tracer)
    assert sorted(graph.edges) == sorted(expected)
