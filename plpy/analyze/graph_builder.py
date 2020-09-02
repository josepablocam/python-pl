from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
import networkx as nx
import pickle
import textwrap

from .dynamic_tracer import DynamicDataTracer, get_nested_references, to_ast_node
from .dynamic_trace_events import *


class MemoryRefinementStrategy(Enum):
    INCLUDE_ALL = 0
    IGNORE_BASE = 1
    MOST_SPECIFIC = 2


class DynamicTraceToGraph(object):
    def __init__(self, ignore_unknown=False, memory_refinement=0):
        # do not construct nodes for unknown memory references
        self.ignore_unknown = ignore_unknown
        # only consume memory update based on policy
        self.memory_refinement = MemoryRefinementStrategy(memory_refinement)
        # graph with statement nodes and edges for data dependencies
        self.graph = nx.DiGraph()
        # counter for node identifiers
        self.counter = 0
        # note that pydot doesn't like negatives...
        self.unknown_id = self.allocate_node_id()
        # mappings to node identifiers
        self.lineno_to_nodeid = {}
        self.mem_loc_to_lineno = {}
        self.consuming = []

    def allocate_node_id(self):
        _id = self.counter
        self.counter += 1
        return _id

    def create_and_add_node(self, node_id, trace_event):
        self.graph.add_node(node_id)
        # set up attributes
        attributes = [
            'src', 'lineno', 'event', 'complete_defs', 'defs', 'calls', 'uses'
        ]
        for attr in attributes:
            self.graph.nodes[node_id][attr] = None
        if node_id == self.unknown_id:
            self.graph.nodes[node_id]['src'] = 'UNKNOWNS: '
        else:
            self.graph.nodes[node_id]['src'] = trace_event.line
            self.graph.nodes[node_id]['lineno'] = trace_event.lineno
            self.graph.nodes[node_id]['event'] = trace_event
        return self.graph.nodes[node_id]

    def handle_ExecLine(self, event):
        if self.consuming:
            return
        # TODO: this currently ignores loops and allocates a new node per statement executed
        node_id = self.allocate_node_id()
        self.create_and_add_node(node_id, event)
        self.graph.nodes[node_id]['uses'] = event.uses
        dependencies = []
        for var in event.uses:
            if (not self.ignore_unknown) or var.id in self.mem_loc_to_lineno:
                ml_id = self.get_latest_node_id_for_mem_loc(var.name, var.id)
                dependencies.append((ml_id, node_id))

        self.graph.add_edges_from(dependencies)
        # set node id for this lineno
        self.lineno_to_nodeid[event.lineno] = node_id

    def get_latest_node_id_for_mem_loc(self, name, mem_loc):
        if not mem_loc in self.mem_loc_to_lineno:
            # one of the unknown locations
            # create new node if needed and accumulate string for debugging when drawn
            if not self.unknown_id in self.graph.nodes:
                self.create_and_add_node(self.unknown_id, None)
            self.graph.nodes[self.unknown_id]['src'] += ('%s,' % name)
            return self.unknown_id
        else:
            lineno = self.mem_loc_to_lineno[mem_loc]
            return self.lineno_to_nodeid[lineno]

    @staticmethod
    def refine_ignore_base(_vars):
        """
        ignore base memory update for references to 'containers'
        """
        bases = set([])
        for var in _vars:
            ast_node_name = to_ast_node(var.name)
            refs = get_nested_references(ast_node_name, exclude_first=True)
            refs = sorted(refs, key=len)
            if refs:
                bases.add(refs[0])
        return [var for var in _vars if not var.name in bases]

    @staticmethod
    def refine_most_specific(_vars):
        """
        ignore all but the most specific memory update for references to 'containers'
        """
        nested_references = set([])
        for var in _vars:
            ast_node_name = to_ast_node(var.name)
            refs = get_nested_references(ast_node_name, exclude_first=True)
            nested_references.update(refs)
        return [var for var in _vars if not var.name in nested_references]

    def refine_memory_updates(self, _vars):
        # refine memory locations according to a specific strategy
        # this refinement is purely syntactic
        # doing this based on actual memory addresses isn't really feasible
        # for example, given a data frame df, df['c1'] always returns the same
        # id as long as unmodified, but retrieving first element,
        # id(df['c1'][0]), repeatedly returns different value as it allocates a new
        # np.dtype object.
        # If instead we retrieve with df['c1'].values.item(0) we always get same id
        # as it copies it to a Python object...
        # all this to say: inferring related objects from memory addresses hardly
        # seems bulletproof, so might as well just do syntactically
        if self.memory_refinement == MemoryRefinementStrategy.INCLUDE_ALL:
            return _vars, _vars
        elif self.memory_refinement == MemoryRefinementStrategy.IGNORE_BASE:
            return _vars, self.refine_ignore_base(_vars)
        elif self.memory_refinement == MemoryRefinementStrategy.MOST_SPECIFIC:
            return _vars, self.refine_most_specific(_vars)
        else:
            raise Exception(
                "Invalid memory refinement strategy: %s" %
                self.memory_refinement
            )

    def handle_MemoryUpdate(self, event):
        if self.consuming:
            return
        defs = list(event.defs)
        # complete defs maintain all information
        # but edges in the graph are only built off of defs
        complete_defs, defs = self.refine_memory_updates(defs)
        for d in defs:
            self.mem_loc_to_lineno[d.id] = event.lineno
        # add these defs to the line node that created them
        line_node_id = self.lineno_to_nodeid[event.lineno]
        self.graph.nodes[line_node_id]['defs'] = defs
        self.graph.nodes[line_node_id]['complete_defs'] = complete_defs

    def handle_EnterCall(self, event):
        self.consuming += [event]
        # add call information to node associated with the stmt that triggered the call event
        if event.lineno in self.lineno_to_nodeid:
            node_id = self.lineno_to_nodeid[event.lineno]
            calls = self.graph.nodes[node_id]['calls']
            calls = [] if calls is None else calls
            calls.append(event)
            self.graph.nodes[node_id]['calls'] = calls

    def handle_ExitCall(self, event):
        self.consuming.pop()

    def handle_ExceptionEvent(self, event):
        print(
            'Graph has an exception event, stopped processing. Saving current progress.'
        )

    def run(self, tracer):
        assert isinstance(
            tracer, DynamicDataTracer
        ), 'This graph builder only works for dynamic data traces'

        handlers = {
            ExecLine: self.handle_ExecLine,
            MemoryUpdate: self.handle_MemoryUpdate,
            EnterCall: self.handle_EnterCall,
            ExitCall: self.handle_ExitCall,
            ExceptionEvent: self.handle_ExceptionEvent,
        }
        for e in tracer.trace_events:
            handlers[type(e)](e)
        return self.graph


def draw(g, dot_layout=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    labels = nx.get_node_attributes(g, 'src')
    # use better graphviz layout
    pos = nx.drawing.nx_pydot.graphviz_layout(g) if dot_layout else None
    nx.draw(g, labels=labels, node_size=100, ax=ax, pos=pos)
    return plt, plt.gcf()


def main(args):
    with open(args.input_path, 'rb') as f:
        tracer = pickle.load(f)
    builder = DynamicTraceToGraph(
        ignore_unknown=args.ignore_unknown,
        memory_refinement=args.memory_refinement
    )
    graph = builder.run(tracer)
    with open(args.output_path, 'wb') as f:
        pickle.dump(graph, f)

    if args.draw:
        plt, plot_fig = draw(graph)
        plot_path = args.output_path + '_graph.pdf'
        plt.savefig(plot_path)
        plt.show(block=args.block)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Build networkx graph from tracer (with events)',
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        'input_path', type=str, help='Path to pickled tracer (with events)'
    )
    parser.add_argument(
        'output_path', type=str, help='Path to store pickled networkx graph'
    )
    parser.add_argument(
        '-i',
        '--ignore_unknown',
        action='store_true',
        help='Exclude unknown memory locations from graph'
    )
    parser.add_argument(
        '-m',
        '--memory_refinement',
        type=int,
        help=textwrap.dedent(
            """
    0: apply all memory updates (MemoryRefinementStrategy.INCLUDE_ALL) (DEFAULT)
    1: ignore base (MemoryRefinementStrategy.IGNORE_BASE)
    2: ignore all but most specific (MemoryRefinementStrategy.MOST_SPECIFIC)
    Determined syntactically
    """
        ),
        default=0
    )
    parser.add_argument(
        '-d', '--draw', action='store_true', help='Draw graph and display'
    )
    parser.add_argument(
        '-b',
        '--block',
        action='store_true',
        help='Block when displaying graph'
    )
    args = parser.parse_args()

    try:
        main(args)
    except Exception as err:
        import pdb
        pdb.post_mortem()
