from argparse import ArgumentParser
import networkx as nx
import pickle

from .dynamic_tracer import DynamicDataTracer
from .dynamic_trace_events import *


class DynamicTraceToGraph(object):
    def __init__(self, ignore_unknown=False):
        self.counter = 0
        # note that pydot doesn't like negatives...
        self.unknown_id = self.allocate_node_id()
        self.graph = nx.DiGraph()
        self.ignore_unknown = ignore_unknown
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
        attributes = ['src', 'lineno']
        for attr in attributes:
            self.graph.nodes[node_id][attr] = None
        if node_id == self.unknown_id:
            self.graph.nodes[node_id]['src'] = 'UNKNOWNS: '
        else:
            self.graph.node[node_id]['src'] = trace_event.line
            self.graph.node[node_id]['lineno'] = trace_event.lineno

    def handle_ExecLine(self, event):
        if self.consuming:
            return
        # TODO: this currently ignores loops and allocates a new node per statement executed
        node_id = self.allocate_node_id()
        self.create_and_add_node(node_id, event)
        dependencies = []
        # ignore mem locations that are unknown
        for name, ml in event.uses_mem_locs.items():
            if (not self.ignore_unknown) or ml in self.mem_loc_to_lineno:
                ml_id = self.get_latest_node_id_for_mem_loc(name, ml)
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

    def handle_MemoryUpdate(self, event):
        if self.consuming:
            return

        for name, mem_loc in event.mem_locs.items():
            self.mem_loc_to_lineno[mem_loc] = event.lineno

    def handle_EnterCall(self, event):
        self.consuming += [event]

    def handle_ExitCall(self, event):
        self.consuming.pop()

    def run(self, tracer):
        assert isinstance(tracer, DynamicDataTracer), 'This graph builder only works for dynamic data traces'

        handlers = {
            ExecLine: self.handle_ExecLine,
            MemoryUpdate: self.handle_MemoryUpdate,
            EnterCall: self.handle_EnterCall,
            ExitCall: self.handle_ExitCall,
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
    builder = DynamicTraceToGraph(ignore_unknown=args.ignore_unknown)
    graph = builder.run(tracer)
    with open(args.output_path, 'wb') as f:
        pickle.dump(graph, f)

    if args.draw:
        plt, plot_fig = draw(graph)
        plot_path = args.output_path + '_graph.pdf'
        plt.savefig(plot_path)
        plt.show(block=args.block)

if __name__ == '__main__':
    parser = ArgumentParser(description='Build networkx graph from tracer (with events)')
    parser.add_argument('input_path', type=str, help='Path to pickled tracer (with events)')
    parser.add_argument('output_path', type=str, help='Path to store pickled networkx graph')
    parser.add_argument('-i', '--ignore_unknown', action='store_true', help='Exclude unknown memory locations from graph')
    parser.add_argument('-d', '--draw', action='store_true', help='Draw graph and display')
    parser.add_argument('-b', '--block', action='store_true', help='Block when displaying graph')
    args = parser.parse_args()
    try:
        main(args)
    except:
        import pdb
        pdb.post_mortem()
