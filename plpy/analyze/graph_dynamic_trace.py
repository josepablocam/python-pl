import matplotlib.pyplot as plt
import networkx as nx

from .dynamic_trace_events import *

GRAPH_SRC_ATTRIBUTE = 'src'

class TraceToGraph(object):
    def __init__(self, include_unknown=True):
        self.counter = 0
        # note that pydot doesn't like negatives...
        self.unknown_id = self.allocate_node_id()
        self.graph = nx.DiGraph()
        self.include_unknown = include_unknown
        self.lineno_to_nodeid = {}
        self.mem_loc_to_lineno = {}
        self.consuming = []

    def allocate_node_id(self):
        _id = self.counter
        self.counter += 1
        return _id

    def handle_ExecLine(self, event):
        if self.consuming:
            return

        node_id = self.allocate_node_id()
        self.graph.add_node(node_id)
        self.graph.node[node_id][GRAPH_SRC_ATTRIBUTE] = event.line
        self.lineno_to_nodeid[event.lineno] = node_id
        dependencies = []
        # ignore mem locations that are unknown
        for name, ml in event.uses_mem_locs.items():
            if self.include_unknown or ml in self.mem_loc_to_lineno:
                ml_id = self.get_latest_node_id_for_mem_loc(name, ml)
                dependencies.append((ml_id, node_id))

        self.graph.add_edges_from(dependencies)

    def get_latest_node_id_for_mem_loc(self, name, mem_loc):
        if not mem_loc in self.mem_loc_to_lineno:
            # one of the unknown locations
            # create new node if needed and accumulate string for debugging when drawn
            if not self.unknown_id in self.graph.nodes:
                self.graph.add_node(self.unknown_id)
                self.graph.nodes[self.unknown_id][GRAPH_SRC_ATTRIBUTE] = 'UNKNOWNS: '
            self.graph.nodes[self.unknown_id][GRAPH_SRC_ATTRIBUTE] += ('%s,' % name)
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

    def run(self, trace_events):
        handlers = {
            ExecLine: self.handle_ExecLine,
            MemoryUpdate: self.handle_MemoryUpdate,
            EnterCall: self.handle_EnterCall,
            ExitCall: self.handle_ExitCall,
        }
        for e in trace_events:
            handlers[type(e)](e)
        return self.graph



def draw(g, dot_layout=True):
    fig, ax = plt.subplots(1)
    labels = nx.get_node_attributes(g, GRAPH_SRC_ATTRIBUTE)
    # use better graphviz layout
    pos = nx.drawing.nx_pydot.graphviz_layout(g) if dot_layout else None
    nx.draw(g, labels=labels, node_size=100, ax=ax, pos=pos)
    plt.show()
