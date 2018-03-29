import matplotlib.pyplot as plt
import networkx as nx

from .dynamic_trace_events import *

class TraceToGraph(object):
    def __init__(self):
        self.counter = 0
        self.graph = nx.DiGraph()
        self.lineno_to_nodeid = {}
        self.mem_loc_to_lineno = {}
        self.consuming = []
        self.unknown_id = -1

    def allocate_node_id(self):
        _id = self.counter
        self.counter += 1
        return _id

    def handle_ExecLine(self, event):
        if self.consuming:
            return

        node_id = self.allocate_node_id()
        self.graph.add_node(node_id)
        self.graph.node[node_id]['line'] = event.line
        self.lineno_to_nodeid[event.lineno] = node_id
        dependencies = [(self.get_latest_node_id_for_mem_loc(ml), node_id) for ml in event.uses_mem_locs]
        self.graph.add_edges_from(dependencies)

    def get_latest_node_id_for_mem_loc(self, mem_loc):
        if not mem_loc in self.mem_loc_to_lineno:
            # we shoulda llocate a new node here
            if not self.unknown_id in self.graph.nodes:
                self.graph.add_node(self.unknown_id)
                self.graph.node[self.unknown_id]['line'] = 'UNKNOWN'
            return self.unknown_id
        else:
            lineno = self.mem_loc_to_lineno[mem_loc]
            return self.lineno_to_nodeid[lineno]

    def handle_MemoryUpdate(self, event):
        if self.consuming:
            return

        for mem_loc in event.mem_locs:
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
    labels = nx.get_node_attributes(g, 'line')
    # use better graphviz layout
    pos = nx.drawing.nx_pydot.graphviz_layout(g) if dot_layout else None
    nx.draw(g, labels=labels, node_size=100, ax=ax, pos=pos)
    plt.show()
