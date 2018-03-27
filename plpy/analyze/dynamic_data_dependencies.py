import ast
import astunparse
import inspect
import sys
import dis
import matplotlib.pyplot as plt
plt.ion()

import networkx as nx

def to_ast_node(line):
    return ast.parse(line).body[0]

class TraceEvent(object):
    pass

class MemoryUpdate(TraceEvent):
    """
    Added to trace to update the last line assigning
    to a set of memory locations
    """
    def __init__(self, mem_locs, lineno):
        self.mem_locs = list(mem_locs)
        self.lineno = lineno

    def __str__(self):
        return 'mem-update(%s)' % self.mem_locs

class ExecLine(TraceEvent):
    def __init__(self, lineno, line, uses_mem_locs):
        self.lineno = lineno
        self.line = line
        self.uses_mem_locs = uses_mem_locs

    def __str__(self):
        return 'exec line: %s (line=%d)' % (self.line, self.lineno)

class EnterCall(TraceEvent):
    def __init__(self, call_site_lineno, call_site_line, stuff):
        self.call_site_lineno = call_site_lineno
        self.call_site_line = call_site_line
        self.stuff = stuff

    def __str__(self):
        return 'enter call: %s (line=%d)' % (self.call_site_line, self.call_site_lineno)

class ExitCall(TraceEvent):
    def __init__(self, call_site_lineno, call_site_line, stuff):
        self.call_site_lineno = call_site_lineno
        self.call_site_line = call_site_line
        self.stuff = stuff

    def __str__(self):
        return 'exit call: %s (line=%d)' % (self.call_site_line, self.call_site_lineno)

# TODO:
# We should have detailed information for a function call
# like the memory location of each argument, and the type
# and constants where possible
class TraceToGraph(object):
    def __init__(self):
        self.counter = 0
        self.graph = nx.DiGraph()
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
        self.graph.node[node_id]['line'] = event.line
        self.lineno_to_nodeid[event.lineno] = node_id

        dependencies = [(self.get_latest_node_id_for_mem_loc(ml), node_id) for ml in event.uses_mem_locs]
        self.graph.add_edges_from(dependencies)

    def get_latest_node_id_for_mem_loc(self, mem_loc):
        if not mem_loc in self.mem_loc_to_lineno:
            # we shoulda llocate a new node here
            pass
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


class ExtractReferences(ast.NodeVisitor):
    """
    Extract nested names and attributes references.
    """
    def __init__(self):
        self.acc = []

    def visit_Name(self, node):
        self.acc.append(node)

    def visit_Attribute(self, node):
        self.acc.append(node)
        self.generic_visit(node)

    def run(self, node):
        self.visit(node)
        return [astunparse.unparse(ref).strip() for ref in self.acc]

class RegisterAssignments(ast.NodeTransformer):
    def __init__(self, stub_name):
        self.stub_name = stub_name

    def visit_Assign(self, node):
        return node, self._create_register_node(node.targets)

    def visit_AugAssign(self, node):
        return node, self._create_register_node([node.target])

    def _create_register_node(self, targets):
        references = []
        for tgt in targets:
            references.extend(ExtractReferences().run(tgt))
        ref_str = ','.join(references)
        call_str = f'{self.stub_name}([{ref_str}])'
        return to_ast_node(call_str)


def get_lineno(frame):
    return frame.f_lineno

def get_filename(frame):
    return frame.f_code.co_filename

# def get_function_name(frame):
#     caller_frame = frame.f_back
#     module = inspect.getmodule(caller_frame)
#     func_ref_name = frame.f_code.co_name
#     func_obj = caller_frame.f_locals[func_ref_name]
#     name = func_obj.__name__
#     if module:
#         name = '%s.%s' % (module.__name__, name)
#     return name

def get_function_obj(frame):
    caller_frame = frame.f_back
    func_ref_name = frame.f_code.co_name
    if func_ref_name in caller_frame.f_locals:
        return caller_frame.f_locals[func_ref_name]
    elif func_ref_name in caller_frame.f_globals:
        return caller_frame.f_globals[func_ref_name]
    else:
        return None

def register_stub(regs):
    pass

STUBS = [register_stub]

class DynamicDataTracer(object):
    def __init__(self):
        self.acc = []
        self.orig_trace = None
        self.tracer_file_name = inspect.getsourcefile(inspect.getmodule(self))
        self.check = []
        self.stubs = set(STUBS)

    def _load_src_code(self, file_path):
        with open(file_path, 'r') as f:
            code = f.readlines()
        return code

    def _split_src_code(self, src):
        return [line.strip() for line in src.split('\n')]

    def get_line_of_code(self, frame):
        if get_filename(frame) == self.file_name:
            line = self.src_code_lines[get_lineno(frame) - 1]
        else:
            line = None
        return line

    def should_ignore(self, frame):
        # ignore calls that are inside tracer
        return frame.f_code.co_filename == self.tracer_file_name

    def trace(self, frame, event, arg):
        if event == 'line':
            return self.trace_line(frame, event, arg)

        if event == 'call':
            return self.trace_call(frame, event, arg)

        if event == 'return':
            return self.trace_return(frame, event, arg)

        return None

    def trace_line(self, frame, event, arg):
        try:
            line = self.get_line_of_code(frame)
            load_mem_locs = self.get_loads_mem_locs(line, frame)
            trace_event = ExecLine(get_lineno(frame), line, load_mem_locs)
            self.acc.append(trace_event)
            return self.trace
        except SyntaxError:
            return self.trace

    def get_loads_mem_locs(self, line, frame):
        node = to_ast_node(line)
        references = []
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            # if this is a attribute access or slice
            # then the accessing the LHS counts as a load
            for target in node.targets:
                if isinstance(target, (ast.Subscript, ast.Attribute, ast.Slice)):
                    references.extend(ExtractReferences().run(target))
            references.extend(ExtractReferences().run(node.value))
        else:
            references.extend(ExtractReferences().run(node))
        _locals = frame.f_locals
        _globals = frame.f_globals
        mem_locs = []
        for ref in references:
            if ref in _locals:
                mem_locs.append(id(_locals[ref]))
            elif ref in _globals:
                mem_locs.append(id(_globals[ref]))
            else:
                mem_locs.append(-1)
        return mem_locs

    def is_stub_call(self, frame):
        func_obj = get_function_obj(frame)
        return func_obj in STUBS

    def trace_call(self, frame, event, arg):
        if self.is_stub_call(frame):
            return self.trace_stub(frame, event, arg)
        # call site
        call_site_lineno = get_lineno(frame.f_back)
        call_site_line = self.get_line_of_code(frame.f_back)
        # TODO: stuff are things like memory locations
        # of arguments, values for each argument if constant, and type for all
        stuff = (frame)
        trace_event = EnterCall(call_site_lineno, call_site_line, stuff)
        self.acc.append(trace_event)
        if frame.f_code.co_filename == self.file_name:
            return self.trace
        else:
            return None

    def trace_stub(self, frame, event, arg):
        stub_obj = get_function_obj(frame)
        if stub_obj == register_stub:
            self.register_assignments(frame, event, arg)
        else:
            raise Exception("Unknown stub type")

    def register_assignments(self, frame, event, arg):
        # extract line no, and remove stuff
        caller = frame.f_back
        # the actual line is off by one
        lineno = get_lineno(caller) - 1
        arg_name = frame.f_code.co_varnames
        assert len(arg_name) == 1, 'assignment stub should have only 1 argument: list of references'
        # get argument: a list of memory references
        arg = frame.f_locals[arg_name[0]]
        # memory locations
        memory_locations = [id(ref) for ref in arg]
        # do something with these
        trace_event = MemoryUpdate(memory_locations, lineno)
        # overwrite the previous event, which is
        # the dummy execline
        self.acc[-1] = trace_event

    def trace_return(self, frame, event, arg):
        call_site_lineno = get_lineno(frame.f_back)
        call_site_line = self.get_line_of_code(frame.f_back)
        stuff = (frame)
        trace_event = ExitCall(call_site_lineno, call_site_line, stuff)
        self.acc.append(trace_event)

    def setup(self):
        self.orig_tracer = sys.gettrace()
        sys.settrace(self.trace)

    def shutdown(self):
        sys.settrace(self.orig_tracer)

    def run(self, src):
        self.acc = []
        tree = ast.parse(src)
        ext_tree = RegisterAssignments(stub_name=register_stub.__qualname__).visit(tree)
        ext_src = astunparse.unparse(ext_tree)
        self.src_code_lines = self._split_src_code(ext_src)
        self.file_name = '__main__'
        compiled = compile(ext_src, filename=self.file_name, mode='exec')
        self.setup()
        exec(compiled)
        self.shutdown()
        # remove enter/exit calls associated with tracer itself
        self.acc = self.acc[1:-2]


src = """
def f(x):
    return x + 2

def get_address(x):
    return id(x)

start = 3
y = f(start)
z = y * 2
get_address(y)
x = 100
x,z = (10, 20)
w = z
v = [1,2,[10, 20, 30]]
v[-1][0] = z
v
"""
