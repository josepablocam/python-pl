import ast
import astunparse
import inspect
import sys

import matplotlib.pyplot as plt
plt.ion()
import networkx as nx
import pandas as pd

from .dynamic_trace_events import *

# helper functions
def to_ast_node(line):
    return ast.parse(line).body[0]

def get_lineno(frame):
    return frame.f_lineno

def get_caller_frame(frame):
    return frame.f_back
 
# this is a horrendous hack... but not sure that there is a better way
def get_function_obj(frame):
    # can't get function object if we don't actually
    # have the frame
    if frame is None or frame.f_back is None:
        return None
    # note that this hack is based on the info
    # described in 
    # https://stackoverflow.com/questions/16589547/get-fully-qualified-method-name-from-inspect-stack
    parent = get_caller_frame(frame)
    parent_info = inspect.getframeinfo(parent)
    src_line = parent_info.code_context[0].strip()
    # extract the function in the src call
    # import pdb
    # pdb.set_trace()
    call_func_str = astunparse.unparse(to_ast_node(src_line).value.func)
    func_obj = eval(call_func_str, parent.f_globals, parent.f_locals)
    return func_obj

def get_function_unqual_name(frame):
    return inspect.getframeinfo(frame).function
    
def get_function_qual_name(frame):
    # relies on the function object
    obj = get_function_obj(frame)
    if obj is None:
        return None
    return obj.__qualname__    


# stub functions that are inserted
# into source code to mark events for trace tracking
def memory_update_stub(regs):
    pass

STUBS = [memory_update_stub]

def is_stub_call(func_obj):
    return func_obj in STUBS



# TODO:
# We should have detailed information for a function call
# like the memory location of each argument, and the type
# and constants where possible


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


class AddMemoryUpdateStubs(ast.NodeTransformer):
    def __init__(self, stub_name):
        self.stub_name = stub_name

    def visit_Assign(self, node):
        return node, self._create_stub_node(node.targets)

    def visit_AugAssign(self, node):
        return node, self._create_stub_node([node.target])

    def _create_stub_node(self, targets):
        references = []
        for tgt in targets:
            references.extend(ExtractReferences().run(tgt))
        ref_str = ','.join(references)
        call_str = f'{self.stub_name}([{ref_str}])'
        return to_ast_node(call_str)


class DynamicDataTracer(object):
    def __init__(self):
        self.event_counter = 0
        self.trace = []
        self.trace_errors = []
        self.orig_tracer = None
        
    def _allocate_event_id(self):
        _id = self.event_counter
        self.event_counter += 1
        return _id

    def _load_src_code(self, file_path):
        with open(file_path, 'r') as f:
            code = f.readlines()
        return code

    def _split_src_code(self, src):
        return [line.strip() for line in src.split('\n')]

    def get_line_of_code(self, frame):
        if inspect.getfile(frame) == self.file_name:
            line = self.src_code_lines[inspect.getlineno(frame) - 1]
        else:
            line = None
        return line

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
            event_id = self._allocate_event_id()
            trace_event = ExecLine(event_id, inspect.getlineno(frame), line, load_mem_locs)
            self.trace.append(trace_event)
            return self.trace
        except SyntaxError:
            print('Syntax error')
            self.trace_errors.append((frame, event, arg, self.get_line_of_code(frame)))
            return self.trace

    def get_loads_mem_locs(self, line, frame):
        node = to_ast_node(line)
        references = []
        
        # TODO: rewrite this to make it clear
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            # if this is a attribute access or slice
            # then the accessing the LHS counts as a load
            # TODO; fix, ugh
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, (ast.Subscript, ast.Attribute, ast.Slice)):
                    references.extend(ExtractReferences().run(target))
            references.extend(ExtractReferences().run(node.value))
        else:
            references.extend(ExtractReferences().run(node))
        _locals = frame.f_locals
        _globals = frame.f_globals
        mem_locs = []
        for ref in references:
            try:
                obj = eval(ref, _globals, _locals)
                mem_locs.append(id(obj))
            except NameError:
                mem_locs.append(-1)
        return mem_locs
    
    def _called_by_user(self, frame):
        """ only trace calls to functions directly invoked by the user """
        return frame.f_back.f_code.co_filename == self.file_name
        
    def _defined_by_user(self, frame):
        """ only trace lines inside body of functions that are defined by user in same file """
        return frame.f_code.co_filename == self.file_name

    def trace_call(self, frame, event, arg):
        # TODO: can just use inspect.getargvalues(frame)
        # to get info and stuff
        # to get self etc
        # use inspect.ismethod/isfunction to distinguish between them
        func_obj = get_function_obj(frame)
        if is_stub_call(func_obj):
            return self.trace_stub(frame, event, arg)
        if self._called_by_user(frame):
            # call site
            caller_frame = get_caller_frame(frame)
            call_site_lineno = inspect.getlineno(caller_frame)
            call_site_line = self.get_line_of_code(caller_frame)
            # TODO: stuff are things like memory locations
            # of arguments, values for each argument if constant, and type for all
            # DO STUFF WITH arguments
            stuff = (frame)
            event_id = self._allocate_event_id()
            trace_event = EnterCall(event_id, call_site_lineno, call_site_line, stuff)
            self.trace.append(trace_event)
        if self._defined_by_user(frame):
            return self.trace
        else:
            return None
    
    def trace_return(self, frame, event, arg):
        call_site_lineno = inspect.getlineno(frame.f_back)
        call_site_line = self.get_line_of_code(frame.f_back)
        stuff = (frame)
        event_id = self._allocate_event_id()
        trace_event = ExitCall(event_id, call_site_lineno, call_site_line, stuff)
        self.trace.append(trace_event)

    def trace_stub(self, frame, event, arg):
        stub_obj = get_function_obj(frame)
        stub_event = None
        if stub_obj == memory_update_stub:
            stub_event = self.consume_memory_update_stub(frame, event, arg)
        else:
            raise Exception("Unknown stub type")
        # remove stub from trace of events
        self.trace.pop()
        # and if we have a stub event, accumulate it
        if stub_event:
            self.trace.append(stub_event)

    def consume_memory_update_stub(self, frame, event, arg):
        # extract line no, and remove stuff
        caller = frame.f_back
        # the actual line is off by one
        lineno = inspect.getlineno(caller) - 1
        # Use: inspect.getargvalues
        arg_name = frame.f_code.co_varnames
        assert len(arg_name) == 1, 'assignment stub should have only 1 argument: list of references'
        # get argument: a list of memory references
        arg = frame.f_locals[arg_name[0]]
        # memory locations
        memory_locations = [id(ref) for ref in arg]
        event_id = self._allocate_event_id()
        trace_event = MemoryUpdate(event_id, memory_locations, lineno)
        # overwrite the previous event, which is
        # the stub execline
        return trace_event

    def setup(self):
        self.orig_tracer = sys.gettrace()
        sys.settrace(self.trace)

    def shutdown(self):
        sys.settrace(self.orig_tracer)
        
    def add_stubs(self, src):
        tree = ast.parse(src)
        ext_tree = AddMemoryUpdateStubs(stub_name=memory_update_stub.__qualname__).visit(tree)
        return astunparse.unparse(ext_tree)

    def run(self, src):
        src = self.add_stubs(src)
        self.src_code_lines = self._split_src_code(src)
        self.file_name = '__main__'
        compiled = compile(src, filename=self.file_name, mode='exec')
        _globals = {memory_update_stub.__qualname__: register_stub}
        _locals = {}
        self.setup()
        exec(compiled, _globals, _locals)
        self.shutdown()
        # remove enter/exit calls associated with tracer itself
        self.trace = self.acc[1:-2]

