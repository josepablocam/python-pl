# Add dynamic tracing to a script, records 'trace events'
# that can be used to build up data dependency information later on
from argparse import ArgumentParser
import ast
import astunparse
import inspect
import logging
import os
import pickle
import sys


from .dynamic_trace_events import *

logging.basicConfig(
    filename="test.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
log = logging.getLogger(__name__)

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
        log.debug('Call frame or caller frame was None')
        return None
    # note that this hack is based on the info
    # described in
    # https://stackoverflow.com/questions/16589547/get-fully-qualified-method-name-from-inspect-stack
    try:
        parent = get_caller_frame(frame)
        parent_info = inspect.getframeinfo(parent)
        if parent_info.code_context:
            log.debug('Retrieving function object through caller frame')
            src_line = parent_info.code_context[0].strip()
            log.debug('Call (based on parent frame): %s' % src_line)
            ast_node = to_ast_node(src_line)
            func_str = astunparse.unparse(ast_node.value.func)
        else:
            # we can fall back on looking at the routine name directly
            # rather than how it was called...I believe this is less precise though
            # because we won't necessarily get the exact instance that called it
            log.debug('Retrieving function object through co_name')
            func_str = frame.f_code.co_name
        func_obj = eval(func_str, parent.f_globals, parent.f_locals)
        return func_obj
    except NameError:
        log.exception('Failed to look up name for function object')
        return None
    except SyntaxError:
        log.error('Failed to parse call', exc_info=True)
        return None

def get_function_unqual_name(frame):
    return inspect.getframeinfo(frame).function

def get_function_qual_name(obj):
    # relies on the function object
    if inspect.isframe(obj):
        obj = get_function_obj(obj)
    if obj is None:
        return None
    return obj.__qualname__

def get_abstract_vals(arginfo):
    arg_names = arginfo.args
    _locals = arginfo.locals
    return {name:id(_locals[name]) for name in arg_names}

# stub functions that are inserted
# into source code to mark events for trace tracking
def memory_update_stub(names, values):
    pass

STUBS = [memory_update_stub]

def is_stub_call(func_obj):
    return func_obj in STUBS


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


def get_nested_references(node, exclude_first=False):
    refs = ExtractReferences().run(node)
    if exclude_first:
        refs = refs[1:]
    return refs


class AddMemoryUpdateStubs(ast.NodeTransformer):
    def __init__(self, stub_name):
        self.stub_name = stub_name

    def visit_Assign(self, node):
        return node, self._create_stub_node(node.targets)

    def visit_AugAssign(self, node):
        return node, self._create_stub_node([node.target])

    def visit_Import(self, node):
        names = []
        for _alias in node.names:
            name = _alias.asname if _alias.asname else _alias.name
            names.append(name)
        return node, self._create_stub_node(names, get_nested=False)

    def visit_ImportFrom(self, node):
        _, stubs = self.visit_Import(ast.Import(node.names))
        return node, stubs

    def _create_stub_node(self, targets, get_nested=True):
        references = []
        if get_nested:
            for tgt in targets:
                references.extend(get_nested_references(tgt))
        else:
            references.extend(targets)
        names_str = ','.join(['"%s"' % ref for ref in references])
        values_str = ','.join(references)
        call_str = f'{self.stub_name}([{names_str}], [{values_str}])'
        return to_ast_node(call_str)


class DynamicDataTracer(object):
    def __init__(self):
        self.file_path = None
        self.src_lines = None
        self.event_counter = 0
        self.trace_events = []
        self.trace_errors = []
        self.orig_tracer = None

    def _allocate_event_id(self):
        log.info('Allocated new event id')
        _id = self.event_counter
        self.event_counter += 1
        return _id

    def _called_by_user(self, frame):
        """ only trace calls to functions directly invoked by the user """
        log.info('Current frame is for a user-called function')
        return inspect.getfile(get_caller_frame(frame)) == self.file_path

    def _defined_by_user(self, frame):
        """ only trace lines inside body of functions that are defined by user in same file """
        log.info('Current frame is for a user-defined function')
        try:
            return inspect.getfile(frame) == self.file_path
        except TypeError:
            # will be raised for buildin module/class/function, which by definition are not defined by used
            return False

    def _getsource(self, frame):
        # we strip before returning the line as
        # we often use this line to parse a node
        # and any kind of indentation will lead to SyntaxError raised in ast.parse
        log.info('Retrieving source for frame')
        if self._defined_by_user(frame):
            lineno = inspect.getlineno(frame)
            return self.src_lines[lineno - 1].strip()
        # try using inspect if not
        try:
            return inspect.getsource(frame).strip()
        except IOError:
            log.exception('Unable to retrieve source for frame')
            return None

    def trace(self, frame, event, arg):
        if event == 'line':
            return self.trace_line(frame, event, arg)

        if event == 'call':
            return self.trace_call(frame, event, arg)

        if event == 'return':
            return self.trace_return(frame, event, arg)

        return None

    def trace_external(self, frame, event, arg):
        """ tracing function for when we enter an external function"""
        # This is necessary to address calls to external libraries
        # where the user calls the function directly, so our standard `trace`
        # function creates an enter-call event, but since no other tracing
        # should take place, we never create the corresponding exit-call event
        # by returning this function that just traces returns, we can make
        # sure that exit-call event is created and added to our trace
        if event == 'return':
            log.info('Trace return of external function')
            return self.trace_return(frame, event, arg)

    def trace_line(self, frame, event, arg):
        log.info('Trace line')
        # Note that any C-based function (e.g. max/len etc)
        # won't actually trigger a call, so we need to see what to do here
        line = self._getsource(frame)
        event_id = self._allocate_event_id()
        try:
            load_references = self.get_load_references_from_line(line)
            load_mem_locs = self.get_mem_locs(load_references, frame)
            trace_event = ExecLine(event_id, inspect.getlineno(frame), line, load_mem_locs)
            self.trace_events.append(trace_event)
        except SyntaxError:
            log.exception('Syntax error while tracing line: %s' % line)
            # keeping actual frame does little to help debugging since it
            # gets updated during execution
            frameinfo = inspect.getframeinfo(frame)
            self.trace_errors.append((frameinfo, event, arg, line))
        return self.trace

    def get_load_references_from_line(self, line):
        log.info('Get references from line')
        node = to_ast_node(line)

        # assignments can trigger loads as well
        # attribute: a.b = ... => loads(a), loads(a.b)
        # subscript: val[ix][...] = ... => loads(val)
        # slice: val[1:...] = ... => loads(val)
        assignment_targets = []
        if isinstance(node, ast.Assign):
            assignment_targets.extend(node.targets)
            expr_node = node.value
        elif isinstance(node, ast.AugAssign):
            assignment_targets.append(node.target)
            expr_node = node.value
        else:
            expr_node = node

        references = []
        for target in assignment_targets:
            if isinstance(target, ast.Attribute):
                # in this case we don't want to have as a load the deepest access
                # e.g. a.b.c = ... => load(a), load(a.b) (but not load(a.b.c))
                references.extend(get_nested_references(target, exclude_first=True))
            if isinstance(target, (ast.Subscript, ast.Slice)):
                references.extend(get_nested_references(target))

        # RHS of line (or original node if just expression)
        references.extend(get_nested_references(expr_node))
        return set(references)

    def get_mem_locs(self, str_references, frame):
        """
        Load the memory locations for these references using the environment
        available to the frame provided
        """
        log.info('Get memory locations for references')

        _locals = frame.f_locals
        _globals = frame.f_globals
        mem_locs = {}
        for ref in str_references:
            try:
                obj = eval(ref, _globals, _locals)
                mem_locs[ref] = id(obj)
            except NameError:
                mem_locs[ref] = -1
        return mem_locs

    def trace_call(self, frame, event, arg):
        log.info('Trace call')

        # retrieve the actual function being called
        func_obj = get_function_obj(frame)

        # unable to do much here
        # so we basically just stop tracing until the next call comes around
        if func_obj is None:
            log.warn('Function object was None for source: %s' % self._getsource(frame))
            return None

        if is_stub_call(func_obj):
            log.info('Function object is a stub')
            return self.trace_stub(frame, event, arg)

        if self._called_by_user(frame):
            log.info('Collecting call made by user')
            # call site
            caller_frame = get_caller_frame(frame)
            call_site_lineno = inspect.getlineno(caller_frame)
            call_site_line = self._getsource(caller_frame)

            # call details
            is_method = inspect.ismethod(func_obj)
            qualname = get_function_qual_name(func_obj)
            call_args = inspect.getargvalues(frame)
            abstract_call_args = get_abstract_vals(call_args)
            # we keep track of the memory location of the function object
            # because it can allow us to establish a link between a line that calls
            # an function and the actual function call entry
            mem_loc_func = id(func_obj)

            details = dict(
                is_method          = is_method,
                qualname           = qualname,
                abstract_call_args = abstract_call_args,
                mem_loc_func       = mem_loc_func
                )
            event_id = self._allocate_event_id()
            trace_event = EnterCall(event_id, call_site_lineno, call_site_line, details)
            self.trace_events.append(trace_event)

        # functions that are defined by the user
        # will have line-level tracing
        if self._defined_by_user(frame):
            return self.trace
        else:
            # external functions only get the paired exit-call event
            log.info('Function is external, tracing with trace_external')
            return self.trace_external

    def trace_return(self, frame, event, arg):
        log.info('Trace return')
        if self._called_by_user(frame):
            log.info('Return is from a user-called function')
            caller_frame = get_caller_frame(frame)
            call_site_lineno = inspect.getlineno(caller_frame)
            call_site_line = self._getsource(caller_frame)
            details = None
            event_id = self._allocate_event_id()
            trace_event = ExitCall(event_id, call_site_lineno, call_site_line, details)
            self.trace_events.append(trace_event)

    def trace_stub(self, frame, event, arg):
        log.info('Tracing stub')
        stub_obj = get_function_obj(frame)
        stub_event = None
        if stub_obj == memory_update_stub:
            stub_event = self.consume_memory_update_stub(frame, event, arg)
        else:
            raise Exception("Unknown stub qualified name: %s" % get_function_qual_name(stub_obj))
        # remove stub from trace of events
        self.trace_events.pop()
        if stub_event:
            self.trace_events.append(stub_event)

    def consume_memory_update_stub(self, frame, event, arg):
        log.info('Consuming memory_update_stub call event')
        # extract line no, and remove stuff
        caller = frame.f_back
        # the actual line that triggered this stub call is one line up
        lineno = inspect.getlineno(caller) - 1
        arginfo = inspect.getargvalues(frame)
        if len(arginfo.args) != 2:
            log.error('memory_update_stub should only have 2 argumnets: list of names and list of values')
            arginfo.locals = {}
            log.error('ArgumentInfo: %s' % arginfo.args)
            raise TypeError('memory_update_stub should have 2 arguments')
        # memory locations that need to be updated
        names = arginfo.locals[arginfo.args[0]]
        values = arginfo.locals[arginfo.args[1]]
        # memory locations associated with those references
        memory_locations = {name:id(val) for name, val in zip(names, values)}
        event_id = self._allocate_event_id()
        trace_event = MemoryUpdate(event_id, memory_locations, lineno)
        return trace_event

    def setup(self):
        log.info('Setting up tracing function')
        self.orig_tracer = sys.gettrace()
        sys.settrace(self.trace)

    def shutdown(self):
        sys.settrace(self.orig_tracer)
        # Note: we call this log after
        # because otherwise the tracing gets triggered
        # for the logging function...
        # TODO: figure out better fix
        log.info('Shutting down tracing function')

    def add_stubs(self, src):
        log.info('Adding any stubs to provided source code')
        tree = ast.parse(src)
        ext_tree = AddMemoryUpdateStubs(stub_name=memory_update_stub.__qualname__).visit(tree)
        return astunparse.unparse(ext_tree)

    def clear(self):
        log.info('Clear internal state of tracer')
        self.file_path = None
        self.trace_events = []
        self.trace_errors = []
        self.event_counter = 0

    def run(self, file_path):
        log.info('Running tracer')
        self.clear()

        if os.path.exists(file_path):
            src = open(file_path).read()
        else:
            # we assume the path is actually source code
            src = file_path

        # modify source code as necessary to add any stubs
        src = self.add_stubs(src)
        self.src_lines = src.split('\n')
        # we put the instrumented code in a dummy file so that
        # certain lookups using `inspect` work as expected
        instrumented_file_path = '_instrumented.py'
        with open(instrumented_file_path, 'w') as f:
            f.write(src)

        self.file_path = instrumented_file_path

        # compile, execute instrumented version
        namespace = {
            '__name__'      : '__main__',
            '__file__'      : self.file_path,
            '__builtins__'  : __builtins__,
            # stubs
            memory_update_stub.__qualname__: memory_update_stub,
        }
        compiled = compile(src, filename=self.file_path, mode='exec')
        self.setup()
        exec(compiled, namespace)
        self.shutdown()


def main(args):
    src = open(args.input_path).read()
    tracer = DynamicDataTracer()
    tracer.run(src)
    with open(args.output_path, 'wb') as f:
        pickle.dump(tracer, f)


if __name__ == '__main__':
    parser = ArgumentParser(description='Execute lifted script with dynamic tracing')
    parser.add_argument('input_path', type=str, help='Path to lifted source')
    parser.add_argument('output_path', type=str, help='Path for pickled tracer with results')
    args = parser.parse_args()
    try:
        main(args)
    except:
        import pdb
        pdb.post_mortem()
