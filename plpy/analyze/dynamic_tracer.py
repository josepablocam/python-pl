# Add dynamic tracing to a script, records 'trace events'
# that can be used to build up data dependency information later on
from argparse import ArgumentParser
import ast
import astunparse
import copy
import inspect
import logging
import os
import pickle
import sys
import traceback

import numpy as np
import pandas as pd

from .dynamic_trace_events import *


# helper functions
def to_ast_node(line):
    return ast.parse(line).body[0]


def get_lineno(frame):
    return frame.f_lineno


def get_caller_frame(frame):
    if frame is None:
        return None
    return frame.f_back


def get_filename(frame):
    if frame is None:
        return None
    return frame.f_code.co_filename


def get_co_name(frame):
    return frame.f_code.co_name


def get_columns(obj):
    try:
        if isinstance(obj, pd.DataFrame):
            return obj.columns.values.tolist()
        elif isinstance(obj, pd.core.groupby.DataFrameGroupBy):
            return get_columns(obj.obj)
        elif isinstance(obj, pd.Series):
            name = obj.name
            if name is not None:
                return [name]
            else:
                return None
        elif isinstance(obj, pd.core.groupby.SeriesGroupBy):
            return get_columns(obj.obj)
        else:
            return None
    except AttributeError:
        return None


# this is not quite id, but allow us to avoid
# some issues with libraries that allocate new Python objects on accesses
# https://stackoverflow.com/questions/49782139/pandas-dataframes-series-and-id-in-cpython
# https://stackoverflow.com/questions/11264838/how-to-get-the-memory-address-of-a-numpy-array-for-c/11266170
# https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.ndarray.ctypes.html
def safe_id(obj):
    try:
        if isinstance(obj, pd.Series):
            # note can't call as obj.values because if uninitialized
            # (i.e. these are args to __init__ before call executes)
            # will result in infinite recursion due to way attribute lookups are implemented for pandas
            return obj.__getattribute__('values').ctypes.data
        elif isinstance(obj, np.ndarray):
            return obj.ctypes.data
        else:
            return id(obj)
    except AttributeError:
        # uninitialized pandas objects won't have values yet
        # when looking up values
        return id(obj)


# this is a horrendous hack... but not sure that there is a better way
def get_function_obj(frame, src_lines=None, filename=None):
    # can't get function object if we don't actually
    # have the frame
    if frame is None:
        log.debug('Call frame was None, cannot retrieve function object')
        return None
    # note that this hack is based on the info
    # described in
    # https://stackoverflow.com/questions/16589547/get-fully-qualified-method-name-from-inspect-stack
    try:
        parent = get_caller_frame(frame)
        if parent:
            log.debug('Retrieving function object through caller frame')
            log.debug('File name: %s' % get_filename(parent))
            # always return None if its a function called from the tracer itself
            if get_filename(parent) == __file__:
                return None
            if src_lines and get_filename(parent) == filename:
                src_line = src_lines[get_lineno(parent) - 1].strip()
            else:
                # note that inspect.getframeinfo is super expensive
                # so this is for completeness only...
                log.debug('Looking up function using frameinfo')
                src_line = inspect.getframeinfo(parent).code_context[0].strip()
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
    except (SyntaxError, AttributeError, IndexError):
        # not syntactically a call
        # this is possible if call event was triggered
        # for something like entering a class definition (when first defined)
        # or if attribute access has been defined to trigger a call (e.g. in pandas dataframes)
        log.exception('Failed to parse call')
        return None


def get_function_qual_name(obj):
    # relies on the function object
    if inspect.isframe(obj):
        log.debug('Checking if object is frame')
        obj = get_function_obj(obj)
    try:
        log.debug('Attempting to access __qualname__ for function object')
        return obj.__qualname__
    except AttributeError:
        log.warning('Function does not have __qualname__ attribute: %s' % obj)
        return None


def get_function_module(obj):
    if inspect.isframe(obj):
        log.debug('Object is stack frame')
        obj = get_function_obj(obj)
    try:
        log.debug('Attempting to access __module__ for function object')
        return obj.__module__
    except AttributeError:
        log.warning('Function does not have __module__ attribute')
        return None


def get_type_name(val):
    try:
        return type(val).__name__
    except AttributeError:
        return None


def get_abstract_vals(arginfo):
    arg_names = arginfo.args
    _locals = arginfo.locals
    info = []
    for name in arg_names:
        _var = Variable(
            name, safe_id(_locals[name]), get_type_name(_locals[name])
        )
        cols = get_columns(_locals[name])
        if cols is not None:
            _var.extra['columns'] = cols
        info.append(_var)
    return info


# stub functions that are inserted
# into source code to mark events for trace tracking
def memory_update_stub(names):
    pass


def loop_counter_init_stub(name):
    pass


def loop_counter_incr_stub(name):
    pass


def loop_counter_clear_stub(name):
    pass


STUBS = [
    memory_update_stub, loop_counter_init_stub, loop_counter_incr_stub,
    loop_counter_clear_stub
]


def is_stub_call(func_obj):
    return func_obj in STUBS


def is_comprehension(frame):
    comps = set(['<listcomp>', '<setcomp>', '<dictcomp>'])
    return frame.f_code.co_name in comps


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

    def visit_Subscript(self, node):
        self.acc.append(node)
        self.generic_visit(node)

    def run(self, node):
        self.visit(node)
        return [astunparse.unparse(ref).strip() for ref in self.acc]


def get_nested_references(node, exclude_first=False):
    refs = ExtractReferences().run(node)
    if exclude_first:
        refs = refs[1:]
    return set(refs)


class AddMemoryUpdateStubs(ast.NodeTransformer):
    def __init__(self, stub_name):
        self.stub_name = stub_name

    def visit_Assign(self, node):
        return node, self._create_stub_node(node.targets, treat_as_names=False)

    def visit_AugAssign(self, node):
        return node, self._create_stub_node([node.target],
                                            treat_as_names=False)

    def visit_Import(self, node):
        names = []
        for _alias in node.names:
            name = _alias.asname if _alias.asname else _alias.name
            names.append(name)
        return node, self._create_stub_node(names, treat_as_names=True)

    def visit_ImportFrom(self, node):
        _, stubs = self.visit_Import(ast.Import(node.names))
        return node, stubs

    def visit_With(self, node):
        # with binds names so we also need a memory update
        alias_nodes = []
        for withitem in node.items:
            alias = withitem.optional_vars
            if alias:
                alias_nodes.append(alias.id)
        # add in the memory update inside the with
        stub_node = self._create_stub_node(alias_nodes, treat_as_names=True)
        node.body = [stub_node] + node.body
        # add in stubs to body as necessary
        return self.generic_visit(node)

    def _create_stub_node(self, targets, treat_as_names=False):
        references = []
        # if we don't have all the names already, extract them
        if not treat_as_names:
            for tgt in targets:
                references.extend(
                    get_nested_references(tgt, exclude_first=False)
                )
        else:
            references.extend(targets)
        # always produce them in ascending order of reference string length
        references = sorted(references)
        names_str = ','.join(['"%s"' % ref for ref in references])
        call_str = f'{self.stub_name}([{names_str}])'
        return to_ast_node(call_str)


class AddLoopCounterStubs(ast.NodeTransformer):
    def __init__(
        self,
        init_stub_name,
        incr_stub_name,
        clear_stub_name,
        loop_var_format='_loop_var_%d'
    ):
        self.init_stub_name = init_stub_name
        self.incr_stub_name = incr_stub_name
        self.clear_stub_name = clear_stub_name
        self.loop_counter = 0
        self.loop_var_format = loop_var_format

    def _allocate_loop_var(self):
        loop_var = self.loop_var_format % self.loop_counter
        self.loop_counter += 1
        return loop_var

    def _add_loop_stubs(self, loop_node):
        # right before
        loop_var = self._allocate_loop_var()
        init_stub = self._create_stub_node(self.init_stub_name, loop_var)
        incr_stub = self._create_stub_node(self.incr_stub_name, loop_var)
        clear_stub = self._create_stub_node(self.clear_stub_name, loop_var)
        loop_node.body.append(incr_stub)
        results = [init_stub]
        recursive_nodes = self.generic_visit(loop_node)
        try:
            results.extend(recursive_nodes)
        except TypeError:
            results.append(recursive_nodes)
        results.append(clear_stub)
        return results

    def visit_While(self, node):
        return self._add_loop_stubs(node)

    def visit_For(self, node):
        return self._add_loop_stubs(node)

    def _create_stub_node(self, stub_name, loop_var):
        call_str = f'{stub_name}("{loop_var}")'
        return to_ast_node(call_str)


class DynamicDataTracer(object):
    def __init__(self, loop_bound=None):
        # information for program being traced
        self.file_path = None
        self.src_lines = None
        # information for trace events
        self.event_counter = 0
        self.trace_events = []
        self.trace_errors = []
        self.orig_tracer = None
        # information to manage tracing on/off
        self.traced_stack_depth = 0
        self.watch_frame = None
        self.loop_bound = loop_bound
        self.loop_counters = {}
        self.ignored_loops = set([])

    def _allocate_event_id(self):
        log.info('Allocated new event id')
        _id = self.event_counter
        self.event_counter += 1
        return _id

    def _called_by_user(self, frame):
        """ only trace calls to functions directly invoked by the user """
        called = get_filename(get_caller_frame(frame)) == self.file_path
        # log.info('Checking if user called function: %s' % called)
        return called

    def _called_by_tracer(self, frame):
        called = get_filename(get_caller_frame(frame)) == __file__
        return called

    def _defined_by_user(self, frame):
        """ only trace lines inside body of functions that are defined by user in same file """
        defined = get_filename(frame) == self.file_path
        # log.info('Checking if frame is for user defined function: %s' % defined)
        return defined

    def _defined_by_tracer(self, frame):
        defined = get_filename(frame) == __file__
        return defined

    def _getsource(self, frame):
        # we strip before returning the line as
        # we often use this line to parse a node
        # and any kind of indentation will lead to SyntaxError raised in ast.parse
        if self._defined_by_user(frame):
            log.info('Function is defined by user, fetching source')
            lineno = inspect.getlineno(frame)
            return self.src_lines[lineno - 1].strip()
        # try using inspect if not
        try:
            return inspect.getsource(frame).strip()
        except IOError:
            log.exception('Unable to retrieve source for frame')
            return None

    def trace(self, frame, event, arg):
        try:
            # we are waiting to return to a user code frame
            if self.watch_frame:
                if frame != self.watch_frame:
                    # we let return's continue to pair with the enter call for non-user function
                    if event != 'return':
                        # ignore the rest of this if the event was not a return and frame doesn't match watched
                        return None
                else:
                    # returned to the point in the stack we wanted, so clear the watch
                    self.watch_frame = None

            if not self._called_by_user(frame) and not self._called_by_tracer(
                    frame):
                # don't instrument if we're executing code that was not called by the user or tracer itself
                # note that it is possible for non-user code to invoke user-defined code (e.g. a lambda defined by user)
                # so this check avoids cases where we would not instrument the body of third party library function
                # but each call in the body to a user-defined function would result in a trace event
                #
                # look for the last frame called by the user and wait there
                current_frame = frame
                while current_frame and not self._called_by_user(current_frame
                                                                 ):
                    current_frame = get_caller_frame(current_frame)
                if current_frame:
                    self.watch_frame = current_frame
                    log.info('Code not called by user or tracer')
                    log.info('At frame: %s' % str(inspect.getframeinfo(frame)))
                    log.info(
                        'Setting as watch frame: %s' %
                        str(inspect.getframeinfo(self.watch_frame))
                    )
                return None

            # the user can call functions they don't define (e.g. third party libraries)
            # if this is the case, then, set the watch frame as the last frame
            # in code that they defined
            if event == 'call' and not self._defined_by_user(frame):
                current_frame = frame
                while current_frame and not self._defined_by_user(current_frame
                                                                  ):
                    current_frame = get_caller_frame(current_frame)
                if current_frame:
                    self.watch_frame = current_frame
                    log.info('Call for code not defined by user')
                    log.info('At frame: %s' % str(inspect.getframeinfo(frame)))
                    log.info(
                        'Setting as watch frame: %s' %
                        str(inspect.getframeinfo(self.watch_frame))
                    )

            # don't record info inside a comprehension
            # this way we only get the initial line event
            # but EnterCall, body, ExitCall are not recorded
            if is_comprehension(frame):
                return self.trace

            if event == 'call':
                return self.trace_call(frame, event, arg)

            if event == 'line':
                return self.trace_line(frame, event, arg)

            if event == 'return':
                return self.trace_return(frame, event, arg)

        except Exception as err:
            log.exception("Exception raised by tracer code")
            self.add_exception_event(traceback.format_exc())
            self.shutdown()

    # events can only be pushed or popped from
    # our tracer if we are not ignoring due to loops
    # note that this still traces, just doesn't record the events
    def push_trace_event(self, event):
        if not self.ignored_loops:
            self.trace_events.append(event)

    def pop_trace_event(self):
        if not self.ignored_loops:
            return self.trace_events.pop()
        return None

    def convert_with_items_to_lines(self, line):
        log.info(
            'Converting with-statement items to separate lines for tracer'
        )
        # a dummy to actually parse this
        with_dummy = line + '\n\tpass'
        with_node = to_ast_node(with_dummy)
        lines = []
        for withitem in with_node.items:
            rhs = astunparse.unparse(withitem.context_expr).strip()
            if withitem.optional_vars:
                lhs = astunparse.unparse(withitem.optional_vars).strip()
                line = f'{lhs} = {rhs}'
            else:
                line = rhs
            lines.append(line)
        return lines

    def trace_line(self, frame, event, arg):
        log.info('Trace line')
        # Note that any C-based function (e.g. max/len etc)
        # won't actually trigger a call, so we need to see what to do here
        line = self._getsource(frame)
        log.info('Line: %s' % line)
        event_id = self._allocate_event_id()
        try:
            # some statements we break up and treat as multiple lines
            lines = [line]
            if line.startswith('with'):
                # this is actually parseable
                lines = self.convert_with_items_to_lines(line)
            # using these lines, get any load references
            load_references = []
            for l in lines:
                if len(l.strip()) > 0:
                    load_references.extend(
                        self.get_load_references_from_line(l)
                    )
            # using the load references, get the memory locations
            uses = self.get_mem_locs(
                load_references, frame, include_global_references=True
            )
            # associate all these with the initial line/lineno and store the event
            trace_event = ExecLine(
                event_id, inspect.getlineno(frame), line, uses
            )
            log.info('Appending trace event: %s' % trace_event)
            self.push_trace_event(trace_event)
        except SyntaxError:
            log.exception('Syntax error while tracing line: %s' % line)
            # self.trace_errors.append((frame.f_code.co_name, event, arg, line))
        return self.trace

    def get_load_references_from_line(self, line):
        log.info('Get references from line')
        assert line is not None
        node = to_ast_node(line)

        # assignments can trigger loads as well
        # attribute: a.b = ... => loads(a), loads(a.b)
        # subscript: val[ix][...] = ... => loads(val), loads(val[ix])
        lhs_nodes = []
        rhs_nodes = []

        if isinstance(node, ast.Assign):
            rhs_nodes.append(node.value)
            lhs_nodes.extend(node.targets)
        elif isinstance(node, ast.AugAssign):
            rhs_nodes.append(node.value)
            rhs_nodes.append(node.target)
        else:
            rhs_nodes.append(node)

        references = []
        for target in lhs_nodes:
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                # in this case we don't want to have as a load the deepest access
                # e.g. a.b.c = ... => load(a), load(a.b) (but not load(a.b.c))
                references.extend(
                    get_nested_references(target, exclude_first=True)
                )

        # RHS of line (or original node if just expression)
        for val_node in rhs_nodes:
            references.extend(get_nested_references(val_node))

        return set(references)

    def get_mem_locs(
        self, str_references, frame, include_global_references=False
    ):
        """
        Load the memory locations for these references using the environment
        available to the frame provided
        """
        log.info('Get memory locations for references')

        _locals = frame.f_locals
        _globals = frame.f_globals
        mem_locs = set([])
        for ref in str_references:
            try:
                obj = eval(ref, _globals, _locals)
                var = Variable(ref, safe_id(obj), get_type_name(obj))
                cols = get_columns(obj)
                if cols is not None:
                    var.extra['columns'] = cols
                mem_locs.add(var)

                # we can extend with global references where relevant
                if include_global_references and (inspect.isfunction(obj)
                                                  or inspect.ismethod(obj)):
                    global_locs = self.get_globals_mem_locs_in_user_defined_function(
                        frame, obj
                    )
                    mem_locs.update(global_locs)
            except (NameError, AttributeError):
                var = Variable(ref, None, None)
                mem_locs.add(var)
            except:
                # we can't do anything here, so may as well just continue
                pass
        return mem_locs

    def get_globals_mem_locs_in_user_defined_function(self, frame, obj):
        if obj.__code__.co_filename != self.file_path:
            return {}
        else:
            # given static scoping in Python
            # we can check global vars based on the caller for the current
            # frame and get memory locations for any nested global references
            _globals = frame.f_globals
            _locals = frame.f_locals
            global_refs = [obj]
            mem_locs = set([])
            while global_refs:
                curr_ref = global_refs.pop()
                try:
                    code = curr_ref.__code__
                except AttributeError:
                    continue
                # global references
                for var in code.co_names:
                    try:
                        var_obj = eval(var, _globals, _locals)
                        is_func = inspect.isfunction(
                            var_obj
                        ) or inspect.ismethod(var_obj)

                        # ignore anything coming from tracer
                        if is_func and var_obj.__code__.co_filename == __file__:
                            continue

                        # for classes, look at the constructor
                        if inspect.isclass(var_obj):
                            var_obj = var_obj.__init__

                        summary = Variable(
                            var, safe_id(var_obj), get_type_name(var_obj)
                        )
                        cols = get_columns(var_obj)
                        if cols is not None:
                            summary.extra['columns'] = cols
                        mem_locs.add(summary)

                        # search recursively inside user functions/methods
                        if is_func and var_obj.__code__.co_filename == self.file_path:
                            global_refs.append(var_obj)

                    except (NameError, AttributeError):
                        pass
            return mem_locs

    def trace_call(self, frame, event, arg):
        log.info('Trace call (co_name=%s)' % get_co_name(frame))

        # retrieve the object for the function being called
        func_obj = get_function_obj(
            frame, src_lines=self.src_lines, filename=self.file_path
        )

        # either not a function call (e.g. entered code block)
        # or couldn't retrieve function source to get object
        if func_obj is None:
            log.info(
                'Function object was None for source: %s' %
                self._getsource(frame)
            )
            if self._defined_by_user(frame):
                log.info('Tracing as defined by user')
                return self.trace
            else:
                log.info(
                    'Ignoring and treating as external since not defined by user'
                )
                log.info('Function from frame: %s' % get_co_name(frame))
                return None

        if is_stub_call(func_obj):
            log.info('Function object is a stub: %s' % get_co_name(frame))
            return self.trace_stub(func_obj, frame, event, arg)

        log.info('Collecting call made by user for: %s' % func_obj)
        # # increase the depth of the traced stack
        self.traced_stack_depth += 1
        caller_frame = get_caller_frame(frame)
        call_site_lineno = inspect.getlineno(caller_frame)
        call_site_line = self._getsource(caller_frame)

        # call details
        is_method = inspect.ismethod(func_obj)
        co_name = get_co_name(frame)
        qualname = get_function_qual_name(func_obj)
        module = get_function_module(func_obj)
        call_args = inspect.getargvalues(frame)
        abstract_call_args = get_abstract_vals(call_args)
        # we keep track of the memory location of the function object
        # because it can allow us to establish a link between a line that calls
        # an function and the actual function call entry
        mem_loc_func = safe_id(func_obj)
        details = dict(
            is_method=is_method,
            co_name=co_name,
            qualname=qualname,
            module=module,
            abstract_call_args=abstract_call_args,
            mem_loc_func=mem_loc_func,
            called_by_user=self._called_by_user(frame),
            defined_by_user=self._defined_by_user(frame),
        )
        event_id = self._allocate_event_id()
        trace_event = EnterCall(
            event_id, call_site_lineno, call_site_line, details
        )
        log.info('Appending trace event: %s' % trace_event)
        self.push_trace_event(trace_event)

        return self.trace

    def trace_return(self, frame, event, arg):
        log.info('Trace return (co_name=%s)' % get_co_name(frame))
        # we need to make sure we are expecting a return event
        # since things like exiting a class def produces a return event
        if self.traced_stack_depth > 0:
            log.info(
                'Return from a user-called function: %s' % frame.f_code.co_name
            )
            self.traced_stack_depth -= 1
            caller_frame = get_caller_frame(frame)
            call_site_lineno = inspect.getlineno(caller_frame)
            call_site_line = self._getsource(caller_frame)
            details = dict(
                co_name=get_co_name(frame),
                called_by_user=self._called_by_user(frame),
                defined_by_user=self._defined_by_user(frame),
            )
            event_id = self._allocate_event_id()
            trace_event = ExitCall(
                event_id, call_site_lineno, call_site_line, details
            )
            log.info('Appending trace event: %s' % trace_event)
            self.push_trace_event(trace_event)

    def trace_stub(self, stub_obj, frame, event, arg):
        log.info('Tracing stub')
        if stub_obj.__name__ != get_co_name(frame):
            # stub objects are guaranteed to have the co_name match the object name
            # if this is not the case, then this is not actually a stub
            # this can happen for example if there is a stub
            # as the last line in a with statement.
            # when __exit__ is called, it raises a call. We try to retrieve
            # function object by looking at the source line that is the caller
            # this could be a stub line, and then that function gets retrieved as an object
            # despite the mismatch
            return None
        if stub_obj == memory_update_stub:
            self.consume_memory_update_stub(frame, event, arg)
        elif stub_obj == loop_counter_init_stub:
            self.consume_loop_counter_init_stub(frame, event, arg)
        elif stub_obj == loop_counter_incr_stub:
            self.consume_loop_counter_incr_stub(frame, event, arg)
        elif stub_obj == loop_counter_clear_stub:
            self.consume_loop_counter_clear_stub(frame, event, arg)
        else:
            raise Exception(
                "Unhandled stub qualified name: %s" %
                get_function_qual_name(stub_obj)
            )

    def consume_memory_update_stub(self, frame, event, arg):
        log.info('Consuming memory_update_stub call event')
        # extract line no, and remove stuff
        caller = frame.f_back
        # the actual line that triggered this stub call is one line up
        lineno = inspect.getlineno(caller) - 1
        arginfo = inspect.getargvalues(frame)
        # memory locations that need to be updated
        names = arginfo.locals[arginfo.args[0]]
        # memory locations associated with those references by looking for the globals/locals in the caller frame
        defs = self.get_mem_locs(names, get_caller_frame(frame))
        event_id = self._allocate_event_id()
        trace_event = MemoryUpdate(event_id, lineno, defs)
        self.pop_trace_event()
        # # we should add the defs from memory update to execline before it
        # line_event = self.pop_trace_event()
        # if line_event is not None:
        #     line_event.defs = copy.deepcopy(defs)
        #     self.push_trace_event(line_event)
        self.push_trace_event(trace_event)

    def check_loop_name(self, loop_name):
        # we are bounding loops and this loop has exceeded the limit set by user
        if self.loop_bound is not None and self.loop_counters.get(
                loop_name, 0) >= self.loop_bound:
            log.debug('Adding loop variable %s to ignored loops' % loop_name)
            self.ignored_loops.add(loop_name)
            # turn off non essential logging
            log.setLevel(logging.CRITICAL)

    def consume_loop_counter_init_stub(self, frame, event, arg):
        log.info('Consuming loop_counter_init call event')
        arginfo = inspect.getargvalues(frame)
        loop_name = arginfo.locals[arginfo.args[0]]
        log.info('Init loop counter at %s' % loop_name)
        self.loop_counters[loop_name] = 0
        self.check_loop_name(loop_name)
        self.pop_trace_event()

    def consume_loop_counter_incr_stub(self, frame, event, arg):
        log.info('Consuming loop_counter_incr call event')
        arginfo = inspect.getargvalues(frame)
        loop_name = arginfo.locals[arginfo.args[0]]
        log.info('Increasing loop counter at %s' % loop_name)
        self.loop_counters[loop_name] += 1
        self.check_loop_name(loop_name)
        self.pop_trace_event()

    def consume_loop_counter_clear_stub(self, frame, event, arg):
        log.info('Consume loop_counter_clear call event')
        arginfo = inspect.getargvalues(frame)
        loop_name = arginfo.locals[arginfo.args[0]]
        log.info('Clearing loop counter at %s' % loop_name)
        self.loop_counters[loop_name] = 0
        self.ignored_loops.discard(loop_name)
        self.pop_trace_event()
        if not self.ignored_loops:
            log.setLevel(LOG_LEVEL)

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
        ext_tree = AddMemoryUpdateStubs(
            stub_name=memory_update_stub.__qualname__
        ).visit(tree)
        loop_stubber = AddLoopCounterStubs(
            init_stub_name=loop_counter_init_stub.__qualname__,
            incr_stub_name=loop_counter_incr_stub.__qualname__,
            clear_stub_name=loop_counter_clear_stub.__qualname__
        )
        ext_tree = loop_stubber.visit(ext_tree)
        return astunparse.unparse(ext_tree)

    def add_exception_event(self, msg):
        event_id = self._allocate_event_id()
        trace_event = ExceptionEvent(event_id, msg)
        log.debug('Appending exception event before tracer exit')
        self.trace_events.append(trace_event)

    def clear(self):
        log.info('Clear internal state of tracer')
        # for program that will be traced
        self.file_path = None
        self.src_lines = []
        self.trace_events = []
        self.trace_errors = []
        self.event_counter = 0
        # keep track of what to trace and not
        self.traced_stack_depth = 0
        self.watch_frame = None
        # loop bounding
        self.loop_counters = {}
        self.ignored_loops = set([])

    def run(self, file_path):
        log.info('Running tracer')
        self.clear()

        if os.path.exists(file_path):
            src = open(file_path).read()
        else:
            # we assume the path is actually source code
            src = file_path

        if len(src.strip()) == 0:
            # empty file
            return

        # modify source code as necessary to add any stubs
        src = self.add_stubs(src)
        self.src_lines = src.split('\n')
        # we put the instrumented code in a dummy file so that
        # certain lookups using `inspect` work as expected
        instrumented_file_path = '_instrumented.py'
        with open(instrumented_file_path, 'w') as f:
            f.write(src)
        self.file_path = instrumented_file_path
        # add in additional mappings
        _globals = {}
        _globals['__name__'] = '__main__'
        _globals['__file__'] = self.file_path
        _globals['__builtins__'] = __builtins__
        # add stubs to namespace
        for stub in STUBS:
            _globals[stub.__qualname__] = stub
        compiled = compile(src, filename=self.file_path, mode='exec')
        try:
            self.setup()
            # don't provide _locals based on
            # https://stackoverflow.com/questions/39647566/why-does-python-3-exec-fail-when-specifying-locals
            # otherwise we don't get the correct scope for exec
            exec(compiled, _globals)
        except:
            # don't just catch Exception, may have other errors
            self.shutdown()
            log.exception('Error during script execution')
            exception_triplet = sys.exc_info()
            self.add_exception_event(
                traceback.format_exception(*exception_triplet)
            )
            raise exception_triplet[1]
        finally:
            self.watch_frame = None
            self.shutdown()


def main(args):
    src = open(args.input_path).read()
    tracer = DynamicDataTracer(loop_bound=args.loop_bound)
    tracer.run(src)
    with open(args.output_path, 'wb') as f:
        pickle.dump(tracer, f)
    # print(list(map(str, tracer.trace_events)))


def setup_logger(filename, level):
    logging.basicConfig(
        filename=filename,
        level=level,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
    return logging.getLogger(__name__)


LOG_FILE = None
LOG_LEVEL = logging.CRITICAL

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Execute lifted script with dynamic tracing'
    )
    parser.add_argument('input_path', type=str, help='Path to lifted source')
    parser.add_argument(
        'output_path', type=str, help='Path for pickled tracer with results'
    )
    parser.add_argument(
        '-l',
        '--log',
        type=str,
        help='Path for logging file (slows down tracing significantly)'
    )
    parser.add_argument(
        '-b', '--loop_bound', type=int, help='Loop bound for tracing'
    )
    args = parser.parse_args()
    if args.log:
        LOG_LEVEL = logging.DEBUG
        LOG_FILE = args.log
    log = setup_logger(LOG_FILE, LOG_LEVEL)
    main(args)
else:
    log = setup_logger(LOG_FILE, LOG_LEVEL)
