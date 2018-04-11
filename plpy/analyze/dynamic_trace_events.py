# Events for a dynamic trace

class TraceEvent(object):
    pass

class MemoryUpdate(TraceEvent):
    """
    Added to trace to update the last line assigning
    to a set of memory locations
    """
    def __init__(self, event_id, lineno, mem_locs):
        self.event_id = event_id
        self.mem_locs = mem_locs
        self.lineno = lineno
        self.line = None

    def __str__(self):
        return 'mem-update(%s)' % self.mem_locs

class ExecLine(TraceEvent):
    def __init__(self, event_id, lineno, line, uses_mem_locs):
        self.event_id = event_id
        self.lineno = lineno
        self.line = line
        self.uses_mem_locs = uses_mem_locs

    def __str__(self):
        return 'exec line: %s (line=%d)' % (self.line, self.lineno)

class EnterCall(TraceEvent):
    def __init__(self, event_id, call_site_lineno, call_site_line, details):
        self.event_id = event_id
        self.lineno = call_site_lineno
        self.line = call_site_line
        self.details = details

    def __str__(self):
        return 'enter call: %s (line=%d)' % (self.line, self.lineno)

class ExitCall(TraceEvent):
    def __init__(self, event_id, call_site_lineno, call_site_line, details):
        self.event_id = event_id
        self.lineno = call_site_lineno
        self.line = call_site_line
        self.details = details

    def __str__(self):
        return 'exit call: %s (line=%d)' % (self.line, self.lineno)

# mainly used to mark the abnormal end of tracing
class ExceptionEvent(TraceEvent):
    def __init__(self, event_id, msg):
        self.event_id = event_id
        self.msg = msg

    def __str__(self):
        return 'abnormal exit (exception)'