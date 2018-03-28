# Trace events
class TraceEvent(object):
    pass

class MemoryUpdate(TraceEvent):
    """
    Added to trace to update the last line assigning
    to a set of memory locations
    """
    def __init__(self, event_id, mem_locs, lineno):
        self.event_id = event_id
        self.mem_locs = list(mem_locs)
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
    def __init__(self, event_id, call_site_lineno, call_site_line, stuff):
        self.event_id = event_id
        self.lineno = call_site_lineno
        self.line = call_site_line
        self.stuff = stuff

    def __str__(self):
        return 'enter call: %s (line=%d)' % (self.line, self.lineno)

class ExitCall(TraceEvent):
    def __init__(self, event_id, call_site_lineno, call_site_line, stuff):
        self.event_id = event_id
        self.lineno = call_site_lineno
        self.line = call_site_line
        self.stuff = stuff

    def __str__(self):
        return 'exit call: %s (line=%d)' % (self.line, self.lineno)