# Events for a dynamic trace
import inspect


class TraceEvent(object):
    def data(self):
        fields = dir(self)
        _data = {}
        for field in fields:
            if field.startswith('_'):
                continue
            obj = getattr(self, field)
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                continue
            _data[field] = obj
        return _data


class MemoryUpdate(TraceEvent):
    """
    Added to trace to update the last line assigning
    to a set of memory locations
    """
    def __init__(self, event_id, lineno, defs):
        self.event_id = event_id
        self.defs = list(defs)
        self.lineno = lineno
        self.line = None

    def __str__(self):
        return 'mem-update(%s)' % self.defs


class Variable(object):
    def __init__(self, name, _id, _type):
        self.name = name
        self.id = _id
        # really type.__name__ to pickle w/o issues
        self.type = _type
        self.extra = {}

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return (self.name, self.id) == (other.name, other.id)

    def __hash__(self):
        return hash((self.name, self.id))

    def __str__(self):
        return f'{self.name}: {self.type} @ {self.id}'

    def __repr__(self):
        return str(self)


class ExecLine(TraceEvent):
    def __init__(self, event_id, lineno, line, uses):
        self.event_id = event_id
        self.lineno = lineno
        self.line = line
        self.uses = list(uses)
        self.defs = []

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
