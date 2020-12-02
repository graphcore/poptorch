# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import os
from .logging import logger

if os.environ.get("PVTI_OPTIONS") is None:
    _pvti_available = False
else:
    try:
        import libpvti as pvti
        _pvti_available = True
    except ImportError as e:
        logger.info("Tracepoints disabled (Couldn't import libpvti: %s)")
        _pvti_available = False


class Channel:
    """Profiling channel.

    .. note:: If the ``libpvti`` profiling library is not available at runtime
        this class becomes a no-op.

    Example:

    >>> channel = poptorch.profiling.Channel("MyApp")
    >>> with channel.tracepoint("TimeThis"):
    ...     functionToTime()
    >>> channel.instrument(myobj, "methodName", "otherMethod")
    """

    def __init__(self, name):
        if _pvti_available:
            self._tracepoint_prefix = name
            self._channel = _Channels.getOrCreate(name)

    def instrument(self, obj, *methods):
        """Instrument the methods of an object.

        :param obj: Object to instrument
        :param methods: One or more methods to wrap in profiling tracepoints.
        """
        if _pvti_available:
            pvti.instrument(obj, methods, self._channel)
        return obj

    def tracepoint(self, name):
        """Create a context tracepoint

        >>> with channel.tracepoint("DoingSomething"):
        ...     expensiveCall()

        :param name: Name associated to this tracepoint.
        """
        if _pvti_available:
            tracepoint_name = self._tracepoint_prefix + "." + name
            return pvti.Tracepoint(self._channel, tracepoint_name)
        return _DummyTracepoint()


class _DummyTracepoint:
    """Dummy context used when pvti is not available"""

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


class _Channels:
    """Singleton library of registered Channels"""
    _channels = {}

    @staticmethod
    def getOrCreate(name):
        if name not in _Channels._channels:
            _Channels._channels[name] = pvti.createTraceChannel(name)
        return _Channels._channels.get(name)
