# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import abc
import copy
import logging
import datetime
import torch
import tqdm

from ._logging import logger

_begin_ipu_block = torch.ops.poptorch.begin_ipu_block

# Disable tqdm locks: this might cause some visual artifacts
# in the console but this will prevent dead locks in multiprocessing
# applications.
# https://github.com/tqdm/tqdm/issues/461#issuecomment-334343230
tqdm.tqdm.get_lock().locks = []


class ProgressBar:
    def __init__(self):
        self.compilation_time = None
        self._start_time = None
        self._bar = None
        self._last = 0

    def __call__(self, progress: int, total: int):
        if self._bar is None:
            self._start_time = datetime.datetime.now()
            # Remove {rate_fmt}{postfix} from the default format
            # as it doesn't really make sense for a compilation process
            #
            # Note: this is *not* a f-string
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            bar_format += "[{elapsed}<{remaining}]"
            self._bar = tqdm.tqdm(desc="Graph compilation",
                                  total=total,
                                  bar_format=bar_format)
        self._bar.update(progress - self._last)
        self._last = progress
        self.compilation_time = datetime.datetime.now() - self._start_time
        if progress == total:
            self._bar.close()
            self._bar = None
            self.compilation_time = datetime.datetime.now() - self._start_time


class OptionsDict:
    """Safe dictionary to store options: only keys which have been passed to
    the constructor can later be updated.
    """

    def __init__(self, **default_values):
        # Keep a dictionary of warnings messages based on the parameter options:
        # these are printed when the dictionaries are consolidated. The use of a
        # dictionary allows a warning to be removed by the key, e.g. if there is
        # a warning that the default parameter has changed but the parameter is
        # specified explicitly.
        self._warnings = {}

        # Allow warnings to be disabled by adding them to the list
        self._warnings_disabled = set()

        # Option object will be frozen after first use.
        self._is_frozen = False

        # _values must be the last attribute set in the __init__
        self._values = default_values

    def set(self, **kwargs):
        self.checkIsFrozen()
        for option, value in kwargs.items():
            assert self.exists(option), ("Invalid option %s, valid options"
                                         " are %s") % (option,
                                                       self._values.keys())
            assert isinstance(
                value, type(self._values[option])
            ), "Unexpected type %s for option %s. Expected %s" % (
                type(value), option, type(self._values[option]))
            self._values[option] = value

    def createOrSet(self, **kwargs):
        self.checkIsFrozen()
        for option, value in kwargs.items():
            if option in self._values:
                self.set(**{option: value})
            else:
                self._values[option] = value

    def exists(self, option):
        return option in self._values

    def deleteIfExists(self, option):
        if self.exists(option):
            del self._values[option]

    def _hasattr(self, option):
        if option == "__class__":
            return True
        if option.startswith("_"):
            return option in self.__getstate__().keys()
        return self.exists(option)

    # pylint: disable=protected-access
    def _changeFreezeState(self, new_state):
        self._is_frozen = new_state
        for _, value in self.__dict__.items():
            if isinstance(value, OptionsDict):
                if value._hasattr('_is_frozen'):
                    value._is_frozen = new_state
            else:
                if hasattr(value, '_is_frozen'):
                    value._is_frozen = new_state

    def _freeze(self):
        self._changeFreezeState(True)

    def _unfreeze(self):
        self._changeFreezeState(False)

    def checkIsFrozen(self, option=None):
        # Skip check during object initialization.
        if self._hasattr('_is_frozen'):
            if option != '_is_frozen' and self._is_frozen:
                raise AttributeError("Can't modify frozen Options")

    def __deepcopy__(self, memory):
        opts_class = self.__class__
        copied_options = opts_class.__new__(opts_class)
        memory[id(self)] = copied_options
        for key, val in self.__dict__.items():
            if key == '_is_frozen':
                val = False
            setattr(copied_options, key, copy.deepcopy(val, memory))
        return copied_options

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __setattr__(self, option, value):
        # Private attributes are allowed, but should be set in the __init__ before _values
        # public ones must be declared in default_values.

        self.checkIsFrozen(option)
        if option.startswith("_"):
            # Option cannot be defined after _values definition.
            if self._hasattr('_values') and not self._hasattr(option):
                raise AttributeError(
                    f"Invalid private attribute {option}. "
                    f"Valid attributes: {list(self.__dict__.keys())}")
            super().__setattr__(option, value)
        else:
            self.set(**{option: value})

    def __getattr__(self, option):
        if not self._hasattr(option):
            raise AttributeError(f"Invalid attribute {option}.")
        if option.startswith("_"):
            return self.__getstate__()[option]
        return self._values[option]

    def update(self, other):
        for warning in self._warnings.values():
            if warning not in self._warnings_disabled:
                logger.warning(warning)

        assert not set(self._values.keys()).intersection(
            other), "Can't merge dictionaries, they have some keys in common"
        other.update(self._values)
        return other

    def toDict(self):
        return self.update({})

    def __call__(self, option):
        return getattr(self, option)

    def __repr__(self):
        # Call __repr__ on v so that strings display with quotes.
        return (f"{type(self).__name__}(" +
                ", ".join(f"{k}={v.__repr__()}"
                          for k, v in self._values.items()) + ", " +
                ", ".join(f"{k}={v.__repr__()}"
                          for k, v in self.__dict__.items()
                          if k != "_values") + ")")


default_source_location_excludes = [
    "install/poptorch", "site-packages/torch", "site-packages/poptorch"
]


class IStageManager(abc.ABC):
    def __init__(self):
        self._next_auto_id = 0
        self._current_ipu = None
        # We expect Torch to trace the graph 3 times, so to avoid printing
        # the same messages several times we store all the messages and
        # print the first third of them at the end.
        self._debug_messages = []

    def clearDebug(self):
        self._debug_messages = []

    def _debug(self, *args):
        if logger.isEnabledFor(logging.DEBUG):
            self._debug_messages.append(args)

    def printDebug(self):
        n = len(self._debug_messages)
        # We assume the graph was traced 3 times if:
        # - Number of messages can be divided by 3
        # - The first message is identical to the n/3th and 2n/3th ones.
        is_triple_trace = n > 0 and n % 3 == 0\
                and self._debug_messages[0] == self._debug_messages[n//3] \
                == self._debug_messages[2*n//3]
        if is_triple_trace:
            for i in range(n // 3):
                logger.debug(*self._debug_messages[i])
        else:
            # Not sure what happened: in doubt print everything
            for m in self._debug_messages:
                logger.debug(m)

    def nextAutoId(self):
        id = self._next_auto_id
        self._next_auto_id += 1
        return str(id)

    @abc.abstractmethod
    def getStage(self, block_id):
        """Return the stage corresponding to the given block_id.
        """

    def beginStage(self, user_id, ipu_id_from_block):
        user_id = user_id or self.nextAutoId()
        self._current_ipu = ipu_id_from_block
        stage = self.getStage(user_id)
        # If the user specified an ipu_id in the option use that one
        ipu = stage._ipu if stage._ipu is not None else ipu_id_from_block  # pylint: disable=protected-access
        if ipu is None:
            self._debug(
                "No IPU specified for block %s: default to stage_id %d",
                user_id, stage._stage_id)  # pylint: disable=protected-access
            ipu = stage._stage_id  # pylint: disable=protected-access
        self._debug("Starting block id=%s stage=%d phase=%d ipu=%d", user_id,
                    stage._stage_id, stage._phase_id, ipu)  # pylint: disable=protected-access
        _begin_ipu_block(stage._stage_id, stage._phase_id, ipu)  # pylint: disable=protected-access

    def resetAutoId(self):
        self._next_auto_id = 0
