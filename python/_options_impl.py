# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import abc
from . import ops


class OptionsDict:
    """Safe dictionary to store options: only keys which have been passed to
    the constructor can later be updated.
    """

    def __init__(self, **default_values):
        self._values = default_values

    def set(self, **kwargs):
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

    def __getstate__(self):
        return self._values

    def __setstate__(self, state):
        self._values = state

    def __getattr__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]

    def update(self, other):
        assert not set(self._values.keys()).intersection(
            other), "Can't merge dictionaries, they have some keys in common"
        other.update(self._values)
        return other

    def toDict(self):
        return self.update({})

    def __call__(self, option):
        assert self.exists(
            option), ("Invalid option %s, "
                      "valid options are %s") % (option, self._values.keys())
        return self._values[option]


class IStageManager(abc.ABC):
    def __init__(self):
        self._next_auto_id = 0
        self._default_stage = None

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
        self._default_stage = ipu_id_from_block
        stage = self.getStage(user_id)
        # If the user specified an ipu_id in the option use that one
        ipu = stage._ipu if stage._ipu is not None else ipu_id_from_block  # pylint: disable=protected-access
        if ipu is None:
            ipu = stage._stage_id  # pylint: disable=protected-access
        ops.begin_ipu_block(stage._stage_id, stage._phase_id, ipu)  # pylint: disable=protected-access

    def resetAutoId(self):
        self._next_auto_id = 0
