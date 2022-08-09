# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest

import poptorch


# pylint: disable=import-outside-toplevel
def test_blocked_options():
    import popdist.poptorch
    opts = popdist.poptorch.Options(ipus_per_replica=2)

    with pytest.raises(
            RuntimeError,
            match=r"Cannot call `useIpuId` with popdist\.poptorch\.Options"):
        opts.useIpuId(1)

    with pytest.raises(RuntimeError,
                       match=r"Cannot call `replicationFactor` with "
                       r"popdist\.poptorch\.Options"):
        opts.replicationFactor(1)

    with pytest.raises(RuntimeError,
                       match=r"Cannot call `Distributed.disable` with "
                       r"popdist\.poptorch\.Options"):
        opts.Distributed.disable()

    with pytest.raises(RuntimeError,
                       match=r"Cannot call `Distributed.setEnvVarNames` with "
                       r"popdist\.poptorch\.Options"):
        opts.Distributed.setEnvVarNames("A", "B")

    with pytest.raises(
            RuntimeError,
            match=r"Cannot call `Distributed.configureProcessId` with "
            r"popdist\.poptorch\.Options"):
        opts.Distributed.configureProcessId(1)


# pylint: disable=import-outside-toplevel
def test_getters():
    import popdist.poptorch
    opts = popdist.poptorch.Options(ipus_per_replica=2)

    assert opts.Distributed.processId == 0
    assert opts.Distributed.numProcesses == 1


# pylint: disable=protected-access,import-outside-toplevel
@pytest.mark.ipuHardwareRequired
def test_to_dict():
    import popdist.poptorch
    opts = popdist.poptorch.Options(ipus_per_replica=2)
    opts.outputMode(poptorch.enums.OutputMode.All)
    opts.toDict()

    # Should not be frozen here
    opts.checkIsFrozen()

    opts._freeze()

    # Should unfeeze and freeze again
    opts.toDict()

    with pytest.raises(AttributeError, match=r"Can't modify frozen Options"):
        opts.checkIsFrozen()
