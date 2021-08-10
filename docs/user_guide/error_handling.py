# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import torch
import poptorch
from poptorch.poptorch_core import TestErrorType


# pragma pylint: disable=broad-except
# This is a fake model which actually throws an exception
class PytorchModel(torch.nn.Module):
    def __init__(self, error):
        super().__init__()
        if error is not None:
            poptorch.poptorch_core._throwTestError(error)

    def forward(self, x, y):
        return x + y


def run_example(model_param=None):

    rebooted = False
    shutdown = False

    def reboot_server():
        nonlocal rebooted
        rebooted = True

    def shutdown_system():
        nonlocal shutdown
        shutdown = True

    # error_handling_start
    try:
        m = PytorchModel(model_param)
        inference_model = poptorch.inferenceModel(m)
        t1 = torch.tensor([1.])
        t2 = torch.tensor([2.])
        assert inference_model(t1, t2) == 3.0
    except poptorch.RecoverableError as e:
        print(e)
        if e.recovery_action == "FULL_RESET":
            reboot_server()
        elif e.recovery_action == "IPU_RESET":
            print("Need to reset the IPU")
        elif e.recovery_action == "PARITION_RESET":
            print("Need to reset the partition")
    except poptorch.UnrecoverableError as e:
        print(f"Unrecoverable error: machine needs to be taken offline: {e}")
        shutdown_system()
    except poptorch.Error as e:
        print(f"Received {e.message} from component {e.type}, "
              f"location: {e.location}")
        # Or you could just print all the information at once:
        print(e)
    except Exception as e:
        print(e)
    # error_handling_end
    if model_param == TestErrorType.PoplarRecoverableFullReset:
        assert rebooted
    elif model_param == TestErrorType.PoplarUnrecoverable:
        assert shutdown
    else:
        assert not rebooted
        assert not shutdown


if __name__ == "__main__":
    # Check the example is valid
    run_example()
    for t in TestErrorType.__members__.values():
        run_example(t)
