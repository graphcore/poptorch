# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import sys
import poptorch

if not poptorch.ipuHardwareIsAvailable():
    sys.exit(0)

ipu_target_version = poptorch.ipuHardwareVersion()
filename = "training.poptorch"

# pylint: disable=unused-variable, wrong-import-position, reimported, ungrouped-imports, wrong-import-order
# precomp_start
import torch
import poptorch


class ExampleModelWithLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)
        self.loss = torch.nn.MSELoss()

    def forward(self, x, target=None):
        fc = self.fc(x)
        if self.training:
            return fc, self.loss(fc, target)
        return fc


torch.manual_seed(0)
model = ExampleModelWithLoss()

opts = poptorch.Options()
# You don't need a real IPU to compile the executable.
opts.useOfflineIpuTarget(ipu_target_version)

# Wrap the model in our PopTorch annotation wrapper.
poptorch_model = poptorch.trainingModel(model, opts)

# Some dummy inputs.
input = torch.randn(10)
target = torch.randn(10)

poptorch_model.compileAndExport(filename, input, target)
# precomp_end
poptorch_model.destroy()

# load_start
poptorch_model = poptorch.load(filename)

# That's all: your model is ready to be used.
poptorch_model(input, target)  # Run on IPU
# load_end
poptorch_model.destroy()


# load_setIpu_start
def setIpuDevice(opts):
    opts.useIpuId(1)  # always use IPU 1


poptorch_model = poptorch.load(filename, edit_opts_fn=setIpuDevice)
poptorch_model(input, target)  # Run on IPU 1
# load_setIpu_end
poptorch_model.destroy()

# precomp_no_python_start
poptorch_model.compileAndExport(filename, input, target, export_model=False)
# precomp_no_python_end
poptorch_model.destroy()

# load_exe_start
model = ExampleModelWithLoss()

opts = poptorch.Options()

# Wrap the model in our PopTorch annotation wrapper.
poptorch_model = poptorch.trainingModel(model, opts)
poptorch_model.loadExecutable(filename)

# Some dummy inputs.
input = torch.randn(10)
target = torch.randn(10)

poptorch_model(input, target)  # Run on IPU
# load_exe_end
poptorch_model.destroy()

# precomp_train_val_start
model = ExampleModelWithLoss()

opts = poptorch.Options()

# Some dummy inputs.
input = torch.randn(10)
target = torch.randn(10)

# Wrap the model in our PopTorch annotation wrapper.
training_model = poptorch.trainingModel(model, opts)
training_model.compileAndExport("training.poptorch", input, target)
model.eval()
validation_model = poptorch.inferenceModel(model, opts)
validation_model.compileAndExport("validation.poptorch", input)
# precomp_train_val_end

epochs = range(2)


def run_training(_):
    pass


def run_validation(_):
    pass


# implicit_cp_start
model = ExampleModelWithLoss()

opts = poptorch.Options()

# Wrap the model in our PopTorch annotation wrapper.
training_model = poptorch.trainingModel(model, opts)
model.eval()
validation_model = poptorch.inferenceModel(model, opts)

# Some dummy inputs.
input = torch.randn(10)
target = torch.randn(10)

# Train the model:
for epoch in epochs:
    training_model(input, target)

# Weights are implicitly copied from the training model
# to the validation model
prediction = validation_model(input)
# implicit_cp_end
training_model.destroy()
validation_model.destroy()

# load_train_val_start
training_model = poptorch.load("training.poptorch")
validation_model = poptorch.load("validation.poptorch")

for epoch in epochs:
    print("Epoch ", epoch)
    run_training(training_model)
    # Need to explicitly copy weights between the two models
    # because they're not connected anymore.
    training_model.copyWeightsToHost()
    validation_model.copyWeightsToDevice()
    run_validation(validation_model)
# load_train_val_end
training_model.destroy()
validation_model.destroy()

# load_train_val_connected_start
training_model = poptorch.load("training.poptorch")
# Create a validation python model based on the training model
validation_model = poptorch.inferenceModel(training_model)
validation_model.model.eval()
# Load the executable for that model:
validation_model.loadExecutable("validation.poptorch")

for epoch in epochs:
    print("Epoch ", epoch)
    run_training(training_model)
    # Nothing to do: training_model and validation_model are now connected
    # and PopTorch will implicitly keep the weights in sync between them.
    run_validation(validation_model)
# load_train_val_connected_end
training_model.destroy()
validation_model.destroy()
