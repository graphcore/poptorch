# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# The batchsize in the conventional sense of being the size that
# runs through an operation in the model at any given time. In
# this case BS2
model_batch_size = 2

# Run a 100 iteration loop on the IPU. Fetching a new batch each time.
device_iterations = 100

# Combine them to get the amount of data we need to give PopTorch.
combined_batch_size = model_batch_size * device_iterations

# Just tell the PyTorch data loader to grab that much at each iteration of the data loader.
training_data = torch.utils.data.DataLoader(dataset,
                                            batch_size=combined_batch_size,
                                            shuffle=True,
                                            drop_last=True)

# Wrap the mdoel in a PopTorch training wrapper.
poptorch_model = poptorch.trainingModel(model,
                                        device_iterations=device_iterations,
                                        loss=loss_function,
                                        optimizer=optimizer)

# Run over the training data with "batch_size" 200 essentially.
for batch_number, (data, labels) in enumerate(training_data):
    # Execute the device with a 100 iteration loop of batchsize 2.
    # Output will be the output at each iteration and loss will be
    # loss at each iteration (100*N, *).
    output, loss = poptorch_model(batch, loss_input)

model_batch_size = 2
device_iterations = 100

# Duplicate the model over 4 replicas.
number_of_replicas = 4

# Combine them to get the amount of data to fetch at any given iteration.
combined_batch_size = model_batch_size * number_of_replicas * device_iterations

model_batch_size = 2
device_iterations = 100
number_of_replicas = 4

# Accumulate the gradient 8 times before applying it.
gradient_accumulation = 8

# Combine them to get the amount of data to fetch at any given iteration.
combined_batch_size = model_batch_size * gradient_accumulation * number_of_replicas * device_iterations
