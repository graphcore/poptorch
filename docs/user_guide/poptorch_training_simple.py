# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# Wrap the model in a PopTorch training wrapper.
poptorch_model = poptorch.trainingModel(model,
                                        device_iterations=1,
                                        loss=loss_function,
                                        optimizer=optimizer)
for batch, loss_input in batches:
    # Performs forward pass, loss function evaluation,
    # backward pass and weight update in one go on the device.
    output, loss = poptorch_model(batch, loss_input)

    # Optimizer can be updated via setOptimizer.
    if someConditionIsMet:
        poptorch_model.setOptimizer(newOptimizer)

# Host equiv
model.train()
for batch,loss_input in batches:
    # Zero gradients
    optimizer.zero_grad()

    # Run model.
    outputs = model(batch)

    # Get the loss.
    loss = loss_function(loss_input, outputs)

    # Back probagate the gradients.
    loss.backward()

    # Update the weights.
    optimizer.step()
