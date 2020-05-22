Introduction
------------

PopTorch is a set of extensions for PyTorch to make it operate on the Graphcore
IPU hardware.

By wrapping a `pytorch` Model class instance in a `PopTorch` wrapper, the model
can be trained on the IPU, and then saved or inspected as a normal pytorch
model.  It can also be used for making predictions using the IPU hardware.


