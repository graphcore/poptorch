.. _supported_operations:

====================
Supported operations
====================


Known issues
~~~~~~~~~~~~

* If a PyTorch Geometric operation has optional arguments ``num_nodes`` or ``dim_size`` then you must pass that information in PopTorch Geometric. Operations are not able to get that information at runtime because the IPU is not able to read tensor values at runtime.
* If a PyTorch Geometric operator has an ``add_self_loop`` parameter for a  layer to add self loops, then ``add_self_loop`` must be set to False. Adding loops can cause the size of the ``edge_index`` tensor to change, and the IPU does not support dynamic shapes.