# Lower to poplar

Lower to poplar provides the mechanism to lower our MLIR graph to poplar. This will expand as time goes on an will likely have some more passes. Anything which needs to access a poplar or poplibs library directly should go here. Any transform which doesn't should go into the dedicated `transforms` folder.


# Operation hooks

The core mechanism is implemented via the Interface `PoplarImplInterface` as defined in the tablegen. Any operation with this interface must define a hook member function called `lowerToPoplar`. In practice this means a new operation added to the tablegen should be given a poplar implementation here in our `ops` folder.


This enables us to automatically pick up any new operations without any additional boilerplate. This code tells the function to call `lowerToPoplar` on all operations within the function body.

```
function.walk(
    [&](PoplarImplInterface impl) { impl.lowerToPoplar(*context_); });
```

The compiler context is a useful structure used to pass all the poplar state between the ops without it needing to be leaked into the other subcomponents. It containt the poplar graph, sequence, and tensor maps.

```
void fill_::lowerToPoplar(CompilerContext &context) {

  mlir::Value input = this->input(); // Access the IR value from the op (this is a member function of the operation)
  poplar::Tensor in = context.fromSsa(input);  // Convert it to a poplar tensor. If it does not exist it will be allocated.
  float value = this->value().convertToFloat(); // Get a floating point attribute. 

  popops::fill(context.graph, in, context.seq, value); // Call the poplibs implementation.
}
```

To add a tensor to the map you call 

```
  mlir::Value output_value = this->result(); // Get the value to map the output onto.
  context.tensors.insert({output_value, poplar_out_tensor}); // Map the `poplar_out_tensor` onto the MLIR `output_value`.
```

The next operation which uses that MLIR node will then receive that tensor when it evaluates `context.fromSsa`. Values are expected to be the returns of operations. You do not need to overwrite values in the map. Perfoming an inplace operation on a tensor in the map will correctly inplace it globally.