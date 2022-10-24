// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

class Poptorch_IdentityOp<string mnemonic, list<Trait> traits = []> : Poptorch_AbstractOp<mnemonic, traits # [ViewOp]> {
  let arguments = (ins Poptorch_tensor:$input);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$input), [{
    $_state.addOperands({input});
    $_state.addTypes(input.getType());
  }]>];

  let hasCanonicalizeMethod = 1;
}

def Poptorch_alias: Poptorch_IdentityOp<"alias">;

def Poptorch_detach: Poptorch_IdentityOp<"detach"> {
  let results = (outs Poptorch_tensor_no_grad:$result);
}

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> (Tensor(a))
def Poptorch_as_strided: Poptorch_AbstractOp<"as_strided", [ViewOp]> {
  let arguments = (ins Poptorch_tensor:$input,
                       I64ArrayAttr:$size,
                       I64ArrayAttr:$strides,
                       OptionalAttr<I64Attr>:$storage_offset);

  let results = (outs Poptorch_tensor:$result);


  let builders = [OpBuilder<(ins "mlir::Value":$val1,
                                  "const std::vector<std::int64_t>&":$size,
                                  "const std::vector<std::int64_t>&":$strides,
                                  "std::optional<std::int64_t>":$storage_offset),[{
    ERROR_ON_MSG(storage_offset.value_or(0), "as_strided: Storage offset "
                                             "not supported");
    $_state.addOperands({val1});
    $_state.addAttribute("size", $_builder.getI64ArrayAttr(size));
    $_state.addAttribute("strides", $_builder.getI64ArrayAttr(strides));

    ERROR_ON_MSG(
        strides.size() != size.size(),
        "The new strides have a different number of dimensions to the new shape (" +
        std::to_string(strides.size()) + " != " + std::to_string(size.size()) + ")");

    // Check that we don't actually change the stride here
    // TODO(T59600) consider moving this check to the MLIR level
    std::vector<std::size_t> expected_strides(size.size(), 1);
    std::partial_sum(size.rbegin(), size.rend() - 1,
                    expected_strides.rbegin() + 1, std::multiplies<>{});

    const auto new_num_elements = std::accumulate(
        size.begin(), size.end(), 1.0, std::multiplies<>{});

    const auto first_diff = std::mismatch(strides.begin(), strides.end(),
                                          expected_strides.begin());

    mlir::RankedTensorType tensor = val1.getType().cast<mlir::RankedTensorType>();

    // Note: this will have false positives/negatives if it follows a pytorch
    // operation that would normally change the stride like expand, reshape or
    // dimshuffle
    ERROR_ON_MSG(
        tensor.getNumElements() != new_num_elements ||
            first_diff.first != strides.end(),
        "Poptorch does not support arbitrary manipulations of the shape and "
        "stride of a tensor. Prefer other view functions like "
        "torch.tensor.expand() over setting the shape and stride of a view "
        "manually.");

    $_state.addTypes(mlir::RankedTensorType::get(size, tensor.getElementType()));
  }]>];

  let hasCanonicalizeMethod = 1;
}

// aten::transpose.int(Tensor(a) self, int dim0, int dim1)
def Poptorch_transpose: Poptorch_AbstractOp<"transpose", [ViewOp]> {
  let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim0,  I64Attr:$dim1);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$value, "std::int64_t":$dim0, "std::int64_t":$dim1),[{
    $_state.addOperands({value});

    mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
    llvm::SmallVector<std::int64_t, 4> shape(t1.getShape().begin(),
                                             t1.getShape().end());

    dim0 = convertToPositiveDim(dim0, shape.size());
    $_state.addAttribute("dim0", $_builder.getI64IntegerAttr(dim0));

    dim1 = convertToPositiveDim(dim1, shape.size());
    $_state.addAttribute("dim1", $_builder.getI64IntegerAttr(dim1));

    // Transpose the type.
    if (dim0 < shape.size() && dim1 < shape.size()) {
        std::swap(shape[dim0], shape[dim1]);
    }

    $_state.addTypes(mlir::RankedTensorType::get(shape, t1.getElementType()));
  }]>];

  let hasCanonicalizeMethod = 1;
}

def Poptorch_permuteOutplace : Poptorch_InternalOp<"permuteOutplace"> {
  let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$dims);
  let results = (outs Poptorch_tensor:$view);
}

def Poptorch_permuteInverse : Poptorch_AbstractInternalOp<"permuteInverse"> {
  let arguments = (ins Poptorch_tensor:$view, Poptorch_tensor:$input, I64ArrayAttr:$dims);
  let results = (outs Poptorch_tensor:$inverse);

  let hasCanonicalizeMethod = 1;
}

def Poptorch_permute : Poptorch_ViewOp<"permute", "permuteOutplace", "permuteInverse"> {
  let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$dims);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$dims),[{
    $_state.addOperands(self);
    $_state.addAttribute("dims", $_builder.getI64ArrayAttr(dims));

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto input_shape = tensor.getShape();

    ERROR_ON_MSG(dims.size() != input_shape.size(),
                 "Must have the same number of dims (" + std::to_string(dims.size()) +
                 ") as the input dimension (" + std::to_string(input_shape.size()) + ")");

    // Check that dims is a perumation
    for (std::size_t i = 0; i < dims.size(); ++i) {
        ERROR_ON_MSG(std::find(dims.begin(), dims.end(), i) == dims.end() &&
                     std::find(dims.begin(), dims.end(), i - dims.size()) == dims.end(),
                     "dims must be a perumation. Missing dim " << i);
    }

    llvm::SmallVector<std::int64_t, 4> shape;
    shape.reserve(dims.size());

    for (auto dim : dims) {
        shape.push_back(input_shape[dim >= 0 ? dim : dim + dims.size()]);
    }

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

def Poptorch_viewOutplace : Poptorch_InternalOp<"viewOutplace"> {
  let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$shape);
  let results = (outs Poptorch_tensor:$result);

  let hasCanonicalizeMethod = 1;
}

def Poptorch_viewInverse : Poptorch_AbstractInternalOp<"viewInverse"> {
  let arguments = (ins Poptorch_tensor:$view, Poptorch_tensor:$input, I64ArrayAttr:$shape);
  let results = (outs Poptorch_tensor:$result);

  let hasCanonicalizeMethod = 1;
}

def Poptorch_view : Poptorch_ViewOp<"view", "viewOutplace", "viewInverse"> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$shape);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$value, "const std::vector<std::int64_t>&":$shape),[{
        $_state.addOperands({value});

        // At most one element can be -1, for which the shape will need to be
        // resolved based on the number of elements.
        const int64_t no_dim = -1;
        int64_t inferred_dim = no_dim;

        // Store the product of the shape (excluding the -1 dimension)
        uint64_t shape_product = 1;

        for(size_t dim_idx = 0; dim_idx < shape.size(); dim_idx++) {
            if(shape[dim_idx] == -1) {
                if(inferred_dim != no_dim) {
                    // Match PyTorch's error message
                    ERROR("only one dimension can be inferred");
                }
                inferred_dim = dim_idx;
            } else if(shape[dim_idx] <= 0) {
                // This should only occur as an internal error.
                ERROR("invalid shape dimension, " << shape[dim_idx]
                      << ", in Poptorch_view");
            } else {
                shape_product *= shape[dim_idx];
            }
        }

        // Obtain the tensor type and clone the output shape in case we modify
        // it when resolving the "-1".
        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        auto numel_in = t1.getNumElements();

        // Clone the const shape in case we need to resolve a -1 dim
        auto shape_out = shape;

        // Resolve the -1 dim if required
        if (inferred_dim != no_dim) {
            if(numel_in % shape_product == 0) {
                shape_out[inferred_dim] = numel_in/shape_product;
                shape_product = numel_in;
            }
        }

        // At this stage, shape_product should be numel_in, or the shape is
        // invalid (with or without a -1 dim)
        if (shape_product != numel_in) {
            std::ostringstream err;
            err << "shape'[";

            for(size_t dim_idx = 0; dim_idx < shape.size(); dim_idx++) {
                err << shape[dim_idx];
                if(dim_idx + 1 != shape.size()) {
                    err << ", ";
                }
            }

            err << "]' is invalid for input of size ";
            err << numel_in;
            throw std::runtime_error(err.str());
        }

        // Otherwise, the view is succesful
        // Ensure the attribute is the resolved shape (without -1)
        $_state.addAttribute("shape", $_builder.getI64ArrayAttr(shape_out));

        mlir::Type e1 = t1.getElementType();
        $_state.addTypes(mlir::RankedTensorType::get(shape_out, e1));
     }]>
    ];
}


def Poptorch_expandOutplace: Poptorch_InternalOp<"expandOutplace"> {
  let arguments = (ins Poptorch_tensor:$input,
                       I64ArrayAttr:$shape,
                       DefaultValuedAttr<BoolAttr,"false">:$implicit);
  let results = (outs Poptorch_tensor:$result);
}

def Poptorch_expandInverse: Poptorch_AbstractInternalOp<"expandInverse", [NotImplementedOp]> {
  let arguments = (ins Poptorch_tensor:$view,
                       Poptorch_tensor:$input,
                       I64ArrayAttr:$shape,
                       DefaultValuedAttr<BoolAttr,"false">:$implicit);
  let results = (outs Poptorch_tensor:$result);
}


// Expand is like reshape but can broadcast unity dimensions and sets -1
// dimensions to the same as the input. New dimensions are added at the front
def Poptorch_expand: Poptorch_ViewOp<"expand", "expandOutplace", "expandInverse"> {
  let arguments = (ins Poptorch_tensor:$input,
                       I64ArrayAttr:$shape,
                       DefaultValuedAttr<BoolAttr,"false">:$implicit);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$value,
                                 "const std::vector<std::int64_t>&":$shape,
                                 "bool":$implicit),[{
    ERROR_ON_MSG(implicit,"expand: implicit=True not supported");
    $_state.addOperands({value});

    const auto tensor = asTensor(value);

    // Resolve -1s in the desired shape into actual lengths.
    const auto input_shape = tensor.getShape();
    std::vector<std::int64_t> desired_shape = shape;

    ERROR_ON_MSG(desired_shape.size() < input_shape.size(),
                 "The desired shape passed to expand should have at least "
                 "as many dimensions as the input tensor (required at "
                 "least " << input_shape.size() << ", got "
                 << desired_shape.size() << ")");

    for (size_t i = 0; i < input_shape.size(); i++) {
      // If you give more dimensions in the desired shape than there are in
      // the input tensor, they'll get *pre*pended -- so to turn the -1s
      // into lengths from the input, work backwards.
      const auto input_idx = input_shape.size() - (i + 1);
      const auto input_len = input_shape[input_idx];
      const auto desired_idx = desired_shape.size() - (i + 1);
      const auto desired_len = desired_shape[desired_idx];

      if (desired_len == -1) {
        desired_shape[desired_idx] = input_len;
      } else if (desired_len != input_len && input_len != 1) {
        ERROR("Can only expand dimensions of size 1; however, trying "
              "to expand dimension " << input_idx << " of size " <<
              input_len << " to " << desired_len);
      }
    }

    const auto last = std::next(desired_shape.begin(),
                                desired_shape.size() - input_shape.size());
    ERROR_ON_MSG(std::find(desired_shape.begin(), last, -1) != last,
                 "torch.expand: tried to set an added dimension's length "
                 "to -1");

    $_state.addAttribute("shape", $_builder.getI64ArrayAttr(desired_shape));

    $_state.addTypes(mlir::RankedTensorType::get(desired_shape, tensor.getElementType()));
   }]>
  ];
}


class Poptorch_SqueezeOp<string name>: Poptorch_AbstractOp<name, [ViewOp]> {
  let extraClassDeclaration = [{
      static int64_t processDim(int64_t dim, mlir::Value v) {
        auto shape = getShape(v);
        return dim < 0 ? dim + shape.size() : dim;
      }

      static mlir::Type inferType(mlir::Value v, const std::vector<int64_t> &dims) {
          auto shape = getShape(v);
          std::vector<int64_t> out_shape;

          for (auto i = 0u; i < shape.size(); i++) {
            auto it = std::find(dims.begin(), dims.end(), i);
            if (it == dims.end() || shape[i] != 1) {
              out_shape.push_back(shape[i]);
            }
          }

          return mlir::RankedTensorType::get(out_shape, getElementType(v));
      }
  }];

  let hasCanonicalizeMethod = 1;
}

def Poptorch_squeeze: Poptorch_SqueezeOp<"squeeze"> {
  let arguments = (ins Poptorch_tensor:$input);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$value),[{
      $_state.addOperands({value});

      auto dims = getShape(value);
      std::iota(dims.begin(), dims.end(), 0);
      $_state.addTypes(inferType(value, dims));
    }]>
  ];
}

// squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
def Poptorch_squeeze_dim: Poptorch_SqueezeOp<"squeeze_dim"> {
  let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$value, "std::int64_t":$dim),[{
    $_state.addOperands({value});
    dim = processDim(dim, value);
    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
    $_state.addTypes(inferType(value, {dim}));
    }]>
  ];
}

// unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
def Poptorch_unsqueeze : Poptorch_AbstractOp<"unsqueeze", [ViewOp]> {
  let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$value, "std::int64_t":$dim),[{
    $_state.addOperands({value});
    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

    mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
    mlir::Type e1 = t1.getElementType();
    std::vector<int64_t> shape = t1.getShape();
    // "Negative dim will correspond to unsqueeze() applied at
    // dim = dim + input.dim() + 1."
    // Source:
    // https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
    if (dim < 0) {
        dim += shape.size() + 1;
    }
    shape.insert(shape.begin() + dim, 1);
    $_state.addTypes(mlir::RankedTensorType::get(shape, e1));
  }]>];

  let hasCanonicalizeMethod = 1;
}

// gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor(a!)
def Poptorch_gather: Poptorch_NotImplementedOp<"gather"> {
  let arguments = (ins Poptorch_tensor:$self, I64Attr:$dim, Poptorch_tensor:$index, DefaultValuedAttr<BoolAttr,"false">:$sparse_grad);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self, "std::int64_t":$dim, "mlir::Value":$index, "bool":$sparse_grad),[{
    ERROR_ON_MSG(sparse_grad, "gather: sparse_grad=True not supported");

    $_state.addOperands({self, index});
    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

    mlir::RankedTensorType t1 = self.getType().cast<mlir::RankedTensorType>();
    auto index_shape = index.getType().cast<mlir::RankedTensorType>().getShape();
    mlir::Type e1 = t1.getElementType();
    auto shape = t1.getShape();
    if (dim < 0) {
        dim += shape.size();
    }
    // From Torch's documentation:
    // input and index must have the same number of dimensions.
    // It is also required that index.size(d) <= input.size(d) for all
    // dimensions d != dim.
    // out will have the same shape as index.
    // Note that input and index do not broadcast against each other.
    ERROR_ON_MSG(shape.size() != index_shape.size(),"self and index must "
        "have the same number of dimensions");
    for(auto d = 0u; d < shape.size(); d++){
      ERROR_ON_MSG(d != dim && index_shape[d] > shape[d], "index.size("
          <<d<<") > input.size("<<d<<") ("<<index_shape[d]<<" > "<<shape[d]<<")");
    }
    $_state.addTypes(mlir::RankedTensorType::get(index_shape, e1));
   }]>
  ];
}

def Poptorch_sliceOutplace : Poptorch_InternalOp<"sliceOutplace"> {
  let arguments = (ins Poptorch_tensor:$input,
                       I64Attr:$dim,
                       OptionalAttr<I64Attr>:$start,
                       OptionalAttr<I64Attr>:$end,
                       I64Attr:$step);

  let results = (outs Poptorch_tensor:$result);
}

def Poptorch_sliceInverse : Poptorch_InternalOp<"sliceInverse"> {
  let arguments = (ins Poptorch_tensor:$view,
                       Poptorch_tensor:$input,
                       I64Attr:$dim,
                       OptionalAttr<I64Attr>:$start,
                       OptionalAttr<I64Attr>:$end,
                       I64Attr:$step);

  let results = (outs Poptorch_tensor:$result);
}

def Poptorch_slice_Tensor : Poptorch_ViewOp<"slice_Tensor", "sliceOutplace", "sliceInverse"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$dim,
                       OptionalAttr<I64Attr>:$start,
                       OptionalAttr<I64Attr>:$end,
                       I64Attr:$step);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::int64_t":$dim,
                                 "std::optional<std::int64_t>":$start,
                                 "std::optional<std::int64_t>":$end,
                                 "std::int64_t":$step),[{
    // The logic here is taken from pytorch in the function aten/src/ATen/native/TensorShape.cpp:slice()
    $_state.addOperands({self});

    mlir::RankedTensorType input_tensor = self.getType().cast<mlir::RankedTensorType>();
    llvm::SmallVector<std::int64_t, 4> out_shape {input_tensor.getShape().begin(),
                                                  input_tensor.getShape().end()};

    ERROR_ON_MSG(out_shape.size() == 0, "slice() cannot be applied to a 0-dim tensor.");

    ERROR_ON_MSG(step <= 0, "slice step must be positive");
    // TODO: there is a TODO in the pytorch code to support negative step slices
    //ERROR_ON_MSG(step == 0, "slice step cannot be zero");

    dim = convertToPositiveDim(dim, out_shape.size());

    std::int64_t actual_start = start.value_or(0);
    std::int64_t actual_end = end.value_or(out_shape[dim]);

    if (actual_start < 0) {
        actual_start += out_shape[dim];
    }
    if (actual_end < 0) {
        actual_end += out_shape[dim];
    }

    actual_start = std::clamp(actual_start, int64_t{0}, out_shape[dim]);
    actual_end = std::clamp(actual_end, actual_start, out_shape[dim]);

    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
    $_state.addAttribute("start", $_builder.getI64IntegerAttr(actual_start));
    $_state.addAttribute("end", $_builder.getI64IntegerAttr(actual_end));
    $_state.addAttribute("step", $_builder.getI64IntegerAttr(step));

    // Get the number of elements for the dim rounding up
    out_shape[dim] = (actual_end - actual_start + (std::abs(step) - 1)) / std::abs(step);

    $_state.addTypes(mlir::RankedTensorType::get(out_shape, input_tensor.getElementType()));
   }]>
  ];
}

// select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
def Poptorch_select : Poptorch_AbstractOp<"select", [ViewOp]> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$dim,
                       I64Attr:$idx);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::int64_t":$dim,
                                 "std::int64_t":$idx),[{

    // Operands
    $_state.addOperands({self});

    // Attributes
    mlir::RankedTensorType input_tensor = self.getType().cast<mlir::RankedTensorType>();
    llvm::SmallVector<std::int64_t, 4> out_shape {input_tensor.getShape().begin(),
                                                  input_tensor.getShape().end()};

    dim = convertToPositiveDim(dim, out_shape.size());
    
    if (idx < 0) {
        idx += out_shape[dim];
    }

    ERROR_ON_MSG(!(0 <= idx && idx < out_shape[dim]),
                 "The index (" << idx << ") must be within the range of the "
                 "requested dimension (" << dim << ")");

    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
    $_state.addAttribute("idx", $_builder.getI64IntegerAttr(idx));

    // Returns
    out_shape.erase(out_shape.begin() + dim);

    auto out_type = mlir::RankedTensorType::get(out_shape, input_tensor.getElementType());

    $_state.addTypes({out_type});
   }]>
  ];

  let hasCanonicalizeMethod = 1;
}

// repeat(Tensor self, int[] repeats) -> Tensor
def Poptorch_repeat : Poptorch_Op<"repeat"> {
  let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$repeats);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$value,
                                 "const std::vector<std::int64_t>&":$repeats),[{
    $_state.addOperands({value});
    $_state.addAttribute("repeats", $_builder.getI64ArrayAttr(repeats));

    mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
    mlir::Type e1 = t1.getElementType();
    std::vector<int64_t> shape = t1.getShape();

    ERROR_ON_MSG(repeats.size() < shape.size(),
                 "Number of dimensions of repeat dims can not be smaller "
                 "than number of dimensions of tensor");

    std::vector<int64_t> new_shape(repeats.size());

    int64_t num_new_dimensions = repeats.size()-shape.size();
    for (int64_t i=0; i < new_shape.size(); i++) {
        int64_t ishape = 1;
        if (i >= num_new_dimensions) {
            ishape = shape[i-num_new_dimensions];
        }
       new_shape[i] = ishape * repeats[i];
    }
    $_state.addTypes(mlir::RankedTensorType::get(new_shape, e1));
  }]>];
}

class Poptorch_NotImplemented_padding<string name> : Poptorch_NotImplementedOp<name, []> {
  let arguments = (ins Poptorch_tensor:$self, I64ArrayAttr:$pad);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                  "const std::vector<std::int64_t>&":$pad),[{
    $_state.addOperands({self});
    $_state.addAttribute("pad", $_builder.getI64ArrayAttr(pad));

    $_state.addTypes(inferShape(self, pad));
  }]>];

  let extraClassDeclaration = [{
    using CheckPadding = std::function<void(std::int64_t shape, std::int64_t lpad, std::int64_t rpad, std::size_t dim)>;

    static void defaultCheckPadding(std::int64_t shape, std::int64_t lpad, std::int64_t rpad, std::size_t dim) {
      ERROR_ON_MSG(shape + std::min(lpad, 0l) + std::min(rpad, 0l) < 0,
                  "You cannot pad negatively by more than the size of dimension " +
                  std::to_string(dim));
    }

    static mlir::Type inferShape(mlir::Value self, const std::vector<std::int64_t>& pad, const CheckPadding& checkPadding = &defaultCheckPadding) {
      ERROR_ON_MSG(pad.size() % 2 != 0,
                  "Length of pad (" + std::to_string(pad.size()) +
                  ") must be even");

      mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
      const auto input_shape = tensor.getShape();

      ERROR_ON_MSG(pad.size() > 2 * input_shape.size(),
                  "Length of pad should be no more that twice the number of dimension "
                  "of the input. Pad has " + std::to_string(pad.size()) +
                  " dimensions while the input has " +
                  std::to_string(input_shape.size()) + " dimensions");

      llvm::SmallVector<std::int64_t, 4> shape{input_shape.begin(),
                                              input_shape.end()};

      for (std::size_t i = 0; i < pad.size() / 2; ++i) {
        const auto dim = shape.size() - 1 - i;
        const auto lpad = pad[2 * i];
        const auto rpad = pad[2 * i + 1];

        checkPadding(shape[dim], lpad, rpad, dim);

        shape[dim] += lpad + rpad;
      }

      return mlir::RankedTensorType::get(shape, tensor.getElementType());
    }
  }];
}

def Poptorch_replication_pad1d_out : Poptorch_NotImplemented_padding<"replication_pad1d_out"> {}
def Poptorch_replication_pad2d_out : Poptorch_NotImplemented_padding<"replication_pad2d_out"> {}
def Poptorch_replication_pad3d_out : Poptorch_NotImplemented_padding<"replication_pad3d_out"> {}

def Poptorch_constant_pad_nd : Poptorch_NotImplemented_padding<"constant_pad_nd"> {
  let arguments = (ins Poptorch_tensor:$self, I64ArrayAttr:$pad, F32Attr:$scalar);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$pad,
                                 "float":$scalar),[{
    $_state.addOperands({self});
    $_state.addAttribute("pad", $_builder.getI64ArrayAttr(pad));
    $_state.addAttribute("scalar", $_builder.getF32FloatAttr(scalar));

    $_state.addTypes(inferShape(self, pad));
  }]>];
}

class Poptorch_NotImplemented_reflection_padding<string name> : Poptorch_NotImplemented_padding<name> {
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                  "const std::vector<std::int64_t>&":$pad),[{
    $_state.addOperands({self});
    $_state.addAttribute("pad", $_builder.getI64ArrayAttr(pad));

    constexpr auto checkReflectionPadding = [](std::int64_t shape, std::int64_t lpad, std::int64_t rpad, std::size_t dim) {
      ERROR_ON_MSG(lpad >= shape || rpad >= shape,
                   "Note that the padding (" + std::to_string(lpad) + ", " +
                   std::to_string(rpad) +") in dimension " + std::to_string(dim) +
                   " must be strictly less that the dimension size (" +
                   std::to_string(shape) + ")");

      // NOTE: this isn't as strict as constant pad
      ERROR_ON_MSG(shape + lpad + rpad < 0,
                   "You cannot pad negatively by more than the size of dimension " +
                   std::to_string(dim));
    };


    $_state.addTypes(inferShape(self, pad, checkReflectionPadding));
  }]>];
}

def Poptorch_reflection_pad1d_out : Poptorch_NotImplemented_reflection_padding<"reflection_pad1d_out"> {}
def Poptorch_reflection_pad2d : Poptorch_NotImplemented_reflection_padding<"reflection_pad2d"> {}

// See aten/src/ATen/native/TensorAdvancedIndexing.cpp for more details.
def Poptorch_index_tensor: Poptorch_NotImplementedOp<"index_tensor", []> {
  let arguments = (ins Poptorch_tensor:$self, Poptorch_tensorlist:$indices);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                               "mlir::ValueRange":$indices),[{

    auto is_none = [&] (mlir::Value v) {
      return getElementType(v) == $_builder.getNoneType();
    };

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto input_shape = tensor.getShape();

    ERROR_ON_MSG(indices.size() > input_shape.size(),
                 "Cannot index outside the tensor (dims " +
                 std::to_string(input_shape.size()) + ") with dim (" +
                 std::to_string(indices.size()) + ")");
    
    auto broadcasted_index_shape = std::accumulate(indices.begin(), indices.end(),
                                                   std::vector<std::int64_t>{},
                                                   [&] (const auto& acc, const auto &index) {
      if (is_none(index)) {
        return acc;
      }
      return broadcast(acc, getShape(index));
    });
    bool none_after_index = false;
    bool index_after_none = false;
    for (auto i = 0u; i < indices.size() - 1; i++) {
      if (!is_none(indices[i]) && is_none(indices[i + 1])) {
        none_after_index = true;
      }
      if (is_none(indices[i]) && !is_none(indices[i + 1])) {
        index_after_none = true;
      }
    }
    bool indices_are_separated = none_after_index && index_after_none;

    std::vector<std::int64_t> output_shape;
    bool indices_written = false;
    if (indices_are_separated) {
      std::copy(broadcasted_index_shape.begin(), broadcasted_index_shape.end(),
                std::back_inserter(output_shape));
      indices_written = true;
    }

    for (auto i = 0u; i < input_shape.size(); i++) {
      if(i >= indices.size() || is_none(indices[i])){
        output_shape.push_back(input_shape[i]);
        continue;
      }
      mlir::RankedTensorType index_tensor = indices[i].getType().cast<mlir::RankedTensorType>();
      ERROR_ON_MSG(!index_tensor.getElementType().isIntOrIndex(),
                   "index must have integer type");
      
      if (!indices_written) {
        std::copy(broadcasted_index_shape.begin(), broadcasted_index_shape.end(),
                  std::back_inserter(output_shape));
        indices_written = true;
      }
    }

    $_state.addOperands(self);
    $_state.addOperands(indices);
    $_state.addTypes(mlir::RankedTensorType::get(output_shape, tensor.getElementType()));
  }]>];
}

// index_put(Tensor self, Tensor?[] indices, Tensor values,
//           bool accumulate=False) -> Tensor
def Poptorch_index_put: Poptorch_NotImplementedOp<"index_put", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       Poptorch_tensorlist:$indices,
                       Poptorch_tensor:$values,
                       BoolAttr:$accumulate);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "mlir::ValueRange":$indices,
                                 "mlir::Value":$values,
                                 "bool":$accumulate),[{
    ERROR_ON_MSG(accumulate, "index_put is not supported with accumulate=true.");

    $_state.addOperands(self);
    $_state.addOperands(indices);
    $_state.addOperands(values);
    $_state.addTypes(self.getType());
  }]>];
}

def Poptorch_index_fill_Scalar: Poptorch_NotImplementedOp<"index_fill_Scalar", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$dim,
                       Poptorch_tensor:$index,
                       F32Attr:$value);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::int64_t":$dim,
                                 "mlir::Value":$index,
                                 "float":$value),[{
    $_state.addOperands({self, index});
    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
    $_state.addAttribute("value", $_builder.getF32FloatAttr(value));
    $_state.addTypes(self.getType());
  }]>];
}

def Poptorch_index_fill_Tensor: Poptorch_NotImplementedOp<"index_fill_Tensor", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$dim,
                       Poptorch_tensor:$index,
                       Poptorch_tensor:$value);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::int64_t":$dim,
                                 "mlir::Value":$index,
                                 "mlir::Value":$value),[{
    $_state.addOperands({self, index, value});
    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
    $_state.addTypes(self.getType());
  }]>];
}

def Poptorch_index_select : Poptorch_Op<"index_select", []> {
  let arguments = (ins Poptorch_tensor:$self, I64Attr:$dim, Poptorch_tensor:$index);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::int64_t":$dim,
                                 "mlir::Value":$index), [{
    $_state.addOperands({self, index});

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto input_shape = tensor.getShape();
    
    dim = convertToPositiveDim(dim, input_shape.size());
    $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

    mlir::RankedTensorType index_tensor = index.getType().cast<mlir::RankedTensorType>();
    const auto index_shape = index_tensor.getShape();

    ERROR_ON_MSG(index_shape.size() != 1, "index must be a 1-D tensor");
    ERROR_ON_MSG(!index_tensor.getElementType().isIntOrIndex(),
                 "index must have integer type");

    llvm::SmallVector<std::int64_t, 4> shape{input_shape.begin(),
                                             input_shape.end()};

    shape[dim] = index_shape.front();

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

class Poptorch_upsample<string name> : Poptorch_NotImplementedOp<name, []> {
  let results = (outs Poptorch_tensor:$result);
  let extraClassDeclaration = [{
    static auto computeShape(const std::vector<std::int64_t>& output_size, mlir::RankedTensorType tensor) {
      const auto input_shape = tensor.getShape();
      ERROR_ON_MSG(output_size.size() > input_shape.size(),
                  "The number of dimensions of the input (" + std::to_string(input_shape.size()) +
                  ") must be more than the number of dimensions in the output (" +
                  std::to_string(output_size.size()) + ")");

      llvm::SmallVector<std::int64_t, 4> shape(input_shape.begin(), input_shape.end() - output_size.size());
      shape.insert(shape.end(), output_size.begin(), output_size.end());

      return shape;
    }
  }];
}

def Poptorch_upsample_linear1d_out : Poptorch_upsample<"upsample_linear1d_out"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$output_size,
                       BoolAttr:$align_corners,
                       OptionalAttr<F64Attr>:$scales);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$output_size,
                                 "bool":$align_corners,
                                 "std::optional<double>":$scales),[{
    $_state.addOperands(self);
    $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
    $_state.addAttribute("align_corners", $_builder.getBoolAttr(align_corners));
    if (scales) {
        $_state.addAttribute("scales", $_builder.getF64FloatAttr(*scales));
    }

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto shape = computeShape(output_size, tensor);

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

def Poptorch_upsample_nearest1d_out : Poptorch_upsample<"upsample_nearest1d_out"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$output_size,
                       OptionalAttr<F64Attr>:$scales);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$output_size,
                                 "std::optional<double>":$scales),[{
    $_state.addOperands(self);
    $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
    if (scales) {
      $_state.addAttribute("scales", $_builder.getF64FloatAttr(*scales));
    }

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto shape = computeShape(output_size, tensor);

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

class Poptorch_upsample2d_out<string name> : Poptorch_upsample<name> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$output_size,
                       BoolAttr:$align_corners,
                       OptionalAttr<F64Attr>:$scales_h,
                       OptionalAttr<F64Attr>:$scales_w);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$output_size,
                                 "bool":$align_corners,
                                 "std::optional<double>":$scales_h,
                                 "std::optional<double>":$scales_w),[{
    $_state.addOperands(self);
    $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
    $_state.addAttribute("align_corners", $_builder.getBoolAttr(align_corners));
    if (scales_h) {
      $_state.addAttribute("scales_h", $_builder.getF64FloatAttr(*scales_h));
    }
    if (scales_w) {
      $_state.addAttribute("scales_w", $_builder.getF64FloatAttr(*scales_w));
    }

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto shape = computeShape(output_size, tensor);

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

def Poptorch_upsample_bilinear2d_out : Poptorch_upsample2d_out<"upsample_bilinear2d_out"> {}
def Poptorch_upsample_bicubic2d_out : Poptorch_upsample2d_out<"upsample_bicubic2d_out"> {}

def Poptorch_upsample_nearest2d_out : Poptorch_upsample<"upsample_nearest2d_out"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$output_size,
                       OptionalAttr<F64Attr>:$scales_h,
                       OptionalAttr<F64Attr>:$scales_w);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$output_size,
                                 "std::optional<double>":$scales_h,
                                 "std::optional<double>":$scales_w),[{
    $_state.addOperands(self);
    $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
    if (scales_h) {
      $_state.addAttribute("scales_h", $_builder.getF64FloatAttr(*scales_h));
    }
    if (scales_w) {
      $_state.addAttribute("scales_w", $_builder.getF64FloatAttr(*scales_w));
    }

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto shape = computeShape(output_size, tensor);

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}


class Poptorch_upsample3d_out<string name> : Poptorch_upsample<name> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$output_size,
                       BoolAttr:$align_corners,
                       OptionalAttr<F64Attr>:$scales_d,
                       OptionalAttr<F64Attr>:$scales_h,
                       OptionalAttr<F64Attr>:$scales_w);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$output_size,
                                 "bool":$align_corners,
                                 "std::optional<double>":$scales_d,
                                 "std::optional<double>":$scales_h,
                                 "std::optional<double>":$scales_w),[{
    $_state.addOperands(self);
    $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
    $_state.addAttribute("align_corners", $_builder.getBoolAttr(align_corners));
    if (scales_d) {
      $_state.addAttribute("scales_d", $_builder.getF64FloatAttr(*scales_d));
    }
    if (scales_h) {
      $_state.addAttribute("scales_h", $_builder.getF64FloatAttr(*scales_h));
    }
    if (scales_w) {
      $_state.addAttribute("scales_w", $_builder.getF64FloatAttr(*scales_w));
    }

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto shape = computeShape(output_size, tensor);

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

def Poptorch_upsample_trilinear3d_out : Poptorch_upsample3d_out<"upsample_trilinear3d_out"> {}

def Poptorch_upsample_nearest3d_out : Poptorch_upsample<"upsample_nearest3d_out"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$output_size,
                       OptionalAttr<F64Attr>:$scales_d,
                       OptionalAttr<F64Attr>:$scales_h,
                       OptionalAttr<F64Attr>:$scales_w);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::vector<std::int64_t>&":$output_size,
                                 "std::optional<double>":$scales_d,
                                 "std::optional<double>":$scales_h,
                                 "std::optional<double>":$scales_w),[{
    $_state.addOperands(self);
    $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(output_size));
    if (scales_d) {
        $_state.addAttribute("scales_d", $_builder.getF64FloatAttr(*scales_d));
    }
    if (scales_h) {
        $_state.addAttribute("scales_h", $_builder.getF64FloatAttr(*scales_h));
    }
    if (scales_w) {
        $_state.addAttribute("scales_w", $_builder.getF64FloatAttr(*scales_w));
    }

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto shape = computeShape(output_size, tensor);

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

// upsample_nearest3d.vec isn't being forwarded to 'upsample_nearest3d.out' so we need to do our own computation of the output_size in a builder
def Poptorch_upsample_nearest3d_vec : Poptorch_upsample<"upsample_nearest3d_vec"> {
  let arguments = (ins Poptorch_tensor:$self,
                       OptionalAttr<I64ArrayAttr>:$output_size,
                       OptionalAttr<F32ArrayAttr>:$scales);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "const std::optional<std::vector<std::int64_t>>&":$output_size,
                                 "const std::optional<std::vector<float>>&":$scale_factors),[{
    $_state.addOperands(self);

    mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
    const auto input_shape = tensor.getShape();

    ERROR_ON_MSG(!scale_factors && !output_size,
                     "Must specify exactly one of output_size and scale_factors");
    std::vector<int64_t> actual_output_size;
    if (output_size) {
      ERROR_ON_MSG(scale_factors,
                      "Must specify exactly one of output_size and scale_factors");
      actual_output_size = *output_size;
      $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(actual_output_size));
    }
    else if (scale_factors) {
      std::transform(scale_factors->begin(), scale_factors->end(),
                     input_shape.end() - scale_factors->size(),
                     std::back_inserter(actual_output_size),
                     [](double sf, std::int64_t shape) {
                         return static_cast<int64_t>(static_cast<double>(shape) * sf);
                     });

      $_state.addAttribute("output_size", $_builder.getI64ArrayAttr(actual_output_size));
      $_state.addAttribute("scales", $_builder.getF32ArrayAttr(*scale_factors));
    }

    const auto shape = computeShape(actual_output_size, tensor);

    $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
  }]>];
}

def Poptorch_im2col : Poptorch_upsample<"im2col"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$kernel_size,
                       I64ArrayAttr:$dilation,
                       I64ArrayAttr:$padding,
                       I64ArrayAttr:$stride);
  let results = (outs Poptorch_tensor:$result);
  let builders = [
    OpBuilder<(ins "mlir::Value":$self,
                   "const std::vector<std::int64_t>&":$kernel_size,
                   "const std::vector<std::int64_t>&":$dilation,
                   "const std::vector<std::int64_t>&":$padding,
                   "const std::vector<std::int64_t>&":$stride), [{
      $_state.addOperands(self);

      $_state.addAttribute("kernel_size",
                           $_builder.getI64ArrayAttr(kernel_size));
      $_state.addAttribute("dilation",
                           $_builder.getI64ArrayAttr(dilation));
      $_state.addAttribute("padding",
                           $_builder.getI64ArrayAttr(padding));
      $_state.addAttribute("stride", $_builder.getI64ArrayAttr(stride));

      auto input_shape = getShape(self);
      ERROR_ON_MSG(input_shape.size() != 3 && input_shape.size() != 4,
                   "Input must have 3 or 4 dimensions.");
      if (input_shape.size() == 3) {
        input_shape.insert(input_shape.begin(), 1);
      }

      ERROR_ON_MSG(kernel_size.size() != 2,
                   "kernel_size must have 2 elements.");
      ERROR_ON_MSG(dilation.size() != 2,
                   "dilation must have 2 elements.");
      ERROR_ON_MSG(padding.size() != 2, "padding must have 2 elements.");
      ERROR_ON_MSG(stride.size() != 2, "stride must have 2 elements.");

      // Parameters
      std::int64_t kernel_height = kernel_size[0];
      std::int64_t kernel_width = kernel_size[1];
      std::int64_t dilation_height = dilation[0];
      std::int64_t dilation_width = dilation[1];
      std::int64_t padding_height = padding[0];
      std::int64_t padding_width = padding[1];
      std::int64_t stride_height = stride[0];
      std::int64_t stride_width = stride[1];

      // Derived from the input shape
      std::int64_t batch_size = input_shape[0];
      std::int64_t n_input_plane = input_shape[1];
      std::int64_t input_height = input_shape[2];
      std::int64_t input_width = input_shape[3];

      std::int64_t n_output_plane = n_input_plane * kernel_width *
                                      kernel_height;

      int64_t output_height = (input_height + 2 * padding_height -
                     (dilation_height * (kernel_height - 1) + 1)) /
                      stride_height + 1;
      int64_t output_width = (input_width + 2 * padding_width -
                              (dilation_width * (kernel_width - 1) + 1)) /
                              stride_width + 1;
      std::int64_t output_length = output_height * output_width;

      llvm::SmallVector<std::int64_t, 3> shape{batch_size, n_output_plane,
                                               output_length};
      $_state.addTypes(mlir::RankedTensorType::get(shape,
                                                   getElementType(self)));
    }]>
  ];
}

def Poptorch_col2im : Poptorch_upsample<"col2im"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64ArrayAttr:$output_size,
                       I64ArrayAttr:$kernel_size,
                       I64ArrayAttr:$dilation,
                       I64ArrayAttr:$padding,
                       I64ArrayAttr:$stride);
  let results = (outs Poptorch_tensor:$result);
  let builders = [
    OpBuilder<(ins "mlir::Value":$self,
                   "const std::vector<std::int64_t>&":$output_size,
                   "const std::vector<std::int64_t>&":$kernel_size,
                   "const std::vector<std::int64_t>&":$dilation,
                   "const std::vector<std::int64_t>&":$padding,
                   "const std::vector<std::int64_t>&":$stride), [{
      $_state.addOperands(self);

      $_state.addAttribute("output_size",
                           $_builder.getI64ArrayAttr(output_size));
      $_state.addAttribute("kernel_size",
                           $_builder.getI64ArrayAttr(kernel_size));
      $_state.addAttribute("dilation",
                           $_builder.getI64ArrayAttr(dilation));
      $_state.addAttribute("padding", $_builder.getI64ArrayAttr(padding));
      $_state.addAttribute("stride", $_builder.getI64ArrayAttr(stride));

      auto input_shape = getShape(self);
      ERROR_ON_MSG(input_shape.size() != 3,
                   "Input must have 3 dimensions.");

      ERROR_ON_MSG(output_size.size() != 2,
                   "output_size must have 2 elements.");
      ERROR_ON_MSG(kernel_size.size() != 2,
                   "kernel_size must have 2 elements.");

      // Parameters
      std::int64_t output_height = output_size[0];
      std::int64_t output_width = output_size[1];
      std::int64_t kernel_height = kernel_size[0];
      std::int64_t kernel_width = kernel_size[1];

      // Derived from the input shape
      std::int64_t batch_size = input_shape[0];
      std::int64_t n_input_plane = input_shape[1];

      std::int64_t n_output_plane = n_input_plane / (kernel_width *
                                      kernel_height);

      std::int64_t output_length = output_height * output_width;

      llvm::SmallVector<std::int64_t, 4> shape{batch_size, n_output_plane,
                                               output_height, output_width};
      $_state.addTypes(mlir::RankedTensorType::get(shape,
                                                   getElementType(self)));
    }]>
  ];
}

def Poptorch_unfold : Poptorch_upsample<"unfold"> {
  let arguments = (ins Poptorch_tensor:$self,
                       I64Attr:$dimension,
                       I64Attr:$size,
                       I64Attr:$step);
  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::int64_t":$dimension,
                                 "std::int64_t":$size,
                                 "std::int64_t":$step), [{
    $_state.addOperands(self);

    const auto input_shape = getShape(self);
    dimension = convertToPositiveDim(dimension, input_shape.size());

    $_state.addAttribute("dimension",
                         $_builder.getI64IntegerAttr(dimension));
    $_state.addAttribute("size", $_builder.getI64IntegerAttr(size));
    $_state.addAttribute("step", $_builder.getI64IntegerAttr(step));

    auto output_shape = input_shape;
    output_shape[dimension] = (output_shape[dimension] - size) / step + 1;
    output_shape.push_back(size);

    $_state.addTypes(mlir::RankedTensorType::get(output_shape,
                                                 getElementType(self)));
  }]>];
}
