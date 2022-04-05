// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> (Tensor(a))
def Poptorch_as_strided: Poptorch_Op<"as_strided", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$size, I64ArrayAttr:$strides);

    let results = (outs Poptorch_tensor:$result);


    let builders = [OpBuilderDAG<(ins "mlir::Value":$val1,
                                      "const std::vector<std::int64_t>&":$size,
                                      "const std::vector<std::int64_t>&":$strides),[{
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
}



// aten::transpose.int(Tensor(a) self, int dim0, int dim1)
def Poptorch_transpose: Poptorch_Op<"transpose", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim0,  I64Attr:$dim1);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "std::int64_t":$dim0, "std::int64_t":$dim1),[{
        $_state.addOperands({value});
        $_state.addAttribute("dim0", $_builder.getI64IntegerAttr(dim0));
        $_state.addAttribute("dim1", $_builder.getI64IntegerAttr(dim1));

        // Transpose the type.
        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        llvm::SmallVector<std::int64_t, 4> shape = {t1.getShape().begin(),  t1.getShape().end()};

        if (dim0 < shape.size() && dim1 < shape.size()) {
            std::swap(shape[dim0], shape[dim1]);
        }

        $_state.addTypes(mlir::RankedTensorType::get(shape, t1.getElementType()));
     }]>
    ];
}


def Poptorch_reshape: Poptorch_Op<"reshape", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$shape);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "const std::vector<std::int64_t>&":$shape),[{
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
                      << ", in Poptorch_reshape_generic");
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

        // Otherwise, the reshape is succesful
        // Ensure the attribute is the resolved shape (without -1)
        $_state.addAttribute("shape", $_builder.getI64ArrayAttr(shape_out));

        mlir::Type e1 = t1.getElementType();
        $_state.addTypes(mlir::RankedTensorType::get(shape_out, e1));
     }]>
    ];
}


// Expand is like reshape but can broadcast unity dimensions and sets -1
// dimensions to the same as the input. New dimensions are added at the front
// TODO T52507 Fully inplement expand
def Poptorch_expand: Poptorch_Op<"expand", [ViewOp]> {
      let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$shape);
  let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "const std::vector<std::int64_t>&":$shape),[{
        $_state.addOperands({value});
        $_state.addAttribute("shape", $_builder.getI64ArrayAttr(shape));

        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        mlir::Type e1 = t1.getElementType();
        $_state.addTypes(mlir::RankedTensorType::get(shape, e1));
     }]>
    ];
}


// squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
def Poptorch_squeeze_dim: Poptorch_Op<"squeeze_dim", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "std::int64_t":$dim),[{
        $_state.addOperands({value});
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        mlir::Type e1 = t1.getElementType();
        auto shape = t1.getShape();
        if (dim < 0) {
            dim += shape.size();
        }
        std::vector<int64_t> new_shape = shape;
        if (shape[dim] == 1) {
            new_shape.erase(new_shape.begin() + dim);
        }
        $_state.addTypes(mlir::RankedTensorType::get(new_shape, e1));
     }]>
    ];
}

// squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
def Poptorch_squeeze_dim_: Poptorch_Op<"squeeze_dim_", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "std::int64_t":$dim),[{
        $_state.addOperands({value});
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

        mlir::RankedTensorType t1 = value.getType().cast<mlir::RankedTensorType>();
        mlir::Type e1 = t1.getElementType();
        auto shape = t1.getShape();
        if (dim < 0) {
            dim += shape.size();
        }
        std::vector<int64_t> new_shape = shape;
        if (shape[dim] == 1) {
            new_shape.erase(new_shape.begin() + dim);
        }
        $_state.addTypes(mlir::RankedTensorType::get(new_shape, e1));
     }]>
    ];
}

// select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
def Poptorch_select : Poptorch_Op<"select", [ViewOp]> {
    let arguments = (ins Poptorch_tensor:$self,
                         I64Attr:$dim,
                         I64Attr:$idx);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "std::int64_t":$dim,
                                      "std::int64_t":$idx),[{

        // Operands
        $_state.addOperands({self});

        // Attributes
        mlir::RankedTensorType input_tensor = self.getType().cast<mlir::RankedTensorType>();
        llvm::SmallVector<std::int64_t, 4> out_shape {input_tensor.getShape().begin(),
                                                      input_tensor.getShape().end()};

        if (dim < 0) {
            dim += out_shape.size();
        }
        if (idx < 0) {
            idx += out_shape[dim];
        }

        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("idx", $_builder.getI64IntegerAttr(idx));

        // Returns
        out_shape.erase(out_shape.begin() + dim);

        auto out_type = mlir::RankedTensorType::get(out_shape, input_tensor.getElementType());

        $_state.addTypes({out_type});
     }]>
    ];
}

// unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
def Poptorch_unsqueeze : Poptorch_Op<"unsqueeze", []> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim);
    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilderDAG<(ins "mlir::Value":$value, "std::int64_t":$dim),[{
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

}

def Poptorch_constant_pad_nd : Poptorch_NotImplementedOp<"constant_pad_nd", []> {
    let arguments = (ins Poptorch_tensor:$self, I64ArrayAttr:$pad, F32Attr:$scalar);
    let results = (outs Poptorch_tensor:$result);
    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "std::vector<std::int64_t>":$pad,
                                      "float":$scalar),[{
        $_state.addOperands({self});
        $_state.addAttribute("pad", $_builder.getI64ArrayAttr(pad));
        $_state.addAttribute("scalar", $_builder.getF32FloatAttr(scalar));

        ERROR_ON_MSG(pad.size() % 2 != 0,
                     "Length of pad must be even but instead is equals " +
                     std::to_string(pad.size()));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        const auto input_shape = tensor.getShape();

        ERROR_ON_MSG(pad.size() > 2 * input_shape.size(),
                     "Length of pad should be no more that twice the number of dimension "
                     "of the input. Pad has " + std::to_string(pad.size()) +
                     " dimensions while the input has " +
                     std::to_string(input_shape.size()) + " dimensions");

        llvm::SmallVector<std::int64_t, 4> shape{input_shape.begin(),
                                                 input_shape.end()};

        for (int i = 0; i < pad.size() / 2; ++i) {
            const auto dim = shape.size() - 1 - i;
            const auto lpad = pad[2 * i];
            const auto rpad = pad[2 * i + 1];

            ERROR_ON_MSG(shape[dim] + std::min(lpad, 0l) + std::min(rpad, 0l) < 0,
                         "You cannot pad negatively by more than the size of dimension " +
                         std::to_string(dim));
            shape[dim] += lpad + rpad;
        }

        $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
    }]>];
}

def Poptorch_index_select : Poptorch_NotImplementedOp<"index_select", []> {
    let arguments = (ins Poptorch_tensor:$self, I64Attr:$dim, Poptorch_tensor:$index);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "std::int64_t":$dim,
                                      "mlir::Value":$index),[{
        $_state.addOperands({self, index});
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        const auto input_shape = tensor.getShape();

        ERROR_ON_MSG(dim >= input_shape.size() || -dim > input_shape.size(),
                     "Cannot index outside the tensor (dims " +
                     std::to_string(input_shape.size()) + ") with dim (" +
                     std::to_string(dim) + ")");

        if (dim < 0) dim += input_shape.size();

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

def Poptorch_permute : Poptorch_NotImplementedOp<"permute", []> {
    let arguments = (ins Poptorch_tensor:$self, I64ArrayAttr:$dims);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
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
                         "dims must be a perumation. Missing dim " +
                         std::to_string(i));
        }

        llvm::SmallVector<std::int64_t, 4> shape;
        shape.reserve(dims.size());

        for (auto dim : dims) {
            shape.push_back(input_shape[dim >= 0 ? dim : dim + dims.size()]);
        }

        $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
    }]>];
}