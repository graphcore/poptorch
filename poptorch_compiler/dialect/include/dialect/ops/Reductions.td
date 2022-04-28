// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

class Poptorch_reduce<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$input, I64ArrayAttr:$axes, BoolAttr:$keepdim);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values,
                                      "const std::vector<std::int64_t>&":$axes,
                                      "std::int64_t":$keepdim),[{
        $_state.addOperands(values);
        $_state.addAttribute("axes", $_builder.getI64ArrayAttr(axes));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        mlir::RankedTensorType tensor = values[0].getType().cast<mlir::RankedTensorType>();

        auto ref = tensor.getShape();

        llvm::SmallVector<std::int64_t, 4> shape {ref.begin(), ref.end()};

        // Flatten those axes.
        for (std::int64_t dim : axes) {
            // Dim reduced to 1 or zero depending on keep dim.
            shape[dim] = (std::int64_t)keepdim;
        }

        // Remove all zeros if we aren't keeping the dim.

        if (!keepdim) {
            if (ref.size() == axes.size()) {
                $_state.addTypes(mlir::RankedTensorType::get({}, tensor.getElementType()));
                return;
            }

            shape.erase(std::remove(shape.begin(), shape.end(), 0), shape.end());
        }

        $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
     }]>
    ];
}

class Poptorch_partial<string name> : Poptorch_NotImplementedOp<name, []> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "std::int64_t":$dim),[{
        $_state.addOperands({self});
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        $_state.addTypes(tensor);
     }]>
    ];
}

class Poptorch_NotImplemented_reduce_single_dim<string name> : Poptorch_NotImplementedOp<name, []> {
    let arguments = (ins Poptorch_tensor:$self, OptionalAttr<I64Attr>:$dim, BoolAttr:$keepdim);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "std::optional<std::int64_t>":$dim,
                                      "std::int64_t":$keepdim),[{
        $_state.addOperands(self);

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        if (dim.has_value()) {
            *dim = convertToPositiveDim(*dim, tensor.getShape().size());

            $_state.addAttribute("dim", $_builder.getI64IntegerAttr(*dim));
        }
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        auto getShape = [&]() {
            // We are flattenning the dimenion we're passed
            if (dim.has_value()) {
                auto ref = tensor.getShape();

                llvm::SmallVector<std::int64_t, 4> shape{ref.begin(), ref.end()};

                if (!keepdim) {
                    shape.erase(shape.begin() + *dim);
                } else {
                    shape[*dim] = 1;
                }

                return shape;
            } else {
                return llvm::SmallVector<std::int64_t, 4>{0};
            }
        };

        $_state.addTypes(mlir::RankedTensorType::get(getShape(), $_builder.getI32Type()));
     }]>
    ];
}


def Poptorch_reducemean : Poptorch_reduce<"reducemean"> {}
def Poptorch_reducesum : Poptorch_reduce<"reducesum"> {}

def Poptorch_cumsum_out : Poptorch_partial<"cumsum_out"> {}

def Poptorch_argmax_out : Poptorch_NotImplemented_reduce_single_dim<"argmax_out"> {}
def Poptorch_argmin_out : Poptorch_NotImplemented_reduce_single_dim<"argmin_out"> {}


// From the docs: https://pytorch.org/docs/stable/generated/torch.all.html
// NOTE: This function matches the behaviour of NumPy in returning output of dtype bool
// for all supported dtypes except uint8. For uint8 the dtype of output is uint8 itself.
class Poptorch_binary_reduction<string name> : Poptorch_NotImplementedOp<name, []> {
    let arguments = (ins Poptorch_tensor:$input);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values),[{
        $_state.addOperands(values);

        mlir::RankedTensorType tensor = values[0].getType().cast<mlir::RankedTensorType>();

        if (tensor.getElementType().isUnsignedInteger(8)) {
            $_state.addTypes(mlir::RankedTensorType::get({}, tensor.getElementType()));
        }
        else {
            $_state.addTypes(mlir::RankedTensorType::get({}, $_builder.getIntegerType(1, false)));
        }
     }]>
    ];
}

def Poptorch_all : Poptorch_binary_reduction<"all"> {}
def Poptorch_any : Poptorch_binary_reduction<"any"> {}

// From the docs: https://pytorch.org/docs/stable/generated/torch.all.html
// NOTE: This function matches the behaviour of NumPy in returning output of dtype bool
// for all supported dtypes except uint8. For uint8 the dtype of output is uint8 itself.
class Poptorch_binary_reduction_out<string name> : Poptorch_NotImplementedOp<name, []> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim, BoolAttr:$keepdim);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::ValueRange":$values, "std::int64_t":$dim, "bool":$keepdim),[{
        $_state.addOperands(values);

        mlir::RankedTensorType tensor = values[0].getType().cast<mlir::RankedTensorType>();
        const auto ref = tensor.getShape();

        dim = convertToPositiveDim(dim, ref.size());

        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        llvm::SmallVector<std::int64_t, 4> shape {ref.begin(), ref.end()};

        if (keepdim) {
            shape[dim] = 1;
        } else {
            shape.erase(shape.begin() + dim);
        }

        if (tensor.getElementType().isUnsignedInteger(8)) {
            $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
        }
        else {
            $_state.addTypes(mlir::RankedTensorType::get(shape, $_builder.getIntegerType(1, false)));
        }
     }]>
    ];
}

def Poptorch_all_out : Poptorch_binary_reduction_out<"all_out"> {}
def Poptorch_any_out : Poptorch_binary_reduction_out<"any_out"> {}

def Poptorch_prod : Poptorch_NotImplementedOp<"prod", []> {
    let arguments = (ins Poptorch_tensor:$input);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self),[{
        $_state.addOperands(self);

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        $_state.addTypes(mlir::RankedTensorType::get({}, tensor.getElementType()));
     }]>
    ];
}

def Poptorch_argsort : Poptorch_NotImplementedOp<"argsort", []> {
    let arguments = (ins Poptorch_tensor:$self, I64Attr:$dim, BoolAttr:$descending);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "std::int64_t":$dim,
                                      "std::int64_t":$descending),[{
        $_state.addOperands(self);
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("descending", $_builder.getBoolAttr(descending));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        $_state.addTypes(mlir::RankedTensorType::get(tensor.getShape(), $_builder.getIntegerType(32, true)));
     }]>
    ];
}

class Poptorch_NotImplemented_std_var<string name> : Poptorch_NotImplementedOp<name, []> {
    let arguments = (ins Poptorch_tensor:$self, BoolAttr:$unbiased);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "bool":$unbiased),[{
        $_state.addOperands(self);
        $_state.addAttribute("unbiased", $_builder.getBoolAttr(unbiased));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        $_state.addTypes(getTypes(mlir::RankedTensorType::get({}, tensor.getElementType())));
     }]>
    ];

    let extraClassDeclaration = [{
        static std::vector<mlir::Type> getTypes(mlir::Type type) {
            return {type};
        }
    }];
}

def Poptorch_standard_deviation : Poptorch_NotImplemented_std_var<"standard_deviation"> {}
def Poptorch_var : Poptorch_NotImplemented_std_var<"var"> {}

class Poptorch_NotImplemented_std_var_mean<string name> : Poptorch_NotImplemented_std_var<name> {
    let extraClassDeclaration = [{
        static std::vector<mlir::Type> getTypes(mlir::Type type) {
            return {type, type};
        }
    }];
}

def Poptorch_std_mean : Poptorch_NotImplemented_std_var_mean<"std_mean"> {}
def Poptorch_var_mean : Poptorch_NotImplemented_std_var_mean<"var_mean"> {}

class Poptorch_NotImplemented_std_var_correction<string name> : Poptorch_NotImplementedOp<name, []> {
    let arguments = (ins Poptorch_tensor:$self,
                         OptionalAttr<I64ArrayAttr>:$dim,
                         OptionalAttr<I64Attr>:$correction,
                         BoolAttr:$keepdim);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                      "std::optional<std::vector<std::int64_t>>":$dim,
                                      "std::optional<std::int64_t>":$correction,
                                      "bool":$keepdim),[{
        $_state.addOperands(self);

        llvm::SmallVector<std::int64_t, 4> shape;
        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        if (dim.has_value()) {
            *dim = convertToPositiveDim(*dim, tensor.getShape().size());

            // Sort in descending order so when we erase dimension it doesn't invalidate future erasings
            std::sort(dim->begin(), dim->end(), std::greater<>{});
            shape.insert(shape.begin(), tensor.getShape().begin(), tensor.getShape().end());
            // Remove the specified dimensions from the shape inference
            for (auto itr = dim->rbegin(); itr != dim->rend(); ++itr) {
                if (keepdim) {
                    shape[*itr] = 1;
                }
                else {
                    shape.erase(shape.begin() + *itr);
                }
            }

            $_state.addAttribute("dim", $_builder.getI64ArrayAttr(*dim));
        }
        if (correction.has_value()) {
            $_state.addAttribute("correction", $_builder.getI64IntegerAttr(*correction));
        }
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        $_state.addTypes(getTypes(mlir::RankedTensorType::get(shape, tensor.getElementType())));
     }]>
    ];

    let extraClassDeclaration = [{
        static std::vector<mlir::Type> getTypes(mlir::Type type) {
            return {type};
        }
    }];
}

def Poptorch_std_correction : Poptorch_NotImplemented_std_var_correction<"std_correction"> {}
def Poptorch_var_correction : Poptorch_NotImplemented_std_var_correction<"var_correction"> {}

class Poptorch_NotImplemented_std_var_mean_correction<string name> : Poptorch_NotImplemented_std_var_correction<name> {
    let extraClassDeclaration = [{
        static std::vector<mlir::Type> getTypes(mlir::Type type) {
            return {type, type};
        }
    }];
}

def Poptorch_std_mean_correction : Poptorch_NotImplemented_std_var_mean_correction<"std_mean_correction"> {}
def Poptorch_var_mean_correction : Poptorch_NotImplemented_std_var_mean_correction<"var_mean_correction"> {}

// func: _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
// NOTE: shape inference for the cpu version of embedding_bag may by found in pytorch here:
// pytorch/aten/src/ATen/native/EmbeddingBag.cpp:_embedding_bag_cpu_impl()
def Poptorch_embedding_bag : Poptorch_NotImplementedOp<"embedding_bag", []> {
    let arguments = (ins Poptorch_tensor:$weight,
                         Poptorch_tensor:$indices,
                         Poptorch_tensor:$offsets,
                         BoolAttr:$scale_grad_by_freq,
                         I64Attr:$mode,
                         BoolAttr:$sparse,
                         Optional<Poptorch_tensor>:$per_sample_weights,
                         BoolAttr:$include_last_offset,
                         I64Attr:$padding_idx);
    let results = (outs Poptorch_tensorlist:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$weight,
                                      "mlir::Value":$indices,
                                      "mlir::Value":$offsets,
                                      "bool":$scale_grad_by_freq,
                                      "std::int64_t":$mode,
                                      "bool":$sparse,
                                      "mlir::Value":$per_sample_weights,
                                      "bool":$include_last_offset,
                                      "std::int64_t":$padding_idx),[{
        std::vector<mlir::Value> operands{weight, indices, offsets};

        bool has_per_sample_weights = static_cast<bool>(per_sample_weights);

        if (has_per_sample_weights) {
            operands.push_back(per_sample_weights);
        }

        $_state.addOperands(operands);

        $_state.addAttribute("scale_grad_by_freq", $_builder.getBoolAttr(scale_grad_by_freq));
        $_state.addAttribute("mode", $_builder.getI64IntegerAttr(mode));
        $_state.addAttribute("sparse", $_builder.getBoolAttr(sparse));
        $_state.addAttribute("include_last_offset", $_builder.getBoolAttr(include_last_offset));
        $_state.addAttribute("padding_idx", $_builder.getI64IntegerAttr(padding_idx));
        $_state.addAttribute("has_per_sample_weights", $_builder.getBoolAttr(has_per_sample_weights));

        auto weight_tensor = weight.getType().cast<mlir::RankedTensorType>();
        auto indices_tensor = indices.getType().cast<mlir::RankedTensorType>();
        auto offsets_tensor = offsets.getType().cast<mlir::RankedTensorType>();

        const auto num_bags = offsets_tensor.getShape()[0] - (include_last_offset ? 1 : 0);

        const auto output = mlir::RankedTensorType::get(
            {num_bags, weight_tensor.getShape()[1]},
            weight_tensor.getElementType());
        // It looks like this might be `indices_tensor.getShape()[0] + 1` in some cases but these all get overwritten
        const auto offset2bag = mlir::RankedTensorType::get(
            {indices_tensor.getShape()[0]},
            offsets_tensor.getElementType());
        const auto bag_size = mlir::RankedTensorType::get({num_bags}, offsets_tensor.getElementType());
        const auto max_indices = mlir::RankedTensorType::get({num_bags}, offsets_tensor.getElementType());

        $_state.addTypes({output, offset2bag, bag_size, max_indices});
    }]>];
}