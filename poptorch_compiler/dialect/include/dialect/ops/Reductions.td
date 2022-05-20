// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

class Poptorch_reduce<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$input,
                         I64ArrayAttr:$axes,
                         BoolAttr:$keepdim,
                         OptionalAttr<TypeAttr>:$dtype);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::ValueRange":$values,
                                      "const std::vector<std::int64_t>&":$axes,
                                      "std::int64_t":$keepdim,
                                      "std::optional<mlir::Type>":$dtype),[{
        $_state.addOperands(values);

        mlir::RankedTensorType tensor = values[0].getType().cast<mlir::RankedTensorType>();
        auto in_shape = tensor.getShape();

        auto dims = parseDims(axes, in_shape.size());

        $_state.addAttribute("axes", $_builder.getI64ArrayAttr(dims));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));
        if (dtype.has_value()) {
            $_state.addAttribute("dtype", mlir::TypeAttr::get(*dtype));
        }

        const auto in_element_type = tensor.getElementType();

        // It appears that the promotion rules for torch are the similar to numpy.
        // From the numpy manual (https://numpy.org/doc/stable/reference/generated/numpy.sum.html):
        //  The dtype of a is used by default unless a has an integer dtype of less precision
        //  than the default platform integer. In that case, if a is signed then the platform
        //  integer is used while if a is unsigned then an unsigned integer of the same
        //  precision as the platform integer is used.
        // Although, from testing it appears that pytorch converts uint8-s (the only unsigned type pytorch
        // supports) to a signed type
        // Note: our intergal type is 32 bit and we explicity don't support 64 bit at the moment so every
        // integral type is converted to a signed 32 bit integer
        // TODO(T62262): This will need to be updated when we can have int64 inputs
        auto out_element_type =
            dtype.value_or(
                in_element_type.isa<mlir::IntegerType>()
                    ? $_builder.getIntegerType(32, true)
                    : in_element_type);

        $_state.addTypes(mlir::RankedTensorType::get(inferShape(std::move(dims), in_shape, keepdim), out_element_type));
     }]>
    ];

    let extraClassDeclaration = [{
        static std::vector<std::int64_t> parseDims(const std::vector<std::int64_t>& axes,
                                                   std::size_t in_dim_count) {
            auto dims = convertToPositiveDim(axes, in_dim_count);

            // NOTE: an empty list of dimensions means we're reducing over all the dimensions
            if (dims.empty()) {
                dims.resize(in_dim_count);
                std::iota(dims.begin(), dims.end(), 0);
            }

            // If we are dealing with a scalar value don't reduce at all
            if (in_dim_count == 0) {
                dims.clear();
            }

            return dims;
        }
        static llvm::SmallVector<std::int64_t, 4> inferShape(std::vector<std::int64_t> dims,
                                                             llvm::ArrayRef<std::int64_t> in_shape,
                                                             bool keepdim) {
            llvm::SmallVector<std::int64_t, 4> shape {in_shape.begin(), in_shape.end()};

            // Optimization for when we are reducing over all the dimensions or the input is a scalar
            if (in_shape.size() == dims.size() || in_shape.empty()) {
                if (!keepdim) {
                    return {};
                } else {
                    std::fill(shape.begin(), shape.end(), 1);

                    return shape;
                }
            }

            // Sort in descending order so erasing elements doesn't invalidate future erases
            std::sort(dims.begin(), dims.end(), std::greater<>{});

            // Flatten those dims.
            for (std::int64_t dim : dims) {
                // Dim reduced to 1 or zero depending on keep dim.
                if (keepdim) {
                    shape[dim] = 1;
                } else {
                    shape.erase(shape.begin() + dim);
                }
            }

            return shape;
        }
    }];
}

def Poptorch_reducemean : Poptorch_reduce<"reducemean"> {}
def Poptorch_reducesum : Poptorch_reduce<"reducesum"> {}

def Poptorch_prod : Poptorch_Op<"prod", []> {
    let arguments = (ins Poptorch_tensor:$input, OptionalAttr<TypeAttr>:$dtype);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "std::optional<mlir::Type>":$dtype),[{
        $_state.addOperands(self);
        if (dtype.has_value()) {
            $_state.addAttribute("dtype", mlir::TypeAttr::get(*dtype));
        }

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        const auto in_element_type = tensor.getElementType();
        // TODO(T62262): This will need to be updated when we can have int64 inputs
        auto out_element_type =
            dtype.value_or(
                in_element_type.isa<mlir::IntegerType>()
                    ? $_builder.getIntegerType(32, true)
                    : in_element_type);

        $_state.addTypes(mlir::RankedTensorType::get({}, out_element_type));
     }]>
    ];
}


def Poptorch_prod_dim : Poptorch_reduce<"prod_dim"> {
    let arguments = (ins Poptorch_tensor:$input,
                         I64Attr:$dim,
                         BoolAttr:$keepdim,
                         OptionalAttr<TypeAttr>:$dtype);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "std::int64_t":$dim,
                                      "bool":$keepdim,
                                      "std::optional<mlir::Type>":$dtype),[{
        $_state.addOperands(self);

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        const auto ref = tensor.getShape();

        dim = convertToPositiveDim(dim, ref.size());

        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));
        if (dtype.has_value()) {
            $_state.addAttribute("dtype", mlir::TypeAttr::get(*dtype));
        }

        const auto shape = inferShape({dim}, ref, keepdim);

        const auto in_element_type = tensor.getElementType();
        // TODO(T62262): This will need to be updated when we can have int64 inputs
        auto out_element_type =
            dtype.value_or(
                in_element_type.isa<mlir::IntegerType>()
                    ? $_builder.getIntegerType(32, true)
                    : in_element_type);

        $_state.addTypes(mlir::RankedTensorType::get(shape, out_element_type));
     }]>
    ];
}


// From the docs: https://pytorch.org/docs/stable/generated/torch.all.html
// NOTE: This function matches the behaviour of NumPy in returning output of dtype bool
// for all supported dtypes except uint8. For uint8 the dtype of output is uint8 itself.
class Poptorch_binary_reduction<string name> : Poptorch_reduce<name> {
    let arguments = (ins Poptorch_tensor:$input);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::ValueRange":$values),[{
        $_state.addOperands(values);

        mlir::RankedTensorType tensor = values[0].getType().cast<mlir::RankedTensorType>();

        const auto in_element_type = tensor.getElementType();
        const auto out_element_type = in_element_type.isUnsignedInteger(8)
                ? in_element_type
                : $_builder.getIntegerType(1, false);

        $_state.addTypes(mlir::RankedTensorType::get({}, out_element_type));
     }]>
    ];
}

def Poptorch_all : Poptorch_binary_reduction<"all"> {}
def Poptorch_any : Poptorch_binary_reduction<"any"> {}


// From the docs: https://pytorch.org/docs/stable/generated/torch.all.html
// NOTE: This function matches the behaviour of NumPy in returning output of dtype bool
// for all supported dtypes except uint8. For uint8 the dtype of output is uint8 itself.
class Poptorch_binary_reduction_out<string name> : Poptorch_reduce<name> {
    let arguments = (ins Poptorch_tensor:$input, I64Attr:$dim, BoolAttr:$keepdim);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::ValueRange":$values, "std::int64_t":$dim, "bool":$keepdim),[{
        $_state.addOperands(values);

        mlir::RankedTensorType tensor = values[0].getType().cast<mlir::RankedTensorType>();
        const auto ref = tensor.getShape();

        dim = convertToPositiveDim(dim, ref.size());

        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        const auto shape = inferShape({dim}, ref, keepdim);

        const auto in_element_type = tensor.getElementType();
        const auto out_element_type = in_element_type.isUnsignedInteger(8)
                ? in_element_type
                : $_builder.getIntegerType(1, false);

        $_state.addTypes(mlir::RankedTensorType::get(shape, out_element_type));
     }]>
    ];
}

def Poptorch_all_out : Poptorch_binary_reduction_out<"all_out"> {}
def Poptorch_any_out : Poptorch_binary_reduction_out<"any_out"> {}


class Poptorch_std_var_correction<string name> : Poptorch_reduce<name> {
    let arguments = (ins Poptorch_tensor:$self,
                         OptionalAttr<I64ArrayAttr>:$dim,
                         OptionalAttr<I64Attr>:$correction,
                         BoolAttr:$keepdim);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "std::optional<std::vector<std::int64_t>>":$dim,
                                      "std::optional<std::int64_t>":$correction,
                                      "bool":$keepdim),[{
        $_state.addOperands(self);

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        auto in_shape = tensor.getShape();

        dim = parseDims(dim.value_or(std::vector<std::int64_t>{}), in_shape.size());

        $_state.addAttribute("dim", $_builder.getI64ArrayAttr(*dim));
        // Note: correction is always in {0, 1} and corresponds to whether to apply Bessel's correction
        $_state.addAttribute("correction", $_builder.getI64IntegerAttr(correction.value_or(1)));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        const auto shape = inferShape(std::move(*dim), in_shape, keepdim);

        $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
     }]>
    ];
}

def Poptorch_std_correction : Poptorch_std_var_correction<"std_correction"> {}
def Poptorch_var_correction : Poptorch_std_var_correction<"var_correction"> {}


class Poptorch_std_var_mean_correction<string name> : Poptorch_std_var_correction<name> {
    let results = (outs Poptorch_tensor:$result, Poptorch_tensor:$mean);
    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "std::optional<std::vector<std::int64_t>>":$dim,
                                      "std::optional<std::int64_t>":$correction,
                                      "bool":$keepdim),[{
        $_state.addOperands(self);

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();
        auto in_shape = tensor.getShape();

        dim = parseDims(dim.value_or(std::vector<std::int64_t>{}), in_shape.size());

        $_state.addAttribute("dim", $_builder.getI64ArrayAttr(*dim));
        // Note: correction is always in {0, 1} and corresponds to whether to apply Bessel's correction
        $_state.addAttribute("correction", $_builder.getI64IntegerAttr(correction.value_or(1)));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        const auto shape = inferShape(std::move(*dim), in_shape, keepdim);

        $_state.addTypes({
                mlir::RankedTensorType::get(shape, tensor.getElementType()),
                mlir::RankedTensorType::get(shape, tensor.getElementType())
            });
     }]>
    ];
}

def Poptorch_std_mean_correction : Poptorch_std_var_mean_correction<"std_mean_correction"> {}
def Poptorch_var_mean_correction : Poptorch_std_var_mean_correction<"var_mean_correction"> {}

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
    let results = (outs Poptorch_tensor:$output,
                        Poptorch_tensor:$offset2bag,
                        Poptorch_tensor:$bag_size,
                        Poptorch_tensor:$max_indices);

    let builders = [OpBuilder<(ins "mlir::Value":$weight,
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


class Poptorch_prefix_sum<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$input,
                         I64Attr:$dim,
                         OptionalAttr<TypeAttr>:$dtype);

    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                      "std::int64_t":$dim,
                                      "std::optional<mlir::Type>":$dtype),[{
        $_state.addOperands({self});

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        dim = convertToPositiveDim(dim, tensor.getShape().size());
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        if (dtype.has_value()) {
            $_state.addAttribute("dtype", mlir::TypeAttr::get(*dtype));
        }

        const auto in_element_type = tensor.getElementType();
        // TODO(T62262): This will need to be updated when we can have int64 inputs
        auto out_element_type =
            dtype.value_or(
                in_element_type.isa<mlir::IntegerType>()
                    ? $_builder.getIntegerType(32, true)
                    : in_element_type);

        $_state.addTypes(mlir::RankedTensorType::get(tensor.getShape(), out_element_type));
     }]>
    ];
}

def Poptorch_cumsum_out : Poptorch_prefix_sum<"cumsum_out"> {}
