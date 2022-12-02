// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/*
 * Linalg operations.
 */


def Poptorch_conv : Poptorch_Op<"conv", []> {
  let arguments = (ins Poptorch_tensor:$input,
                        Poptorch_tensor:$weight,
                        Optional<Poptorch_tensor>:$bias,
                        I64ArrayAttr:$stride, I64ArrayAttr:$padding,
                        I64ArrayAttr:$dilation, BoolAttr:$transposed, I64ArrayAttr:$output_padding, I64Attr:$groups);

  let results = (outs Poptorch_tensor:$result);
  let builders = [OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$weights, "mlir::Value":$bias,
                                "const std::vector<std::int64_t>&":$stride,
                                "const std::vector<std::int64_t>&":$padding, "const std::vector<std::int64_t>&":$dilation,
                                "bool": $transposed,
                                "const std::vector<std::int64_t>&":$output_padding, "std::int64_t":$groups), [{
        if (bias) {
            $_state.addOperands({input, weights, bias});
        } else {
            $_state.addOperands({input, weights});
        }
        $_state.addAttribute("stride", $_builder.getI64ArrayAttr(stride));
        $_state.addAttribute("padding", $_builder.getI64ArrayAttr(padding));
        $_state.addAttribute("dilation", $_builder.getI64ArrayAttr(dilation));
        $_state.addAttribute("output_padding",
                             $_builder.getI64ArrayAttr(output_padding));
        $_state.addAttribute("groups", $_builder.getI64IntegerAttr(groups));
        $_state.addAttribute("transposed", $_builder.getBoolAttr(transposed));

        const auto weights_shape = getShape(weights);
        const auto input_shape = getShape(input);

        llvm::SmallVector<std::int64_t, 4> out_shape{input_shape.begin(),
                                                     input_shape.end()};

        // See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        // and https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        // for deduction rules.
        
        if (transposed) {
            out_shape[1] = weights_shape[1];
            for (std::size_t i = 0; i < out_shape.size() - 2; ++i) {
                const std::size_t i_dim = i + 2;
                out_shape[i_dim] = (input_shape[i_dim] - 1) * stride[i] - 2 *
                                   padding[i] + dilation[i] *
                                   (weights_shape[i_dim] - 1) +
                                   output_padding[i] + 1;
            }
        } else {
            out_shape[1] = weights_shape[0];
            for (std::size_t i = 0; i < out_shape.size() - 2; ++i) {
                const std::size_t i_dim = i + 2;
                out_shape[i_dim] = ((input_shape[i_dim] + 2 * padding[i] - dilation[i] * (weights_shape[i_dim] -1) - 1) / stride[i]) + 1;
            }
        }

        $_state.addTypes(mlir::RankedTensorType::get(out_shape, getElementType(input)));
  }]>];
}

// aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight,
//                            SymInt[]? bias_sizes, int[] stride, int[] padding,
//                            int[] dilation, bool transposed, int[] output_padding,
//                            int groups, bool[3] output_mask) ->
//                            (Tensor, Tensor, Tensor)
def Poptorch_conv_backward : Poptorch_Op<"conv_backward", []> {
  let arguments = (ins Poptorch_tensor:$grad_output,
                       Poptorch_tensor:$input,
                       Poptorch_tensor:$weight,
                       OptionalAttr<I64ArrayAttr>:$bias_sizes,
                       I64ArrayAttr:$stride, I64ArrayAttr:$padding,
                       I64ArrayAttr:$dilation, BoolAttr:$transposed,
                       I64ArrayAttr:$output_padding, I64Attr:$groups,
                       I64ArrayAttr:$output_mask);

  let results = (outs Poptorch_tensor:$grad_input,
                      Poptorch_tensor:$grad_weight,
                      Poptorch_tensor:$grad_bias);
  let builders = [OpBuilder<(ins "mlir::Value":$grad_output,
                                 "mlir::Value":$input,
                                 "mlir::Value":$weight,
                                 "std::optional<std::vector<std::int64_t>>":$bias_sizes,
                                 "const std::vector<std::int64_t>&":$stride,
                                 "const std::vector<std::int64_t>&":$padding,
                                 "const std::vector<std::int64_t>&":$dilation,
                                 "bool": $transposed,
                                 "const std::vector<std::int64_t>&":$output_padding,
                                 "std::int64_t":$groups,
                                 "const std::vector<std::int64_t>&":$output_mask), [{
        $_state.addOperands({grad_output, input, weight});
        $_state.addAttribute("stride",$_builder.getI64ArrayAttr(stride));
        $_state.addAttribute("padding",$_builder.getI64ArrayAttr(padding));
        $_state.addAttribute("dilation",$_builder.getI64ArrayAttr(dilation));
        $_state.addAttribute("output_padding",$_builder.getI64ArrayAttr(output_padding));
        $_state.addAttribute("groups",$_builder.getI64IntegerAttr(groups));
        $_state.addAttribute("transposed",$_builder.getBoolAttr(transposed));
        $_state.addAttribute("output_mask", $_builder.getI64ArrayAttr(output_mask));

        const auto weight_type = weight.getType().cast<mlir::RankedTensorType>();
        std::vector<std::int64_t> bias_shape = {weight_type.getShape()[0]};
        if (bias_sizes.has_value()) {
          bias_shape = bias_sizes.value();
          $_state.addAttribute("bias_sizes",$_builder.getI64ArrayAttr(bias_sizes.value()));
        }

        $_state.addTypes({input.getType(), weight.getType(),
                          mlir::RankedTensorType::get(bias_shape,
                                                      weight_type.getElementType())});
  }]>];
}

def Poptorch_matmul : Poptorch_Op<"matmul", []> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2),[{
        $_state.addOperands({v1, v2});
        $_state.addTypes(inferType(v1, v2));
     }]>
    ];

    let assemblyFormat = [{
        `(`$in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` `->` type($result) attr-dict
    }];

    let extraClassDeclaration = [{
        static mlir::Type inferType(mlir::Value v1, mlir::Value v2) {
            // Get the types of the inputs.
            const mlir::RankedTensorType t1 = asTensor(v1);
            const mlir::RankedTensorType t2 = asTensor(v2);

            // Elem type.
            const mlir::Type e1 = t1.getElementType();

            // Dot product.
            if (t1.getRank() == 1 && t2.getRank() == 1) {
                return mlir::RankedTensorType::get({}, e1);
            }

            // Vector-matrix product. [n, m, k, j, v] * [v] == [n, m, k, j]
            if (t1.getRank() == 1 && t2.getRank() > 1) {
                return mlir::RankedTensorType::get(t2.getShape().drop_back(1), e1);
            }

            // Matrix-vector product.
            if (t1.getRank() > 1 && t2.getRank() == 1) {
                return mlir::RankedTensorType::get(t1.getShape().drop_back(1), e1);
            }

            // Matrix-matrix product.
            auto s1 = t1.getShape();
            auto s2 = t2.getShape();
            auto shape = broadcast(s1, s2, 2);
            shape[shape.size() - 2] = s1[s1.size() - 2];
            shape[shape.size() - 1] = s2[s2.size() - 1];

            return mlir::RankedTensorType::get(shape, e1);
        }
    }];
}

def Poptorch_addmm : Poptorch_Op<"addmm", []> {
    let arguments = (ins Poptorch_tensor:$input,
                         Poptorch_tensor:$mat1,
                         Poptorch_tensor:$mat2,
                         F32Attr:$beta,
                         F32Attr:$alpha);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input,
                                   "mlir::Value":$mat1,
                                   "mlir::Value":$mat2,
                                   "float":$beta,
                                   "float":$alpha), [{
        $_state.addOperands({input, mat1, mat2});
        $_state.addAttribute("beta", $_builder.getF32FloatAttr(beta));
        $_state.addAttribute("alpha", $_builder.getF32FloatAttr(alpha));

        const auto mat1_shape = getShape(mat1);
        llvm::SmallVector<std::int64_t, 2> shape{mat1_shape.begin(),
                                                 mat1_shape.end()};
        ERROR_ON_MSG(shape.size() != 2,
                     "argument `mat1' to `addmm' must be a 2-D tensor");
        const auto mat2_shape = getShape(mat2);
        ERROR_ON_MSG(shape.size() != 2,
                     "argument `mat2' to `addmm' must be a 2-D tensor");
        shape[1] = mat2_shape[1];
        $_state.addTypes(mlir::RankedTensorType::get(shape,
                                                     getElementType(input)));
    }]>];
}

def Poptorch_baddbmm : Poptorch_Op<"baddbmm", []> {
    let arguments = (ins Poptorch_tensor:$input,
                         Poptorch_tensor:$batch1,
                         Poptorch_tensor:$batch2,
                         F32Attr:$beta,
                         F32Attr:$alpha);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input,
                                   "mlir::Value":$batch1,
                                   "mlir::Value":$batch2,
                                   "float":$beta,
                                   "float":$alpha), [{
        $_state.addOperands({input, batch1, batch2});
        $_state.addAttribute("beta", $_builder.getF32FloatAttr(beta));
        $_state.addAttribute("alpha", $_builder.getF32FloatAttr(alpha));

        auto batch1_shape = getShape(batch1);
        llvm::SmallVector<std::int64_t, 3> shape{batch1_shape.begin(),
                                                 batch1_shape.end()};
        ERROR_ON_MSG(shape.size() != 3,
                     "argument `batch1' to `baddbmm' must be a 3-D tensor");
        const auto batch2_shape = getShape(batch2);
        ERROR_ON_MSG(shape.size() != 3,
                     "argument `batch2' to `baddbmm' must be a 3-D tensor");
        shape[2] = batch2_shape[2];
        $_state.addTypes(mlir::RankedTensorType::get(shape,
                                                     getElementType(input)));
    }]>];
}

// addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def Poptorch_addmv : Poptorch_Op<"addmv", []> {
  let arguments = (ins Poptorch_tensor:$self,
                       Poptorch_tensor:$mat,
                       Poptorch_tensor:$vec,
                       F32Attr:$beta,
                       F32Attr:$alpha);

  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "mlir::Value":$mat,
                                 "mlir::Value":$vec,
                                 "float":$beta,
                                 "float":$alpha),[{
    $_state.addOperands({self, mat, vec});
    $_state.addAttribute("beta", $_builder.getF32FloatAttr(beta));
    $_state.addAttribute("alpha", $_builder.getF32FloatAttr(alpha));

    const auto n = getShape(mat)[0];
    $_state.addTypes(mlir::RankedTensorType::get({n}, getElementType(self)));
  }]>];
}

class Poptorch_not_implemented_norm<string mnemonic, list<Trait> traits = []> :
    Poptorch_NotImplementedOp<mnemonic, traits> {
    let extraClassDeclaration = [{
        static llvm::SmallVector<std::int64_t, 4> inferShape(const mlir::RankedTensorType &tensor,
                                                             std::vector<std::int64_t> dims,
                                                             bool keepdim) {
          if (dims.empty()) {
            return {};
          }

          const llvm::ArrayRef<std::int64_t> input_shape = tensor.getShape();
          llvm::SmallVector<std::int64_t, 4> shape;

          // Sort in descending order so when we erase a dimension it doesn't
          // invalidate future erasings
          std::sort(dims.begin(), dims.end(), std::greater<>{});
          shape.insert(shape.begin(), input_shape.begin(), input_shape.end());
          for (auto d : dims) {
            if (keepdim) {
              shape[d] = 1;
            } else {
              shape.erase(shape.begin() + d);
            }
          }

          return shape;
        }

        static mlir::RankedTensorType inferType(const mlir::Value &input,
                                                std::vector<std::int64_t> dims,
                                                bool keepdim) {
          return mlir::RankedTensorType::get(inferShape(asTensor(input), dims, keepdim),
                                             getElementType(input));
        }
    }];
}

def Poptorch_norm_out : Poptorch_not_implemented_norm<"norm_out"> {
    let arguments = (ins Poptorch_tensor:$self,
                         OptionalAttr<F64Attr>:$p,
                         I64ArrayAttr:$dim,
                         BoolAttr:$keepdim);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "std::optional<double>":$p,
                                   "const std::vector<std::int64_t>&":$dim,
                                   "bool":$keepdim),[{
        $_state.addOperands(self);
        const auto tensor = asTensor(self);

        if (p.has_value()) {
          $_state.addAttribute("p", $_builder.getF64FloatAttr(*p));
        }
        const auto actual_dims = convertToPositiveDim(dim, tensor.getRank());
        $_state.addAttribute("dim", $_builder.getI64ArrayAttr(actual_dims));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        $_state.addTypes(inferType(self, actual_dims, keepdim));
     }]>
    ];
}

def Poptorch_norm_dtype_out : Poptorch_not_implemented_norm<"norm_dtype_out"> {
    let arguments = (ins Poptorch_tensor:$self,
                         OptionalAttr<F64Attr>:$p,
                         I64ArrayAttr:$dim,
                         BoolAttr:$keepdim,
                         TypeAttr:$dtype);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "std::optional<double>":$p,
                                   "const std::vector<std::int64_t>&":$dim,
                                   "bool":$keepdim,
                                   "mlir::Type":$dtype),[{
        $_state.addOperands(self);
        const auto tensor = asTensor(self);

        if (p.has_value()) {
          $_state.addAttribute("p", $_builder.getF64FloatAttr(*p));
        }
        const auto actual_dims = convertToPositiveDim(dim, tensor.getRank());
        $_state.addAttribute("dim", $_builder.getI64ArrayAttr(actual_dims));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));
        $_state.addAttribute("dtype", mlir::TypeAttr::get(dtype));

        $_state.addTypes(inferType(self, actual_dims, keepdim));
     }]>
    ];
}

def Poptorch_frobenius_norm_out : Poptorch_not_implemented_norm<"frobenius_norm_out"> {
    let arguments = (ins Poptorch_tensor:$self,
                         I64ArrayAttr:$dim,
                         BoolAttr:$keepdim);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "const std::vector<std::int64_t>&":$dim,
                                   "bool":$keepdim),[{
        $_state.addOperands(self);
        const auto tensor = asTensor(self);
        poptorch::logging::warn("Undefined behaviour: aten::frobenius_norm was deprecated in PyTorch 1.12, use torch.linalg.norm() instead.");

        const auto actual_dims = convertToPositiveDim(dim, tensor.getRank());
        $_state.addAttribute("dim", $_builder.getI64ArrayAttr(actual_dims));
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        $_state.addTypes(inferType(self, actual_dims, keepdim));
     }]>
    ];
}

// func: cross(Tensor self, Tensor other, int? dim=None, *) -> Tensor(a!)
def Poptorch_cross : Poptorch_NotImplementedOp<"cross", []> {
    let arguments = (ins Poptorch_tensor:$self, Poptorch_tensor:$other,
                     OptionalAttr<I64Attr>:$dim);
    let results = (outs Poptorch_tensor:$result);
    let builders = [
        OpBuilder<(ins "mlir::Value":$self, "mlir::Value":$other,
                   "std::optional<std::int64_t>":$dim), [{
            $_state.addOperands({self, other});

            const auto out_shape = broadcast(getShape(self), getShape(other));

            if (dim.has_value()) {
                dim = convertToPositiveDim(*dim, out_shape.size());
                ERROR_ON_MSG(out_shape[*dim] != 3,
                    "Dimension length for cross must be 3.");
            } else {
                auto it = std::find(out_shape.begin(), out_shape.end(), 3);

                ERROR_ON_MSG(it == out_shape.end(),
                    "At least one dimension of the broadcast shape must be equal to 3.");

                dim = std::distance(out_shape.begin(), it);
            }
            $_state.addAttribute("dim", $_builder.getI64IntegerAttr(*dim));

            $_state.addTypes(mlir::RankedTensorType::get(out_shape, getElementType(self)));
        }]>
    ];
}

// func: bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor
def Poptorch_bilinear : Poptorch_NotImplementedOp<"bilinear", []> {
    let arguments = (ins Poptorch_tensor:$input1,
                         Poptorch_tensor:$input2,
                         Poptorch_tensor:$weight,
                         Optional<Poptorch_tensor>:$bias);
    let results = (outs Poptorch_tensor:$result);
    let builders = [
        OpBuilder<(ins "mlir::Value":$input1,
                       "mlir::Value":$input2,
                       "mlir::Value":$weight,
                       "mlir::Value":$bias), [{
            if (bias) {
              $_state.addOperands({input1, input2, weight, bias});
            } else {
              $_state.addOperands({input1, input2, weight});
            }

            // Get shapes
            const std::vector<std::int64_t> input1_shape = getShape(input1);
            const std::vector<std::int64_t> input2_shape = getShape(input2);
            const std::vector<std::int64_t> weight_shape = getShape(weight);

            // Verify shapes
            ERROR_ON(input1_shape.empty());
            ERROR_ON(input2_shape.empty());
            ERROR_ON_MSG(weight_shape.size() != 3 ||
                             weight_shape[1] != input1_shape.back() ||
                             weight_shape[2] != input2_shape.back(),
                         "Bilinear: weight should be a 3D tensor of "
                         "(out_features, in1_features, in2_features).");

            const std::vector<std::int64_t> in1_shared_dims(
                input1_shape.begin(), input1_shape.end() - 1);
            const std::vector<std::int64_t> in2_shared_dims(
                input2_shape.begin(), input2_shape.end() - 1);

            ERROR_ON_MSG(in1_shared_dims != in2_shared_dims,
                         "Bilinear: input shapes should match in all but the "
                         "last dimension.");

            // Compute output shape
            // Is same as input shapes, except last dimension is same as first
            // dimension of weight.
            std::vector<std::int64_t> out_shape = in1_shared_dims;
            out_shape.push_back(weight_shape[0]);

            $_state.addTypes(mlir::RankedTensorType::get(out_shape, getElementType(input1)));
        }]>
    ];
}
