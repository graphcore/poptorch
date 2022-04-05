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
        if (dim.has_value()) {
            $_state.addAttribute("dim", $_builder.getI64IntegerAttr(*dim));
        }
        $_state.addAttribute("keepdim", $_builder.getBoolAttr(keepdim));

        mlir::RankedTensorType tensor = self.getType().cast<mlir::RankedTensorType>();

        auto getShape = [&]() {
            // We are flattenning the dimenion we're passed
            if (dim.has_value()) {
                auto ref = tensor.getShape();

                // Deal with negative dim
                if (*dim < 0) {
                    *dim += ref.size();
                }

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