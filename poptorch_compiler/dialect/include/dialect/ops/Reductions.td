
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

            shape.erase(std::remove(shape.begin(), shape.end(), 0));
        }

        $_state.addTypes(mlir::RankedTensorType::get(shape, tensor.getElementType()));
     }]>
    ];
}


def Poptorch_reducemean : Poptorch_reduce<"reducemean"> {}
def Poptorch_reducesum : Poptorch_reduce<"reducesum"> {}