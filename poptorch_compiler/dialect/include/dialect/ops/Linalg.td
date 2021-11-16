/*
 * Linalg operations.
 */

def Poptorch_matmul : Poptorch_Op<"matmul", []> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$v1, "mlir::Value":$v2),[{
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
            mlir::RankedTensorType t1 = v1.getType().cast<mlir::RankedTensorType>();
            mlir::RankedTensorType t2 = v2.getType().cast<mlir::RankedTensorType>();

            // Elem type.
            mlir::Type e1 = t1.getElementType();


            // Technically according to pytorch this case is scalar but we will return 1D tensor of shape {1} instead.
            if (t1.getRank() == 1 && t2.getRank() == 1) {
                return mlir::RankedTensorType::get({1}, e1);
            }

            // Matrix-vector product. [n, m, k, j, v] * [v] == [n, m, k, j]
            if (t1.getRank() == 1 && t2.getRank() > 1) {
                return mlir::RankedTensorType::get(t2.getShape().drop_back(1), e1);
            }


            // Matrix-vector
            if (t1.getRank() > 1 && t2.getRank() == 1) {
                return mlir::RankedTensorType::get(t1.getShape().drop_back(1), e1);
            }


            // Normal matrix-matrix
            if (t1.getRank() == 2 && t2.getRank() == 2) {
                return mlir::RankedTensorType::get({t1.getDimSize(0), t2.getDimSize(1)}, e1);
            }


            llvm::SmallVector<std::int64_t, 4> shape1 {t1.getShape().begin(), t1.getShape().end()};
            llvm::SmallVector<std::int64_t, 4> shape2 { t2.getShape().begin(),  t2.getShape().end()};


            // Batched matrix with broadcast.
            if (t1.getRank() >= 2 && t2.getRank() >= 2) {

                // [Groups, M, K] * [K, N] = [Groups, M, N]
                if (t2.getRank() == 2) {
                    // Swap K and N.
                    std::swap(*(shape2.end()-1), *(shape1.end()-1));
                    return mlir::RankedTensorType::get(shape1, e1);
                }

                // [M, K] * [Groups, K, N] = [Groups, M, N]
                if (t1.getRank() == 2) {
                    // Swap M and K.
                    std::swap(*(shape2.end()-2), *(shape1.end()-2));
                    return mlir::RankedTensorType::get(shape2, e1);
                }

                // Broadcast to match.
                // [1, 5, 4, 3, N, K] * [3, K, M] = [1, 5, 4, 3, N, M]
                if (t2.getRank() > t1.getRank()) {
                    shape1.insert(shape1.begin(), t2.getRank() - t1.getRank(), 1);
                }

                if (t1.getRank() > t2.getRank()) {
                    shape2.insert(shape2.begin(), t1.getRank() - t2.getRank(), 1);
                }


                assert(t1.getRank() == t2.getRank());
                // Broadcast to match.
                for (std::int64_t dim = 0; dim < t1.getRank(); dim++) {
                    shape1[dim] = std::max(shape1[dim], shape2[dim]);
                    shape2[dim] = shape1[dim];
                }
            }


            return mlir::RankedTensorType::get(shape1, e1);

        }
    }];
}
