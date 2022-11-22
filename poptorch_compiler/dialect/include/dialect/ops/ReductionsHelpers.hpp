// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include "mlir/IR/BuiltinTypes.h"

namespace poptorch_ir {

llvm::SmallVector<std::int64_t, 4>
inferShapeOfReductionOutput(std::vector<std::int64_t> dims,
                            llvm::ArrayRef<std::int64_t> in_shape,
                            bool keepdim);

std::vector<std::int64_t>
parseReductionDimArgument(const std::vector<std::int64_t> &input_dim,
                          std::size_t input_shape_size);
} // namespace poptorch_ir
