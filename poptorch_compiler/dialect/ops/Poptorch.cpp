// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include "dialect/Helpers.hpp"
#include "dialect/Poptorch.hpp"
#include "poptorch_logging/Error.hpp"

namespace poptorch_ir {

#define VERIFY_SYMBOL_TABLE(op_type, symbol_type)                              \
  ::mlir::LogicalResult op_type::verifySymbolUses(                             \
      mlir::SymbolTableCollection &symbolTable) {                              \
    auto global = symbolTable.lookupNearestSymbolFrom<symbol_type>(            \
        *this, this->handleAttr());                                            \
                                                                               \
    ERROR_ON_MSG(!global, "@" << this->handle().str()                          \
                              << " does not reference a valid global symbol"); \
                                                                               \
    const auto tensor_type =                                                   \
        tensor().getType().cast<mlir::RankedTensorType>();                     \
    const auto global_type = global.type().cast<mlir::RankedTensorType>();     \
    ERROR_ON_MSG(global_type.getShape() != tensor_type.getShape(),             \
                 #op_type ": Shape mismatch between the global symbol ("       \
                     << mlirToStr(global_type) << ") and looked-up value ("    \
                     << mlirToStr(tensor_type) << ") in @"                     \
                     << this->handle().str());                                 \
                                                                               \
    ERROR_ON_MSG(                                                              \
        global_type.getElementType() != tensor_type.getElementType(),          \
        #op_type ": Element type mismatch between the global symbol ("         \
            << mlirToStr(global_type.getElementType())                         \
            << ") and looked-up value ("                                       \
            << mlirToStr(tensor_type.getElementType()) << ") in @"             \
            << this->handle().str());                                          \
    return mlir::success();                                                    \
  }

VERIFY_SYMBOL_TABLE(copy_to_global_state, global_tensor_op)
VERIFY_SYMBOL_TABLE(copy_from_global_state, global_tensor_op)

} // namespace poptorch_ir
