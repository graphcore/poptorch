// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef INCLUDE_POPTORCH_ELIMINATE_LIST_CONSTRUCTS_H
#define INCLUDE_POPTORCH_ELIMINATE_LIST_CONSTRUCTS_H

// Take the following graph:
//     graph(%x.1 : Float(2, 3, 4)):
//           %_output_size.1 : int[] = prim::ListConstruct()
//           %64 : int = prim::Constant[value=5]()
//           %57 : int[] = aten::append(%_output_size.1, %64)
//           %66 : int = prim::Constant[value=7]()
//           %61 : int[] = aten::append(%_output_size.1, %66)
//           %28 : Tensor = aten::adaptive_avg_pool2d(%x.1, %_output_size.1)
//           return (%28)
//
// This should be able to be reduced to:
//     graph(%x.1 : Float(2, 3, 4)):
//           %_output_size.1 : int[] = prim::Constant[value=[5, 7]]()
//           %28 : Tensor = aten::adaptive_avg_pool2d(%x.1, %_output_size.1)
//           return (%28)

namespace torch {
namespace jit {
class Graph;
} // namespace jit
} // namespace torch

namespace poptorch {

void eliminateListConstructs(torch::jit::Graph *graph);

} // namespace poptorch

#endif
