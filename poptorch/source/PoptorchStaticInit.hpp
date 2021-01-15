// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef SOURCE_POPTORCH_STATIC_INIT_H
#define SOURCE_POPTORCH_STATIC_INIT_H

// The constants below set priorities for constructor functions used to
// initialize static data. Functions with lower numbers run first.

// Priority value for symbol initialisation functions
#define SYMBOL_INIT_PRIORITY 101

// Priority value for shape inference registration functions
#define SHAPE_INFERENCE_INIT_PRIORITY 102

// Priority value for handler registration functions
#define HANDLER_INIT_PRIORITY 103

#endif
