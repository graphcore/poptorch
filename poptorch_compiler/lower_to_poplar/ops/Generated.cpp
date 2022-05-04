// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "dialect/PoptorchDialect.hpp"

namespace poptorch_ir {
/*
  Eventually this file should be empty, as we implement new operations we should
  take the function from here and move it into the correct folder.
 */

/*
 * Generated from the below macro. To rerun:
 *     type a random character into this file
 *     ninja lower_to_poplar
 *     get the command line which failed and add "-E"
 *     copy from the output.
 */
/*
#define INT_VEC
#define FLOAT_VEC
#define FLOAT
#define INT
#define BOOL
#define STRING
#define STRING_VEC
#define NONE
#define ARG(Type, Name)
#define BODY_ARG(Name)

#define OP_DECL(name, args, arg_names)                                         \
  void name::lowerToPoplar(CompilerContext &context) {
 (void)this;                         \
    assert(assert(false && "Function: " #name " is currently unimplemented.");
\); } // namespace poptorch_ir

#include "pytorch_bridge/helpers/default_ops.h.inc"

#undef OP_DECL
#undef NONE
#undef STRING_VEC
#undef STRING
#undef BOOL
#undef INT
#undef FLOAT
#undef FLOAT_VEC
#undef INT_VEC
#undef BODY_ARG
#undef ARG*/

void groupnormalization::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "groupnormalization"
                  " is currently unimplemented.");
}
void subsample::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "subsample"
                  " is currently unimplemented.");
}

void lstm::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "lstm"
                  " is currently unimplemented.");
}
void depthtospace::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "depthtospace"
                  " is currently unimplemented.");
}
void dynamicslice::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "dynamicslice"
                  " is currently unimplemented.");
}
void dynamicupdate::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "dynamicupdate"
                  " is currently unimplemented.");
}
void dynamiczero::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "dynamiczero"
                  " is currently unimplemented.");
}
void dynamicadd::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "dynamicadd"
                  " is currently unimplemented.");
}
void sequenceslice::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "sequenceslice"
                  " is currently unimplemented.");
}
void replicatedallreduce::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "replicatedallreduce"
                  " is currently unimplemented.");
}
void l1loss::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "l1loss"
                  " is currently unimplemented.");
}
void _ctcloss::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "_ctcloss"
                  " is currently unimplemented.");
}
void ctcbeamsearchdecoder::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "ctcbeamsearchdecoder"
                  " is currently unimplemented.");
}
void shapeddropout::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "shapeddropout"
                  " is currently unimplemented.");
}
void reverse::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reverse"
                  " is currently unimplemented.");
}
void reducemedian::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducemedian"
                  " is currently unimplemented.");
}
void scatterreduce::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "scatterreduce"
                  " is currently unimplemented.");
}
void averagepool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "averagepool"
                  " is currently unimplemented.");
}
void convinteger::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "convinteger"
                  " is currently unimplemented.");
}
void dequantizelinear::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "dequantizelinear"
                  " is currently unimplemented.");
}
void isinf::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "isinf"
                  " is currently unimplemented.");
}
void matmulinteger::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "matmulinteger"
                  " is currently unimplemented.");
}
void maxpool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "maxpool"
                  " is currently unimplemented.");
}
void mod::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "mod"
                  " is currently unimplemented.");
}
void nonmaxsuppression::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "nonmaxsuppression"
                  " is currently unimplemented.");
}
void qlinearconv::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "qlinearconv"
                  " is currently unimplemented.");
}
void qlinearmatmul::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "qlinearmatmul"
                  " is currently unimplemented.");
}
void quantizelinear::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "quantizelinear"
                  " is currently unimplemented.");
}
void resize::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "resize"
                  " is currently unimplemented.");
}
void reversesequence::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reversesequence"
                  " is currently unimplemented.");
}
void roialign::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "roialign"
                  " is currently unimplemented.");
}
void slice::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "slice"
                  " is currently unimplemented.");
}
void thresholdedrelu::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "thresholdedrelu"
                  " is currently unimplemented.");
}
void upsample::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "upsample"
                  " is currently unimplemented.");
}
void acosh::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "acosh"
                  " is currently unimplemented.");
}
void asinh::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "asinh"
                  " is currently unimplemented.");
}
void atanh::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "atanh"
                  " is currently unimplemented.");
}
void compress::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "compress"
                  " is currently unimplemented.");
}
void cosh::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "cosh"
                  " is currently unimplemented.");
}
void eyelike::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "eyelike"
                  " is currently unimplemented.");
}
void flatten::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "flatten"
                  " is currently unimplemented.");
}
void gemm::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "gemm"
                  " is currently unimplemented.");
}
void maxunpool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "maxunpool"
                  " is currently unimplemented.");
}
void meanvariancenormalization::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "meanvariancenormalization"
                  " is currently unimplemented.");
}
void nonzero::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "nonzero"
                  " is currently unimplemented.");
}
void onehot::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "onehot"
                  " is currently unimplemented.");
}
void scatter::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "scatter"
                  " is currently unimplemented.");
}
void shrink::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "shrink"
                  " is currently unimplemented.");
}
void sign::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "sign"
                  " is currently unimplemented.");
}
void sinh::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "sinh"
                  " is currently unimplemented.");
}
void tfidfvectorizer::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "tfidfvectorizer"
                  " is currently unimplemented.");
}
void mean::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "mean"
                  " is currently unimplemented.");
}
void sum::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "sum"
                  " is currently unimplemented.");
}
void acos::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "acos"
                  " is currently unimplemented.");
}
void atan::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "atan"
                  " is currently unimplemented.");
}
void multinomial::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "multinomial"
                  " is currently unimplemented.");
}
void rnn::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "rnn"
                  " is currently unimplemented.");
}
void logical_xor::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "logical_xor"
                  " is currently unimplemented.");
}
void clip::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "clip"
                  " is currently unimplemented.");
}
void convtranspose::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "convtranspose"
                  " is currently unimplemented.");
}
void gather::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "gather"
                  " is currently unimplemented.");
}
void globalaveragepool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "globalaveragepool"
                  " is currently unimplemented.");
}
void globallppool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "globallppool"
                  " is currently unimplemented.");
}
void globalmaxpool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "globalmaxpool"
                  " is currently unimplemented.");
}
void hardmax::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "hardmax"
                  " is currently unimplemented.");
}
void identity::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "identity"
                  " is currently unimplemented.");
}
void instancenormalization::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "instancenormalization"
                  " is currently unimplemented.");
}
void lrn::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "lrn"
                  " is currently unimplemented.");
}
void lpnormalization::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "lpnormalization"
                  " is currently unimplemented.");
}
void lppool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "lppool"
                  " is currently unimplemented.");
}
void maxroipool::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "maxroipool"
                  " is currently unimplemented.");
}
void pad::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "pad"
                  " is currently unimplemented.");
}
void randomnormallike::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "randomnormallike"
                  " is currently unimplemented.");
}
void randomuniformlike::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "randomuniformlike"
                  " is currently unimplemented.");
}
void reciprocal::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reciprocal"
                  " is currently unimplemented.");
}
void reducel1::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducel1"
                  " is currently unimplemented.");
}
void reducel2::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducel2"
                  " is currently unimplemented.");
}
void reducelogsum::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducelogsum"
                  " is currently unimplemented.");
}
void reducelogsumexp::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducelogsumexp"
                  " is currently unimplemented.");
}
void reducemax::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducemax"
                  " is currently unimplemented.");
}
void reducemin::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducemin"
                  " is currently unimplemented.");
}
void reducesumsquare::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "reducesumsquare"
                  " is currently unimplemented.");
}

void selu::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "selu"
                  " is currently unimplemented.");
}
void shape::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "shape"
                  " is currently unimplemented.");
}
void size::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "size"
                  " is currently unimplemented.");
}
void softsign::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "softsign"
                  " is currently unimplemented.");
}
void spacetodepth::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "spacetodepth"
                  " is currently unimplemented.");
}
void squeeze::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "squeeze"
                  " is currently unimplemented.");
}
void tile::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "tile"
                  " is currently unimplemented.");
}
void ones::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "ones"
                  " is currently unimplemented.");
}
void zeros::lowerToPoplar(CompilerContext & /*context*/) {
  (void)this;
  assert(false && "Function: "
                  "zeros"
                  " is currently unimplemented.");
}

} // namespace poptorch_ir
