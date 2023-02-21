// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 scripts/PopParse.py` to regenerate
// clang-format off

// Ops from AiGraphcoreOpset1
OP_DECL(popart, copyvarupdate, copyvarupdate, AiGraphcoreOpset1.copyvarupdate, NONE, BODY_ARG(DEBUG_CONTEXT("Copyvarupdate")))
OP_DECL(popart, batchnormalization, batchnormalization, AiGraphcoreOpset1.batchnormalization, ARG(INT,num_outputs) ARG(FLOAT,epsilon) ARG(FLOAT,momentum) , BODY_ARG(num_outputs) BODY_ARG(epsilon) BODY_ARG(momentum) BODY_ARG(DEBUG_CONTEXT("Batchnormalization")))
OP_DECL(popart, groupnormalization, groupnormalization, AiGraphcoreOpset1.groupnormalization, ARG(INT,num_groups) ARG(FLOAT,epsilon) , BODY_ARG(num_groups) BODY_ARG(epsilon) BODY_ARG(DEBUG_CONTEXT("Groupnormalization")))
OP_DECL(popart, subsample, subsample, AiGraphcoreOpset1.subsample, ARG(INT_VEC,strides) , BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Subsample")))
OP_DECL(popart, printtensor, printtensor, AiGraphcoreOpset1.printtensor, ARG(INT,print_gradient) ARG(STRING,title) ARG(INT,summariseThreshold) ARG(INT,edgeItems) ARG(INT,maxLineWidth) ARG(INT,digits) ARG(INT,floatFormat) ARG(CHAR,separator) ARG(CHAR,openBracket) ARG(CHAR,closeBracket) , BODY_ARG(print_gradient) BODY_ARG(DEBUG_CONTEXT("Printtensor"))BODY_ARG(title) BODY_ARG(summariseThreshold) BODY_ARG(edgeItems) BODY_ARG(maxLineWidth) BODY_ARG(digits) BODY_ARG(floatFormat) BODY_ARG(separator) BODY_ARG(openBracket) BODY_ARG(closeBracket) )
OP_DECL(popart, nop, nop, AiGraphcoreOpset1.nop, NONE, BODY_ARG(DEBUG_CONTEXT("Nop")))
OP_DECL(popart, scale, scale, AiGraphcoreOpset1.scale, ARG(FLOAT,scale) , BODY_ARG(scale) BODY_ARG(DEBUG_CONTEXT("Scale")))
OP_DECL(popart, scaledadd, scaledadd, AiGraphcoreOpset1.scaledadd, ARG(FLOAT,scale0) ARG(FLOAT,scale1) , BODY_ARG(scale0) BODY_ARG(scale1) BODY_ARG(DEBUG_CONTEXT("Scaledadd")))
OP_DECL(popart, lstm, lstm, AiGraphcoreOpset1.lstm, ARG(INT,outputFullSequence) , BODY_ARG(outputFullSequence) BODY_ARG(DEBUG_CONTEXT("Lstm")))
OP_DECL(popart, gelu, gelu, AiGraphcoreOpset1.gelu, NONE, BODY_ARG(DEBUG_CONTEXT("Gelu")))
OP_DECL(popart, detach, detach, AiGraphcoreOpset1.detach, NONE, BODY_ARG(DEBUG_CONTEXT("Detach")))
OP_DECL(popart, depthtospace, depthtospace, AiGraphcoreOpset1.depthtospace, ARG(INT,blocksize) ARG(STRING,mode) , BODY_ARG(blocksize) BODY_ARG(mode) BODY_ARG(DEBUG_CONTEXT("Depthtospace")))
OP_DECL(popart, round, round, AiGraphcoreOpset1.round, NONE, BODY_ARG(DEBUG_CONTEXT("Round")))
OP_DECL(popart, dynamicslice, dynamicslice, AiGraphcoreOpset1.dynamicslice, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ARG(INT,noOverlap) , BODY_ARG(axes) BODY_ARG(sizes) BODY_ARG(noOverlap) BODY_ARG(DEBUG_CONTEXT("Dynamicslice")))
OP_DECL(popart, dynamicupdate, dynamicupdate, AiGraphcoreOpset1.dynamicupdate, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) ARG(INT,noOverlap) , BODY_ARG(axes) BODY_ARG(sizes) BODY_ARG(noOverlap) BODY_ARG(DEBUG_CONTEXT("Dynamicupdate")))
OP_DECL(popart, dynamiczero, dynamiczero, AiGraphcoreOpset1.dynamiczero, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) , BODY_ARG(axes) BODY_ARG(sizes) BODY_ARG(DEBUG_CONTEXT("Dynamiczero")))
OP_DECL(popart, dynamicadd, dynamicadd, AiGraphcoreOpset1.dynamicadd, ARG(INT_VEC,axes) ARG(INT_VEC,sizes) , BODY_ARG(axes) BODY_ARG(sizes) BODY_ARG(DEBUG_CONTEXT("Dynamicadd")))
OP_DECL(popart, sequenceslice, sequenceslice, AiGraphcoreOpset1.sequenceslice, ARG(INT,zeroUnused) , BODY_ARG(zeroUnused) BODY_ARG(DEBUG_CONTEXT("Sequenceslice")))
OP_DECL(popart, l1loss, l1loss, AiGraphcoreOpset1.l1loss, ARG(FLOAT,lambda) ARG(INT,reduction) , BODY_ARG(lambda) BODY_ARG(static_cast<popart::ReductionType>(reduction)) BODY_ARG(DEBUG_CONTEXT("L1loss")))
OP_DECL(popart, nllloss, nllloss, AiGraphcoreOpset1.nllloss, ARG(INT,reduction) ARG(INT,ignoreIndex) ARG(INT,inputIsLogProbability) , BODY_ARG(static_cast<popart::ReductionType>(reduction)) BODY_ARG(ignoreIndex) BODY_ARG(inputIsLogProbability) BODY_ARG(DEBUG_CONTEXT("Nllloss")))
OP_DECL(popart, identityloss, identityloss, AiGraphcoreOpset1.identityloss, ARG(INT,reduction) , BODY_ARG(static_cast<popart::ReductionType>(reduction)) BODY_ARG(DEBUG_CONTEXT("Identityloss")))
OP_DECL(popart, _ctcloss, _ctcloss, AiGraphcoreOpset1._ctcloss, ARG(INT,reduction) ARG(INT,blank) ARG(STRING,outDataType) ARG(INT,zeroInfinity) , BODY_ARG(static_cast<popart::ReductionType>(reduction)) BODY_ARG(blank) BODY_ARG(outDataType) BODY_ARG(zeroInfinity) BODY_ARG(DEBUG_CONTEXT("_ctcloss")))
OP_DECL(popart, ctcbeamsearchdecoder, ctcbeamsearchdecoder, AiGraphcoreOpset1.ctcbeamsearchdecoder, ARG(INT,blank) ARG(INT,beamWidth) ARG(INT,topPaths) , BODY_ARG(blank) BODY_ARG(beamWidth) BODY_ARG(topPaths) BODY_ARG(DEBUG_CONTEXT("Ctcbeamsearchdecoder")))
OP_DECL(popart, shapeddropout, shapeddropout, AiGraphcoreOpset1.shapeddropout, ARG(INT_VEC,shape) ARG(FLOAT,ratio) , BODY_ARG(shape) BODY_ARG(ratio) BODY_ARG(DEBUG_CONTEXT("Shapeddropout")))
OP_DECL(popart, atan2, atan2, AiGraphcoreOpset1.atan2, NONE, BODY_ARG(DEBUG_CONTEXT("Atan2")))
OP_DECL(popart, expm1, expm1, AiGraphcoreOpset1.expm1, NONE, BODY_ARG(DEBUG_CONTEXT("Expm1")))
OP_DECL(popart, log1p, log1p, AiGraphcoreOpset1.log1p, NONE, BODY_ARG(DEBUG_CONTEXT("Log1p")))
OP_DECL(popart, fmod, fmod, AiGraphcoreOpset1.fmod, NONE, BODY_ARG(DEBUG_CONTEXT("Fmod")))
OP_DECL(popart, remainder, remainder, AiGraphcoreOpset1.remainder, NONE, BODY_ARG(DEBUG_CONTEXT("Remainder")))
OP_DECL(popart, reverse, reverse, AiGraphcoreOpset1.reverse, ARG(INT_VEC,dimensions) , BODY_ARG(dimensions) BODY_ARG(DEBUG_CONTEXT("Reverse")))
OP_DECL(popart, slice, slice, AiGraphcoreOpset1.slice, ARG(INT_VEC,ends) ARG(INT_VEC,starts) ARG(INT_VEC,axes) , BODY_ARG(ends) BODY_ARG(starts) BODY_ARG(axes) BODY_ARG(DEBUG_CONTEXT("Slice")))
OP_DECL(popart, bitwisenot, bitwisenot, AiGraphcoreOpset1.bitwisenot, NONE, BODY_ARG(DEBUG_CONTEXT("Bitwisenot")))
OP_DECL(popart, bitwiseand, bitwiseand, AiGraphcoreOpset1.bitwiseand, NONE, BODY_ARG(DEBUG_CONTEXT("Bitwiseand")))
OP_DECL(popart, bitwiseor, bitwiseor, AiGraphcoreOpset1.bitwiseor, NONE, BODY_ARG(DEBUG_CONTEXT("Bitwiseor")))
OP_DECL(popart, bitwisexor, bitwisexor, AiGraphcoreOpset1.bitwisexor, NONE, BODY_ARG(DEBUG_CONTEXT("Bitwisexor")))
OP_DECL(popart, bitwisexnor, bitwisexnor, AiGraphcoreOpset1.bitwisexnor, NONE, BODY_ARG(DEBUG_CONTEXT("Bitwisexnor")))
OP_DECL(popart, reducemedian, reducemedian, AiGraphcoreOpset1.reducemedian, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducemedian")))
OP_DECL(popart, scatterreduce, scatterreduce, AiGraphcoreOpset1.scatterreduce, ARG(INT,axis_size) ARG(INT,axis) ARG(INT,reduction) ARG(INT, enable_index_broadcast), BODY_ARG(axis_size) BODY_ARG(axis) BODY_ARG(static_cast<popart::ScatterReduction>(reduction)) BODY_ARG(enable_index_broadcast) BODY_ARG(DEBUG_CONTEXT("Scatterreduce")))
OP_DECL(popart, groupedscatterreduce, groupedscatterreduce, AiGraphcoreOpset1.groupedscatterreduce, ARG(INT,axis_size) ARG(INT,axis) ARG(INT,reduction) ARG(INT,group_size) ARG(INT, enable_index_broadcast), BODY_ARG(axis_size) BODY_ARG(axis) BODY_ARG(static_cast<popart::ScatterReduction>(reduction)) BODY_ARG(group_size) BODY_ARG(enable_index_broadcast) BODY_ARG(DEBUG_CONTEXT("Scatterreduce")))
OP_DECL(popart, swish, swish, AiGraphcoreOpset1.swish, NONE, BODY_ARG(DEBUG_CONTEXT("Swish")))
// Ops from AiOnnxOpset10
OP_DECL(popart, averagepool, averagepool, AiOnnxOpset10.averagepool, ARG(INT_VEC,kernel_shape) ARG(INT,ceil_mode) ARG(INT,count_include_pad) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(kernel_shape) BODY_ARG(ceil_mode) BODY_ARG(count_include_pad) BODY_ARG(pads) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Averagepool")))
OP_DECL(popart, convinteger, convinteger, AiOnnxOpset10.convinteger, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Convinteger")))
OP_DECL(popart, dequantizelinear, dequantizelinear, AiOnnxOpset10.dequantizelinear, NONE, BODY_ARG(DEBUG_CONTEXT("Dequantizelinear")))
OP_DECL(popart, dropout, dropout, AiOnnxOpset10.dropout, ARG(INT,num_outputs) ARG(FLOAT,ratio) , BODY_ARG(num_outputs) BODY_ARG(ratio) BODY_ARG(DEBUG_CONTEXT("Dropout")))
OP_DECL(popart, isinf, isinf, AiOnnxOpset10.isinf, ARG(INT,detect_negative) ARG(INT,detect_positive) , BODY_ARG(detect_negative) BODY_ARG(detect_positive) BODY_ARG(DEBUG_CONTEXT("Isinf")))
OP_DECL(popart, matmulinteger, matmulinteger, AiOnnxOpset10.matmulinteger, NONE, BODY_ARG(DEBUG_CONTEXT("Matmulinteger")))
OP_DECL(popart, maxpool, maxpool, AiOnnxOpset10.maxpool, ARG(INT,num_outputs) ARG(INT_VEC,kernel_shape) ARG(INT,ceil_mode) ARG(INT_VEC,dilations) ARG(INT_VEC,pads) ARG(INT,storage_order) ARG(INT_VEC,strides) , BODY_ARG(num_outputs) BODY_ARG(kernel_shape) BODY_ARG(ceil_mode) BODY_ARG(dilations) BODY_ARG(pads) BODY_ARG(storage_order) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Maxpool")))
OP_DECL(popart, mod, mod, AiOnnxOpset10.mod, ARG(INT,fmod) , BODY_ARG(fmod) BODY_ARG(DEBUG_CONTEXT("Mod")))
OP_DECL(popart, nonmaxsuppression, nonmaxsuppression, AiOnnxOpset10.nonmaxsuppression, ARG(INT,center_point_box) , BODY_ARG(center_point_box) BODY_ARG(DEBUG_CONTEXT("Nonmaxsuppression")))
OP_DECL(popart, qlinearconv, qlinearconv, AiOnnxOpset10.qlinearconv, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Qlinearconv")))
OP_DECL(popart, qlinearmatmul, qlinearmatmul, AiOnnxOpset10.qlinearmatmul, NONE, BODY_ARG(DEBUG_CONTEXT("Qlinearmatmul")))
OP_DECL(popart, quantizelinear, quantizelinear, AiOnnxOpset10.quantizelinear, NONE, BODY_ARG(DEBUG_CONTEXT("Quantizelinear")))
OP_DECL(popart, resize, resize, AiOnnxOpset10.resize, ARG(STRING,mode) , BODY_ARG(mode) BODY_ARG(DEBUG_CONTEXT("Resize")))
OP_DECL(popart, reversesequence, reversesequence, AiOnnxOpset10.reversesequence, ARG(INT,batch_axis) ARG(INT,time_axis) , BODY_ARG(batch_axis) BODY_ARG(time_axis) BODY_ARG(DEBUG_CONTEXT("Reversesequence")))
OP_DECL(popart, roialign, roialign, AiOnnxOpset10.roialign, ARG(STRING,mode) ARG(INT,output_height) ARG(INT,output_width) ARG(INT,sampling_ratio) ARG(FLOAT,spatial_scale) , BODY_ARG(mode) BODY_ARG(output_height) BODY_ARG(output_width) BODY_ARG(sampling_ratio) BODY_ARG(spatial_scale) BODY_ARG(DEBUG_CONTEXT("Roialign")))
OP_DECL(popart, thresholdedrelu, thresholdedrelu, AiOnnxOpset10.thresholdedrelu, ARG(FLOAT,alpha) , BODY_ARG(alpha) BODY_ARG(DEBUG_CONTEXT("Thresholdedrelu")))
OP_DECL(popart, topk, topk, AiOnnxOpset10.topk, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Topk")))
OP_DECL(popart, upsample, upsample, AiOnnxOpset10.upsample, ARG(STRING,mode) , BODY_ARG(mode) BODY_ARG(DEBUG_CONTEXT("Upsample")))
// Ops from AiOnnxOpset9
OP_DECL(popart, acosh, acosh, AiOnnxOpset10.acosh, NONE, BODY_ARG(DEBUG_CONTEXT("Acosh")))
OP_DECL(popart, asinh, asinh, AiOnnxOpset10.asinh, NONE, BODY_ARG(DEBUG_CONTEXT("Asinh")))
OP_DECL(popart, atanh, atanh, AiOnnxOpset10.atanh, NONE, BODY_ARG(DEBUG_CONTEXT("Atanh")))
OP_DECL(popart, cast, cast, AiOnnxOpset10.cast, ARG(STRING,to) , BODY_ARG(to) BODY_ARG(DEBUG_CONTEXT("Cast")))
OP_DECL(popart, compress, compress, AiOnnxOpset10.compress, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Compress")))
OP_DECL(popart, cosh, cosh, AiOnnxOpset10.cosh, NONE, BODY_ARG(DEBUG_CONTEXT("Cosh")))
OP_DECL(popart, erf, erf, AiOnnxOpset10.erf, NONE, BODY_ARG(DEBUG_CONTEXT("Erf")))
OP_DECL(popart, eyelike, eyelike, AiOnnxOpset10.eyelike, ARG(INT,dtype) ARG(INT,k) , BODY_ARG(dtype) BODY_ARG(k) BODY_ARG(DEBUG_CONTEXT("Eyelike")))
OP_DECL(popart, flatten, flatten, AiOnnxOpset10.flatten, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Flatten")))
OP_DECL(popart, gemm, gemm, AiOnnxOpset10.gemm, ARG(FLOAT,alpha) ARG(FLOAT,beta) ARG(INT,transA) ARG(INT,transB) , BODY_ARG(alpha) BODY_ARG(beta) BODY_ARG(transA) BODY_ARG(transB) BODY_ARG(DEBUG_CONTEXT("Gemm")))
OP_DECL(popart, greater, greater, AiOnnxOpset10.greater, NONE, BODY_ARG(DEBUG_CONTEXT("Greater")))
OP_DECL(popart, isnan, isnan, AiOnnxOpset10.isnan, NONE, BODY_ARG(DEBUG_CONTEXT("Isnan")))
OP_DECL(popart, less, less, AiOnnxOpset10.less, NONE, BODY_ARG(DEBUG_CONTEXT("Less")))
OP_DECL(popart, matmul, matmul, AiOnnxOpset10.matmul, NONE, BODY_ARG(DEBUG_CONTEXT("Matmul")))
OP_DECL(popart, maxunpool, maxunpool, AiOnnxOpset10.maxunpool, ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Maxunpool")))
OP_DECL(popart, meanvariancenormalization, meanvariancenormalization, AiOnnxOpset10.meanvariancenormalization, ARG(INT_VEC,axes) , BODY_ARG(axes) BODY_ARG(DEBUG_CONTEXT("Meanvariancenormalization")))
OP_DECL(popart, nonzero, nonzero, AiOnnxOpset10.nonzero, NONE, BODY_ARG(DEBUG_CONTEXT("Nonzero")))
OP_DECL(popart, onehot, onehot, AiOnnxOpset10.onehot, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Onehot")))
OP_DECL(popart, scatter, scatter, AiOnnxOpset10.scatter, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Scatter")))
OP_DECL(popart, shrink, shrink, AiOnnxOpset10.shrink, ARG(FLOAT,bias) ARG(FLOAT,lambd) , BODY_ARG(bias) BODY_ARG(lambd) BODY_ARG(DEBUG_CONTEXT("Shrink")))
OP_DECL(popart, sign, sign, AiOnnxOpset10.sign, NONE, BODY_ARG(DEBUG_CONTEXT("Sign")))
OP_DECL(popart, sinh, sinh, AiOnnxOpset10.sinh, NONE, BODY_ARG(DEBUG_CONTEXT("Sinh")))
OP_DECL(popart, tfidfvectorizer, tfidfvectorizer, AiOnnxOpset10.tfidfvectorizer, ARG(INT,max_gram_length) ARG(INT,max_skip_count) ARG(INT,min_gram_length) ARG(STRING,mode) ARG(INT_VEC,ngram_counts) ARG(INT_VEC,ngram_indexes) ARG(INT_VEC,pool_int64s) ARG(STRING_VEC,pool_strings) ARG(FLOAT_VEC,weights) , BODY_ARG(max_gram_length) BODY_ARG(max_skip_count) BODY_ARG(min_gram_length) BODY_ARG(mode) BODY_ARG(ngram_counts) BODY_ARG(ngram_indexes) BODY_ARG(pool_int64s) BODY_ARG(pool_strings) BODY_ARG(weights) BODY_ARG(DEBUG_CONTEXT("Tfidfvectorizer")))
OP_DECL(popart, where, where, AiOnnxOpset10.where, NONE, BODY_ARG(DEBUG_CONTEXT("Where")))
// Ops from AiOnnxOpset8
OP_DECL(popart, expand, expand, AiOnnxOpset10.expand, NONE, BODY_ARG(DEBUG_CONTEXT("Expand")))
OP_DECL(popart, max, max, AiOnnxOpset10.max, NONE, BODY_ARG(DEBUG_CONTEXT("Max")))
OP_DECL(popart, mean, mean, AiOnnxOpset10.mean, NONE, BODY_ARG(DEBUG_CONTEXT("Mean")))
OP_DECL(popart, min, min, AiOnnxOpset10.min, NONE, BODY_ARG(DEBUG_CONTEXT("Min")))
OP_DECL(popart, sum, sum, AiOnnxOpset10.sum, NONE, BODY_ARG(DEBUG_CONTEXT("Sum")))
// Ops from AiOnnxOpset7
OP_DECL(popart, acos, acos, AiOnnxOpset10.acos, NONE, BODY_ARG(DEBUG_CONTEXT("Acos")))
OP_DECL(popart, add, add, AiOnnxOpset10.add, NONE, BODY_ARG(DEBUG_CONTEXT("Add")))
OP_DECL(popart, logical_and, logical_and, AiOnnxOpset10.logical_and, NONE, BODY_ARG(DEBUG_CONTEXT("Logical_and")))
OP_DECL(popart, asin, asin, AiOnnxOpset10.asin, NONE, BODY_ARG(DEBUG_CONTEXT("Asin")))
OP_DECL(popart, atan, atan, AiOnnxOpset10.atan, NONE, BODY_ARG(DEBUG_CONTEXT("Atan")))
OP_DECL(popart, cos, cos, AiOnnxOpset10.cos, NONE, BODY_ARG(DEBUG_CONTEXT("Cos")))
OP_DECL(popart, div, div, AiOnnxOpset10.div, NONE, BODY_ARG(DEBUG_CONTEXT("Div")))
OP_DECL(popart, equal, equal, AiOnnxOpset10.equal, NONE, BODY_ARG(DEBUG_CONTEXT("Equal")))
OP_DECL(popart, mul, mul, AiOnnxOpset10.mul, NONE, BODY_ARG(DEBUG_CONTEXT("Mul")))
OP_DECL(popart, multinomial, multinomial, AiOnnxOpset10.multinomial, ARG(INT,dtype) ARG(INT,sample_size) ARG(FLOAT,seed) , BODY_ARG(dtype) BODY_ARG(sample_size) BODY_ARG(seed) BODY_ARG(DEBUG_CONTEXT("Multinomial")))
OP_DECL(popart, logical_or, logical_or, AiOnnxOpset10.logical_or, NONE, BODY_ARG(DEBUG_CONTEXT("Logical_or")))
OP_DECL(popart, pow, pow, AiOnnxOpset10.pow, NONE, BODY_ARG(DEBUG_CONTEXT("Pow")))
OP_DECL(popart, sin, sin, AiOnnxOpset10.sin, NONE, BODY_ARG(DEBUG_CONTEXT("Sin")))
OP_DECL(popart, sub, sub, AiOnnxOpset10.sub, NONE, BODY_ARG(DEBUG_CONTEXT("Sub")))
OP_DECL(popart, tan, tan, AiOnnxOpset10.tan, NONE, BODY_ARG(DEBUG_CONTEXT("Tan")))
OP_DECL(popart, logical_xor, logical_xor, AiOnnxOpset10.logical_xor, NONE, BODY_ARG(DEBUG_CONTEXT("Logical_xor")))
// Ops from AiOnnxOpset6
OP_DECL(popart, abs, abs, AiOnnxOpset10.abs, NONE, BODY_ARG(DEBUG_CONTEXT("Abs")))
OP_DECL(popart, argmax, argmax, AiOnnxOpset10.argmax, ARG(INT,axis) ARG(INT,keepdims) , BODY_ARG(axis) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Argmax")))
OP_DECL(popart, argmin, argmin, AiOnnxOpset10.argmin, ARG(INT,axis) ARG(INT,keepdims) , BODY_ARG(axis) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Argmin")))
OP_DECL(popart, ceil, ceil, AiOnnxOpset10.ceil, NONE, BODY_ARG(DEBUG_CONTEXT("Ceil")))
OP_DECL(popart, clip, clip, AiOnnxOpset10.clip, ARG(FLOAT,max) ARG(FLOAT,min) , BODY_ARG(max) BODY_ARG(min) BODY_ARG(DEBUG_CONTEXT("Clip")))
OP_DECL(popart, concat, concat, AiOnnxOpset10.concat, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Concat")))
OP_DECL(popart, conv, conv, AiOnnxOpset10.conv, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(pads) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Conv")))
OP_DECL(popart, convtranspose, convtranspose, AiOnnxOpset10.convtranspose, ARG(INT_VEC,dilations) ARG(INT,group) ARG(INT_VEC,kernel_shape) ARG(INT_VEC,output_padding) ARG(INT_VEC,output_shape) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(dilations) BODY_ARG(group) BODY_ARG(kernel_shape) BODY_ARG(output_padding) BODY_ARG(output_shape) BODY_ARG(pads) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Convtranspose")))
OP_DECL(popart, elu, elu, AiOnnxOpset10.elu, ARG(FLOAT,alpha) , BODY_ARG(alpha) BODY_ARG(DEBUG_CONTEXT("Elu")))
OP_DECL(popart, exp, exp, AiOnnxOpset10.exp, NONE, BODY_ARG(DEBUG_CONTEXT("Exp")))
OP_DECL(popart, floor, floor, AiOnnxOpset10.floor, NONE, BODY_ARG(DEBUG_CONTEXT("Floor")))
OP_DECL(popart, gather, gather, AiOnnxOpset10.gather, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Gather")))
OP_DECL(popart, globalaveragepool, globalaveragepool, AiOnnxOpset10.globalaveragepool, NONE, BODY_ARG(DEBUG_CONTEXT("Globalaveragepool")))
OP_DECL(popart, globallppool, globallppool, AiOnnxOpset10.globallppool, ARG(INT,p) , BODY_ARG(p) BODY_ARG(DEBUG_CONTEXT("Globallppool")))
OP_DECL(popart, globalmaxpool, globalmaxpool, AiOnnxOpset10.globalmaxpool, NONE, BODY_ARG(DEBUG_CONTEXT("Globalmaxpool")))
OP_DECL(popart, hardsigmoid, hardsigmoid, AiOnnxOpset10.hardsigmoid, ARG(FLOAT,alpha) ARG(FLOAT,beta) , BODY_ARG(alpha) BODY_ARG(beta) BODY_ARG(DEBUG_CONTEXT("Hardsigmoid")))
OP_DECL(popart, hardmax, hardmax, AiOnnxOpset10.hardmax, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Hardmax")))
OP_DECL(popart, identity, identity, AiOnnxOpset10.identity, NONE, BODY_ARG(DEBUG_CONTEXT("Identity")))
OP_DECL(popart, instancenormalization, instancenormalization, AiOnnxOpset10.instancenormalization, ARG(FLOAT,epsilon) , BODY_ARG(epsilon) BODY_ARG(DEBUG_CONTEXT("Instancenormalization")))
OP_DECL(popart, lrn, lrn, AiOnnxOpset10.lrn, ARG(INT,size) ARG(FLOAT,alpha) ARG(FLOAT,beta) ARG(FLOAT,bias) , BODY_ARG(size) BODY_ARG(alpha) BODY_ARG(beta) BODY_ARG(bias) BODY_ARG(DEBUG_CONTEXT("Lrn")))
OP_DECL(popart, leakyrelu, leakyrelu, AiOnnxOpset10.leakyrelu, ARG(FLOAT,alpha) , BODY_ARG(alpha) BODY_ARG(DEBUG_CONTEXT("Leakyrelu")))
OP_DECL(popart, log, log, AiOnnxOpset10.log, NONE, BODY_ARG(DEBUG_CONTEXT("Log")))
OP_DECL(popart, logsoftmax, logsoftmax, AiOnnxOpset10.logsoftmax, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Logsoftmax")))
OP_DECL(popart, lpnormalization, lpnormalization, AiOnnxOpset10.lpnormalization, ARG(INT,axis) ARG(INT,p) , BODY_ARG(axis) BODY_ARG(p) BODY_ARG(DEBUG_CONTEXT("Lpnormalization")))
OP_DECL(popart, lppool, lppool, AiOnnxOpset10.lppool, ARG(INT_VEC,kernel_shape) ARG(INT,p) ARG(INT_VEC,pads) ARG(INT_VEC,strides) , BODY_ARG(kernel_shape) BODY_ARG(p) BODY_ARG(pads) BODY_ARG(strides) BODY_ARG(DEBUG_CONTEXT("Lppool")))
OP_DECL(popart, maxroipool, maxroipool, AiOnnxOpset10.maxroipool, ARG(INT_VEC,pooled_shape) ARG(FLOAT,spatial_scale) , BODY_ARG(pooled_shape) BODY_ARG(spatial_scale) BODY_ARG(DEBUG_CONTEXT("Maxroipool")))
OP_DECL(popart, neg, neg, AiOnnxOpset10.neg, NONE, BODY_ARG(DEBUG_CONTEXT("Neg")))
OP_DECL(popart, logical_not, logical_not, AiOnnxOpset10.logical_not, NONE, BODY_ARG(DEBUG_CONTEXT("Logical_not")))
OP_DECL(popart, pad, pad, AiOnnxOpset10.pad, ARG(INT_VEC,pads) ARG(STRING,mode) ARG(FLOAT,value) , BODY_ARG(pads) BODY_ARG(mode) BODY_ARG(value) BODY_ARG(DEBUG_CONTEXT("Pad")))
OP_DECL(popart, randomnormallike, randomnormallike, AiOnnxOpset10.randomnormallike, ARG(INT,dtype) ARG(FLOAT,mean) ARG(FLOAT,scale) ARG(FLOAT,seed) , BODY_ARG(dtype) BODY_ARG(mean) BODY_ARG(scale) BODY_ARG(seed) BODY_ARG(DEBUG_CONTEXT("Randomnormallike")))
OP_DECL(popart, randomuniformlike, randomuniformlike, AiOnnxOpset10.randomuniformlike, ARG(INT,dtype) ARG(FLOAT,high) ARG(FLOAT,low) ARG(FLOAT,seed) , BODY_ARG(dtype) BODY_ARG(high) BODY_ARG(low) BODY_ARG(seed) BODY_ARG(DEBUG_CONTEXT("Randomuniformlike")))
OP_DECL(popart, reciprocal, reciprocal, AiOnnxOpset10.reciprocal, NONE, BODY_ARG(DEBUG_CONTEXT("Reciprocal")))
OP_DECL(popart, reducel1, reducel1, AiOnnxOpset10.reducel1, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducel1")))
OP_DECL(popart, reducel2, reducel2, AiOnnxOpset10.reducel2, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducel2")))
OP_DECL(popart, reducelogsum, reducelogsum, AiOnnxOpset10.reducelogsum, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducelogsum")))
OP_DECL(popart, reducelogsumexp, reducelogsumexp, AiOnnxOpset10.reducelogsumexp, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducelogsumexp")))
OP_DECL(popart, reducemax, reducemax, AiOnnxOpset10.reducemax, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducemax")))
OP_DECL(popart, reducemean, reducemean, AiOnnxOpset10.reducemean, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducemean")))
OP_DECL(popart, reducemin, reducemin, AiOnnxOpset10.reducemin, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducemin")))
OP_DECL(popart, reduceprod, reduceprod, AiOnnxOpset10.reduceprod, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reduceprod")))
OP_DECL(popart, reducesum, reducesum, AiOnnxOpset10.reducesum, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducesum")))
OP_DECL(popart, reducesumsquare, reducesumsquare, AiOnnxOpset10.reducesumsquare, ARG(INT_VEC,axes) ARG(INT,keepdims) , BODY_ARG(axes) BODY_ARG(keepdims) BODY_ARG(DEBUG_CONTEXT("Reducesumsquare")))
OP_DECL(popart, relu, relu, AiOnnxOpset10.relu, NONE, BODY_ARG(DEBUG_CONTEXT("Relu")))
OP_DECL(popart, selu, selu, AiOnnxOpset10.selu, ARG(FLOAT,alpha) ARG(FLOAT,gamma) , BODY_ARG(alpha) BODY_ARG(gamma) BODY_ARG(DEBUG_CONTEXT("Selu")))
OP_DECL(popart, shape, shape, AiOnnxOpset10.shape, NONE, BODY_ARG(DEBUG_CONTEXT("Shape")))
OP_DECL(popart, sigmoid, sigmoid, AiOnnxOpset10.sigmoid, NONE, BODY_ARG(DEBUG_CONTEXT("Sigmoid")))
OP_DECL(popart, size, size, AiOnnxOpset10.size, NONE, BODY_ARG(DEBUG_CONTEXT("Size")))
OP_DECL(popart, softmax, softmax, AiOnnxOpset10.softmax, ARG(INT,axis) , BODY_ARG(axis) BODY_ARG(DEBUG_CONTEXT("Softmax")))
OP_DECL(popart, softplus, softplus, AiOnnxOpset10.softplus, NONE, BODY_ARG(DEBUG_CONTEXT("Softplus")))
OP_DECL(popart, softsign, softsign, AiOnnxOpset10.softsign, NONE, BODY_ARG(DEBUG_CONTEXT("Softsign")))
OP_DECL(popart, spacetodepth, spacetodepth, AiOnnxOpset10.spacetodepth, ARG(INT,blocksize) , BODY_ARG(blocksize) BODY_ARG(DEBUG_CONTEXT("Spacetodepth")))
OP_DECL(popart, split, split, AiOnnxOpset10.split, ARG(INT,num_outputs) ARG(INT,axis) ARG(INT_VEC,split) , BODY_ARG(num_outputs) BODY_ARG(axis) BODY_ARG(split) BODY_ARG(DEBUG_CONTEXT("Split")))
OP_DECL(popart, sqrt, sqrt, AiOnnxOpset10.sqrt, NONE, BODY_ARG(DEBUG_CONTEXT("Sqrt")))
OP_DECL(popart, squeeze, squeeze, AiOnnxOpset10.squeeze, ARG(INT_VEC,axes) , BODY_ARG(axes) BODY_ARG(DEBUG_CONTEXT("Squeeze")))
OP_DECL(popart, tanh, tanh, AiOnnxOpset10.tanh, NONE, BODY_ARG(DEBUG_CONTEXT("Tanh")))
OP_DECL(popart, tile, tile, AiOnnxOpset10.tile, NONE, BODY_ARG(DEBUG_CONTEXT("Tile")))
OP_DECL(popart, transpose, transpose, AiOnnxOpset10.transpose, ARG(INT_VEC,perm) , BODY_ARG(perm) BODY_ARG(DEBUG_CONTEXT("Transpose")))
OP_DECL(popart, unsqueeze, unsqueeze, AiOnnxOpset10.unsqueeze, ARG(INT_VEC,axes) , BODY_ARG(axes) BODY_ARG(DEBUG_CONTEXT("Unsqueeze")))

