// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 PopParse.py to regenerate
// clang-format off

torch::jit::Node* createGroupnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t num_groups,float epsilon) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::groupnormalization, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->i_(c10::Symbol::fromQualString("attr::num_groups"),num_groups);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
return new_node;
}
torch::jit::Node* createSubsample(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::subsample, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createPrinttensor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t print_gradient,const std::string & title) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::printtensor, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::print_gradient"),print_gradient);
new_node->s_(c10::Symbol::fromQualString("attr::title"),title);
return new_node;
}
torch::jit::Node* createNop(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::nop, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createScale(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float scale) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::scale, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::scale"),scale);
return new_node;
}
torch::jit::Node* createScaledadd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float scale0,float scale1) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::scaledadd, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::scale0"),scale0);
new_node->f_(c10::Symbol::fromQualString("attr::scale1"),scale1);
return new_node;
}
torch::jit::Node* createLstm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t outputFullSequence) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lstm, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted, 2);
new_node->i_(c10::Symbol::fromQualString("attr::outputFullSequence"),outputFullSequence);
return new_node;
}
torch::jit::Node* createGelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::gelu, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createDetach(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::detach, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createDepthtospace(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize,const std::string & mode) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::depthtospace, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
return new_node;
}
torch::jit::Node* createRound(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::round, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createDynamicslice(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes,std::int32_t noOverlap) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dynamicslice, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->is_(c10::Symbol::fromQualString("attr::sizes"),sizes);
new_node->i_(c10::Symbol::fromQualString("attr::noOverlap"),noOverlap);
return new_node;
}
torch::jit::Node* createDynamicupdate(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes,std::int32_t noOverlap) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dynamicupdate, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->is_(c10::Symbol::fromQualString("attr::sizes"),sizes);
new_node->i_(c10::Symbol::fromQualString("attr::noOverlap"),noOverlap);
return new_node;
}
torch::jit::Node* createDynamiczero(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dynamiczero, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->is_(c10::Symbol::fromQualString("attr::sizes"),sizes);
return new_node;
}
torch::jit::Node* createDynamicadd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,std::vector<int64_t> sizes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dynamicadd, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->is_(c10::Symbol::fromQualString("attr::sizes"),sizes);
return new_node;
}
torch::jit::Node* createSequenceslice(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t zeroUnused) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sequenceslice, args, ImplicitCast::None, OutputType::Unknown);
new_node->i_(c10::Symbol::fromQualString("attr::zeroUnused"),zeroUnused);
return new_node;
}
torch::jit::Node* createReplicatedallreduce(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::replicatedallreduce, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createL1loss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const float lambda,std::int32_t reduction) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::l1loss, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::lambda"),lambda);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
return new_node;
}
torch::jit::Node* createNllloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction,std::int32_t ignoreIndex,bool inputIsLogProbability) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::nllloss, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
new_node->i_(c10::Symbol::fromQualString("attr::ignoreIndex"),ignoreIndex);
new_node->i_(c10::Symbol::fromQualString("attr::inputIsLogProbability"),inputIsLogProbability);
return new_node;
}
torch::jit::Node* createIdentityloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::identityloss, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
return new_node;
}
torch::jit::Node* create_ctcloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction,const unsigned int blank,const std::string & outDataType) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::_ctcloss, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
new_node->i_(c10::Symbol::fromQualString("attr::blank"),blank);
new_node->s_(c10::Symbol::fromQualString("attr::outDataType"),outDataType);
return new_node;
}
torch::jit::Node* createShapeddropout(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & shape,float ratio) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::shapeddropout, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::shape"),shape);
new_node->f_(c10::Symbol::fromQualString("attr::ratio"),ratio);
return new_node;
}
torch::jit::Node* createAtan2(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::atan2, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createExpm1(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::expm1, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createLog1p(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::log1p, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createFmod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::fmod, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createRemainder(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::remainder, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createReverse(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dimensions) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reverse, args, ImplicitCast::None, OutputType::Unknown);
new_node->is_(c10::Symbol::fromQualString("attr::dimensions"),dimensions);
return new_node;
}
torch::jit::Node* createBitwisenot(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::bitwisenot, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createBitwiseand(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::bitwiseand, args, ImplicitCast::None, OutputType::Unknown);
return new_node;
}
torch::jit::Node* createBitwiseor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::bitwiseor, args, ImplicitCast::None, OutputType::Unknown);
return new_node;
}
torch::jit::Node* createBitwisexor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::bitwisexor, args, ImplicitCast::None, OutputType::Unknown);
return new_node;
}
torch::jit::Node* createBitwisexnor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::bitwisexnor, args, ImplicitCast::None, OutputType::Unknown);
return new_node;
}
torch::jit::Node* createReducemedian(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducemedian, args, ImplicitCast::None, OutputType::FirstAsFirstInputSecondAlwaysInt, 2);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createAveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t ceil_mode,int64_t count_include_pad,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::averagepool, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->i_(c10::Symbol::fromQualString("attr::ceil_mode"),ceil_mode);
new_node->i_(c10::Symbol::fromQualString("attr::count_include_pad"),count_include_pad);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createConvinteger(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::convinteger, args, ImplicitCast::None, OutputType::AlwaysInt);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createDequantizelinear(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dequantizelinear, args, ImplicitCast::ExceptSecond, OutputType::AlwaysFloat);
return new_node;
}
torch::jit::Node* createDropout(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float ratio) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dropout, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->f_(c10::Symbol::fromQualString("attr::ratio"),ratio);
return new_node;
}
torch::jit::Node* createIsinf(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t detect_negative,int64_t detect_positive) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::isinf, args, ImplicitCast::None, OutputType::AlwaysBool);
new_node->i_(c10::Symbol::fromQualString("attr::detect_negative"),detect_negative);
new_node->i_(c10::Symbol::fromQualString("attr::detect_positive"),detect_positive);
return new_node;
}
torch::jit::Node* createMatmulinteger(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::matmulinteger, args, ImplicitCast::None, OutputType::AlwaysInt);
return new_node;
}
torch::jit::Node* createMaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,const std::vector<int64_t> & kernel_shape,int64_t ceil_mode,const std::vector<int64_t> & dilations,const std::vector<int64_t> & pads,int64_t storage_order,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::maxpool, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->i_(c10::Symbol::fromQualString("attr::ceil_mode"),ceil_mode);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->i_(c10::Symbol::fromQualString("attr::storage_order"),storage_order);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createMod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t fmod) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::mod, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->i_(c10::Symbol::fromQualString("attr::fmod"),fmod);
return new_node;
}
torch::jit::Node* createNonmaxsuppression(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t center_point_box) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::nonmaxsuppression, args, ImplicitCast::None, OutputType::AlwaysInt);
new_node->i_(c10::Symbol::fromQualString("attr::center_point_box"),center_point_box);
return new_node;
}
torch::jit::Node* createQlinearconv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::qlinearconv, args, ImplicitCast::None, OutputType::AlwaysUint8);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createQlinearmatmul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::qlinearmatmul, args, ImplicitCast::None, OutputType::AlwaysUint8);
return new_node;
}
torch::jit::Node* createQuantizelinear(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::quantizelinear, args, ImplicitCast::None, OutputType::AlwaysUint8);
return new_node;
}
torch::jit::Node* createResize(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::resize, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
return new_node;
}
torch::jit::Node* createReversesequence(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t batch_axis,int64_t time_axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reversesequence, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::batch_axis"),batch_axis);
new_node->i_(c10::Symbol::fromQualString("attr::time_axis"),time_axis);
return new_node;
}
torch::jit::Node* createRoialign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode,int64_t output_height,int64_t output_width,int64_t sampling_ratio,float spatial_scale) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::roialign, args, ImplicitCast::ExceptThird, OutputType::AsImplicitCastPromoted);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
new_node->i_(c10::Symbol::fromQualString("attr::output_height"),output_height);
new_node->i_(c10::Symbol::fromQualString("attr::output_width"),output_width);
new_node->i_(c10::Symbol::fromQualString("attr::sampling_ratio"),sampling_ratio);
new_node->f_(c10::Symbol::fromQualString("attr::spatial_scale"),spatial_scale);
return new_node;
}
torch::jit::Node* createSlice(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::slice, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createThresholdedrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::thresholdedrelu, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return new_node;
}
torch::jit::Node* createTopk(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::topk, args, ImplicitCast::None, OutputType::FirstAsFirstInputSecondAlwaysInt, 2);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createUpsample(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::upsample, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
return new_node;
}
torch::jit::Node* createAcosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::acosh, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createAsinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::asinh, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createAtanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::atanh, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createBatchnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float epsilon,float momentum) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::batchnormalization, args, ImplicitCast::ExceptFifth, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
new_node->f_(c10::Symbol::fromQualString("attr::momentum"),momentum);
return new_node;
}
torch::jit::Node* createCast(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & to) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::cast, args, ImplicitCast::None, OutputType::AsDtype);
new_node->s_(c10::Symbol::fromQualString("attr::to"),to);
setNodeOutputsTypes(new_node, ImplicitCast::None, OutputType::AsDtype);
return new_node;
}
torch::jit::Node* createCompress(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::compress, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createCosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::cosh, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createErf(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::erf, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createEyelike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,int64_t k) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::eyelike, args, ImplicitCast::None, OutputType::AsDtype);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->i_(c10::Symbol::fromQualString("attr::k"),k);
setNodeOutputsTypes(new_node, ImplicitCast::None, OutputType::AsDtype);
return new_node;
}
torch::jit::Node* createFlatten(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::flatten, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createGemm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta,int64_t transA,int64_t transB) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::gemm, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
new_node->i_(c10::Symbol::fromQualString("attr::transA"),transA);
new_node->i_(c10::Symbol::fromQualString("attr::transB"),transB);
return new_node;
}
torch::jit::Node* createGreater(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::greater, args, ImplicitCast::All, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createIsnan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::isnan, args, ImplicitCast::None, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createLess(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::less, args, ImplicitCast::All, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createMatmul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::matmul, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createMaxunpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::maxunpool, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createMeanvariancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::meanvariancenormalization, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return new_node;
}
torch::jit::Node* createNonzero(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::nonzero, args, ImplicitCast::None, OutputType::AlwaysInt);
return new_node;
}
torch::jit::Node* createOnehot(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::onehot, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createPrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::prelu, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createScatter(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::scatter, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createShrink(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float bias,float lambd) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::shrink, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::bias"),bias);
new_node->f_(c10::Symbol::fromQualString("attr::lambd"),lambd);
return new_node;
}
torch::jit::Node* createSign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sign, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createSinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sinh, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createTfidfvectorizer(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t max_gram_length,int64_t max_skip_count,int64_t min_gram_length,const std::string & mode,const std::vector<int64_t> & ngram_counts,const std::vector<int64_t> & ngram_indexes,const std::vector<int64_t> & pool_int64s,const std::vector<std::string> & pool_strings,std::vector<double> weights) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::tfidfvectorizer, args, ImplicitCast::None, OutputType::AlwaysFloat);
new_node->i_(c10::Symbol::fromQualString("attr::max_gram_length"),max_gram_length);
new_node->i_(c10::Symbol::fromQualString("attr::max_skip_count"),max_skip_count);
new_node->i_(c10::Symbol::fromQualString("attr::min_gram_length"),min_gram_length);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
new_node->is_(c10::Symbol::fromQualString("attr::ngram_counts"),ngram_counts);
new_node->is_(c10::Symbol::fromQualString("attr::ngram_indexes"),ngram_indexes);
new_node->is_(c10::Symbol::fromQualString("attr::pool_int64s"),pool_int64s);
new_node->ss_(c10::Symbol::fromQualString("attr::pool_strings"),pool_strings);
new_node->fs_(c10::Symbol::fromQualString("attr::weights"),weights);
return new_node;
}
torch::jit::Node* createWhere(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::where, args, ImplicitCast::ExceptFirst, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createExpand(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::expand, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createMax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::max, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createMean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::mean, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createMin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::min, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createSum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sum, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createAcos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::acos, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createAdd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::add, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createLogical_and(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_and, args, ImplicitCast::All, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createAsin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::asin, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createAtan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::atan, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createCos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::cos, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createDiv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::div, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createEqual(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::equal, args, ImplicitCast::All, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createGru(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,std::vector<double> activation_alpha,std::vector<double> activation_beta,const std::vector<std::string> & activations,float clip,const std::string & direction,std::int32_t hidden_size,int64_t linear_before_reset) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::gru, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted, 2);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->fs_(c10::Symbol::fromQualString("attr::activation_alpha"),activation_alpha);
new_node->fs_(c10::Symbol::fromQualString("attr::activation_beta"),activation_beta);
new_node->ss_(c10::Symbol::fromQualString("attr::activations"),activations);
new_node->f_(c10::Symbol::fromQualString("attr::clip"),clip);
new_node->s_(c10::Symbol::fromQualString("attr::direction"),direction);
new_node->i_(c10::Symbol::fromQualString("attr::hidden_size"),hidden_size);
new_node->i_(c10::Symbol::fromQualString("attr::linear_before_reset"),linear_before_reset);
return new_node;
}
torch::jit::Node* createMul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::mul, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createMultinomial(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t dtype,int64_t sample_size,float seed) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::multinomial, args, ImplicitCast::None, OutputType::AsDtype);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->i_(c10::Symbol::fromQualString("attr::sample_size"),sample_size);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
setNodeOutputsTypes(new_node, ImplicitCast::None, OutputType::AsDtype);
return new_node;
}
torch::jit::Node* createLogical_or(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_or, args, ImplicitCast::All, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createPow(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::pow, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createRnn(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,std::vector<double> activation_alpha,std::vector<double> activation_beta,const std::vector<std::string> & activations,float clip,const std::string & direction,std::int32_t hidden_size) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::rnn, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted, 2);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->fs_(c10::Symbol::fromQualString("attr::activation_alpha"),activation_alpha);
new_node->fs_(c10::Symbol::fromQualString("attr::activation_beta"),activation_beta);
new_node->ss_(c10::Symbol::fromQualString("attr::activations"),activations);
new_node->f_(c10::Symbol::fromQualString("attr::clip"),clip);
new_node->s_(c10::Symbol::fromQualString("attr::direction"),direction);
new_node->i_(c10::Symbol::fromQualString("attr::hidden_size"),hidden_size);
return new_node;
}
torch::jit::Node* createSin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sin, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createSub(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sub, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
return new_node;
}
torch::jit::Node* createTan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::tan, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createLogical_xor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_xor, args, ImplicitCast::All, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createAbs(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::abs, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createArgmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::argmax, args, ImplicitCast::None, OutputType::AlwaysInt);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createArgmin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::argmin, args, ImplicitCast::None, OutputType::AlwaysInt);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createCeil(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::ceil, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createClip(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float max,float min) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::clip, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->f_(c10::Symbol::fromQualString("attr::max"),max);
new_node->f_(c10::Symbol::fromQualString("attr::min"),min);
return new_node;
}
torch::jit::Node* createConcat(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::concat, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createConv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::conv, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createConvtranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & output_padding,const std::vector<int64_t> & output_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::convtranspose, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::output_padding"),output_padding);
new_node->is_(c10::Symbol::fromQualString("attr::output_shape"),output_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createElu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::elu, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return new_node;
}
torch::jit::Node* createExp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::exp, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createFloor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::floor, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createGather(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::gather, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createGlobalaveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::globalaveragepool, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createGloballppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t p) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::globallppool, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
return new_node;
}
torch::jit::Node* createGlobalmaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::globalmaxpool, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createHardsigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::hardsigmoid, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
return new_node;
}
torch::jit::Node* createHardmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::hardmax, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createIdentity(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::identity, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createInstancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float epsilon) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::instancenormalization, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
return new_node;
}
torch::jit::Node* createLrn(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t size,float alpha,float beta,float bias) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lrn, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::size"),size);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
new_node->f_(c10::Symbol::fromQualString("attr::bias"),bias);
return new_node;
}
torch::jit::Node* createLeakyrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::leakyrelu, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return new_node;
}
torch::jit::Node* createLog(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::log, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createLogsoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logsoftmax, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createLpnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t p) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lpnormalization, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
return new_node;
}
torch::jit::Node* createLppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t p,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lppool, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createMaxroipool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pooled_shape,float spatial_scale) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::maxroipool, args, ImplicitCast::All, OutputType::AsImplicitCastPromoted);
new_node->is_(c10::Symbol::fromQualString("attr::pooled_shape"),pooled_shape);
new_node->f_(c10::Symbol::fromQualString("attr::spatial_scale"),spatial_scale);
return new_node;
}
torch::jit::Node* createNeg(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::neg, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createLogical_not(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_not, args, ImplicitCast::None, OutputType::AlwaysBool);
return new_node;
}
torch::jit::Node* createPad(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pads,const std::string & mode,float value) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::pad, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
new_node->f_(c10::Symbol::fromQualString("attr::value"),value);
return new_node;
}
torch::jit::Node* createRandomnormallike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float mean,float scale,float seed) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::randomnormallike, args, ImplicitCast::None, OutputType::AsDtypeOrAsPromoted);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->f_(c10::Symbol::fromQualString("attr::mean"),mean);
new_node->f_(c10::Symbol::fromQualString("attr::scale"),scale);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
setNodeOutputsTypes(new_node, ImplicitCast::All, OutputType::AsDtypeOrAsPromoted);
return new_node;
}
torch::jit::Node* createRandomuniformlike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float high,float low,float seed) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::randomuniformlike, args, ImplicitCast::None, OutputType::AsDtypeOrAsPromoted);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->f_(c10::Symbol::fromQualString("attr::high"),high);
new_node->f_(c10::Symbol::fromQualString("attr::low"),low);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
setNodeOutputsTypes(new_node, ImplicitCast::All, OutputType::AsDtypeOrAsPromoted);
return new_node;
}
torch::jit::Node* createReciprocal(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reciprocal, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createReducel1(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducel1, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducel2(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducel2, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducelogsum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducelogsum, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducelogsumexp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducelogsumexp, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducemax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducemax, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducemean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducemean, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducemin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducemin, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReduceprod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reduceprod, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducesum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducesum, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducesumsquare(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducesumsquare, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createRelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::relu, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createSelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float gamma) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::selu, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::gamma"),gamma);
return new_node;
}
torch::jit::Node* createShape(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::shape, args, ImplicitCast::None, OutputType::AlwaysInt);
return new_node;
}
torch::jit::Node* createSigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sigmoid, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createSize(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::size, args, ImplicitCast::None, OutputType::AlwaysInt);
return new_node;
}
torch::jit::Node* createSoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::softmax, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createSoftplus(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::softplus, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createSoftsign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::softsign, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createSpacetodepth(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::spacetodepth, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
return new_node;
}
torch::jit::Node* createSplit(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,int64_t axis,const std::vector<int64_t> & split) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::split, args, ImplicitCast::None, OutputType::AsFirstInput, num_outputs);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->is_(c10::Symbol::fromQualString("attr::split"),split);
return new_node;
}
torch::jit::Node* createSqrt(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sqrt, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createSqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::squeeze, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return new_node;
}
torch::jit::Node* createTanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::tanh, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createTile(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::tile, args, ImplicitCast::None, OutputType::AsFirstInput);
return new_node;
}
torch::jit::Node* createTranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & perm) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::transpose, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::perm"),perm);
return new_node;
}
torch::jit::Node* createUnsqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::unsqueeze, args, ImplicitCast::None, OutputType::AsFirstInput);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return new_node;
}

