// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 PopParse.py to regenerate
// clang-format off

torch::jit::Node* createGroupnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t num_groups,float epsilon) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::groupnormalization, args);
new_node->i_(c10::Symbol::fromQualString("attr::num_groups"),num_groups);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
return new_node;
}
torch::jit::Node* createSubsample(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::subsample, args);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createPrinttensor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t print_gradient) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::printtensor, args);
new_node->i_(c10::Symbol::fromQualString("attr::print_gradient"),print_gradient);
return new_node;
}
torch::jit::Node* createScale(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float scale) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::scale, args);
new_node->f_(c10::Symbol::fromQualString("attr::scale"),scale);
return new_node;
}
torch::jit::Node* createLstm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t outputFullSequence) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lstm, args, 2);
new_node->i_(c10::Symbol::fromQualString("attr::outputFullSequence"),outputFullSequence);
return new_node;
}
torch::jit::Node* createGelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::gelu, args);
return new_node;
}
torch::jit::Node* createDetach(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,bool pass_through_creation) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::detach, args);
new_node->i_(c10::Symbol::fromQualString("attr::pass_through_creation"),pass_through_creation);
return new_node;
}
torch::jit::Node* createReplicatedallreduce(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::replicatedallreduce, args);
return new_node;
}
torch::jit::Node* createL1loss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const float lamda,std::int32_t reduction) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::l1loss, args);
new_node->f_(c10::Symbol::fromQualString("attr::lamda"),lamda);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
return new_node;
}
torch::jit::Node* createNllloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction,std::int32_t ignoreIndex) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::nllloss, args);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
new_node->i_(c10::Symbol::fromQualString("attr::ignoreIndex"),ignoreIndex);
return new_node;
}
torch::jit::Node* createIdentityloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::identityloss, args);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
return new_node;
}
torch::jit::Node* createAveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t ceil_mode,int64_t count_include_pad,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::averagepool, args);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->i_(c10::Symbol::fromQualString("attr::ceil_mode"),ceil_mode);
new_node->i_(c10::Symbol::fromQualString("attr::count_include_pad"),count_include_pad);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createConvinteger(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::convinteger, args);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createDequantizelinear(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dequantizelinear, args);
return new_node;
}
torch::jit::Node* createDropout(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float ratio) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::dropout, args);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->f_(c10::Symbol::fromQualString("attr::ratio"),ratio);
return new_node;
}
torch::jit::Node* createIsinf(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t detect_negative,int64_t detect_positive) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::isinf, args);
new_node->i_(c10::Symbol::fromQualString("attr::detect_negative"),detect_negative);
new_node->i_(c10::Symbol::fromQualString("attr::detect_positive"),detect_positive);
return new_node;
}
torch::jit::Node* createMatmulinteger(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::matmulinteger, args);
return new_node;
}
torch::jit::Node* createMaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,const std::vector<int64_t> & kernel_shape,int64_t ceil_mode,const std::vector<int64_t> & dilations,const std::vector<int64_t> & pads,int64_t storage_order,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::maxpool, args);
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
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::mod, args);
new_node->i_(c10::Symbol::fromQualString("attr::fmod"),fmod);
return new_node;
}
torch::jit::Node* createNonmaxsuppression(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t center_point_box) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::nonmaxsuppression, args);
new_node->i_(c10::Symbol::fromQualString("attr::center_point_box"),center_point_box);
return new_node;
}
torch::jit::Node* createQlinearconv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::qlinearconv, args);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createQlinearmatmul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::qlinearmatmul, args);
return new_node;
}
torch::jit::Node* createQuantizelinear(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::quantizelinear, args);
return new_node;
}
torch::jit::Node* createResize(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::resize, args);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
return new_node;
}
torch::jit::Node* createReversesequence(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t batch_axis,int64_t time_axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reversesequence, args);
new_node->i_(c10::Symbol::fromQualString("attr::batch_axis"),batch_axis);
new_node->i_(c10::Symbol::fromQualString("attr::time_axis"),time_axis);
return new_node;
}
torch::jit::Node* createRoialign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode,int64_t output_height,int64_t output_width,int64_t sampling_ratio,float spatial_scale) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::roialign, args);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
new_node->i_(c10::Symbol::fromQualString("attr::output_height"),output_height);
new_node->i_(c10::Symbol::fromQualString("attr::output_width"),output_width);
new_node->i_(c10::Symbol::fromQualString("attr::sampling_ratio"),sampling_ratio);
new_node->f_(c10::Symbol::fromQualString("attr::spatial_scale"),spatial_scale);
return new_node;
}
torch::jit::Node* createSlice(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::slice, args);
return new_node;
}
torch::jit::Node* createThresholdedrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::thresholdedrelu, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return new_node;
}
torch::jit::Node* createTopk(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::topk, args, 2);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createUpsample(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & mode) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::upsample, args);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
return new_node;
}
torch::jit::Node* createAcosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::acosh, args);
return new_node;
}
torch::jit::Node* createAsinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::asinh, args);
return new_node;
}
torch::jit::Node* createAtanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::atanh, args);
return new_node;
}
torch::jit::Node* createBatchnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float epsilon,float momentum) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::batchnormalization, args);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
new_node->f_(c10::Symbol::fromQualString("attr::momentum"),momentum);
return new_node;
}
torch::jit::Node* createCast(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::string & to) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::cast, args);
new_node->s_(c10::Symbol::fromQualString("attr::to"),to);
return new_node;
}
torch::jit::Node* createCompress(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::compress, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createCosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::cosh, args);
return new_node;
}
torch::jit::Node* createErf(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::erf, args);
return new_node;
}
torch::jit::Node* createEyelike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,int64_t k) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::eyelike, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->i_(c10::Symbol::fromQualString("attr::k"),k);
return new_node;
}
torch::jit::Node* createFlatten(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::flatten, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createGemm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta,int64_t transA,int64_t transB) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::gemm, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
new_node->i_(c10::Symbol::fromQualString("attr::transA"),transA);
new_node->i_(c10::Symbol::fromQualString("attr::transB"),transB);
return new_node;
}
torch::jit::Node* createGreater(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::greater, args);
return new_node;
}
torch::jit::Node* createIsnan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::isnan, args);
return new_node;
}
torch::jit::Node* createLess(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::less, args);
return new_node;
}
torch::jit::Node* createMatmul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::matmul, args);
return new_node;
}
torch::jit::Node* createMaxunpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::maxunpool, args);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createMeanvariancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::meanvariancenormalization, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return new_node;
}
torch::jit::Node* createNonzero(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::nonzero, args);
return new_node;
}
torch::jit::Node* createOnehot(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::onehot, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createPrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::prelu, args);
return new_node;
}
torch::jit::Node* createScatter(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::scatter, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createShrink(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float bias,float lambd) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::shrink, args);
new_node->f_(c10::Symbol::fromQualString("attr::bias"),bias);
new_node->f_(c10::Symbol::fromQualString("attr::lambd"),lambd);
return new_node;
}
torch::jit::Node* createSign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sign, args);
return new_node;
}
torch::jit::Node* createSinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sinh, args);
return new_node;
}
torch::jit::Node* createWhere(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::where, args);
return new_node;
}
torch::jit::Node* createExpand(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::expand, args);
return new_node;
}
torch::jit::Node* createMax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::max, args);
return new_node;
}
torch::jit::Node* createMean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::mean, args);
return new_node;
}
torch::jit::Node* createMin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::min, args);
return new_node;
}
torch::jit::Node* createSum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sum, args);
return new_node;
}
torch::jit::Node* createAcos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::acos, args);
return new_node;
}
torch::jit::Node* createAdd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::add, args);
return new_node;
}
torch::jit::Node* createLogical_and(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_and, args);
return new_node;
}
torch::jit::Node* createAsin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::asin, args);
return new_node;
}
torch::jit::Node* createAtan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::atan, args);
return new_node;
}
torch::jit::Node* createCos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::cos, args);
return new_node;
}
torch::jit::Node* createDiv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::div, args);
return new_node;
}
torch::jit::Node* createEqual(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::equal, args);
return new_node;
}
torch::jit::Node* createMul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::mul, args);
return new_node;
}
torch::jit::Node* createMultinomial(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t dtype,int64_t sample_size,float seed) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::multinomial, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->i_(c10::Symbol::fromQualString("attr::sample_size"),sample_size);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
return new_node;
}
torch::jit::Node* createLogical_or(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_or, args);
return new_node;
}
torch::jit::Node* createPow(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::pow, args);
return new_node;
}
torch::jit::Node* createSin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sin, args);
return new_node;
}
torch::jit::Node* createSub(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sub, args);
return new_node;
}
torch::jit::Node* createTan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::tan, args);
return new_node;
}
torch::jit::Node* createLogical_xor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_xor, args);
return new_node;
}
torch::jit::Node* createAbs(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::abs, args);
return new_node;
}
torch::jit::Node* createArgmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::argmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createArgmin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::argmin, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createCeil(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::ceil, args);
return new_node;
}
torch::jit::Node* createClip(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float max,float min) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::clip, args);
new_node->f_(c10::Symbol::fromQualString("attr::max"),max);
new_node->f_(c10::Symbol::fromQualString("attr::min"),min);
return new_node;
}
torch::jit::Node* createConcat(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::concat, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createConv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::conv, args);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createConvtranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & output_padding,const std::vector<int64_t> & output_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::convtranspose, args);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::output_padding"),output_padding);
new_node->is_(c10::Symbol::fromQualString("attr::output_shape"),output_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createDepthtospace(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::depthtospace, args);
new_node->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
return new_node;
}
torch::jit::Node* createElu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::elu, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return new_node;
}
torch::jit::Node* createExp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::exp, args);
return new_node;
}
torch::jit::Node* createFloor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::floor, args);
return new_node;
}
torch::jit::Node* createGather(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::gather, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createGlobalaveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::globalaveragepool, args);
return new_node;
}
torch::jit::Node* createGloballppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t p) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::globallppool, args);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
return new_node;
}
torch::jit::Node* createGlobalmaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::globalmaxpool, args);
return new_node;
}
torch::jit::Node* createHardsigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::hardsigmoid, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
return new_node;
}
torch::jit::Node* createHardmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::hardmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createIdentity(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::identity, args);
return new_node;
}
torch::jit::Node* createInstancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float epsilon) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::instancenormalization, args);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
return new_node;
}
torch::jit::Node* createLrn(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t size,float alpha,float beta,float bias) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lrn, args);
new_node->i_(c10::Symbol::fromQualString("attr::size"),size);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
new_node->f_(c10::Symbol::fromQualString("attr::bias"),bias);
return new_node;
}
torch::jit::Node* createLeakyrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::leakyrelu, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return new_node;
}
torch::jit::Node* createLog(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::log, args);
return new_node;
}
torch::jit::Node* createLogsoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logsoftmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createLpnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t p) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lpnormalization, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
return new_node;
}
torch::jit::Node* createLppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t p,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::lppool, args);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return new_node;
}
torch::jit::Node* createMaxroipool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pooled_shape,float spatial_scale) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::maxroipool, args);
new_node->is_(c10::Symbol::fromQualString("attr::pooled_shape"),pooled_shape);
new_node->f_(c10::Symbol::fromQualString("attr::spatial_scale"),spatial_scale);
return new_node;
}
torch::jit::Node* createNeg(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::neg, args);
return new_node;
}
torch::jit::Node* createLogical_not(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::logical_not, args);
return new_node;
}
torch::jit::Node* createPad(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pads,const std::string & mode,float value) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::pad, args);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->s_(c10::Symbol::fromQualString("attr::mode"),mode);
new_node->f_(c10::Symbol::fromQualString("attr::value"),value);
return new_node;
}
torch::jit::Node* createRandomnormallike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float mean,float scale,float seed) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::randomnormallike, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->f_(c10::Symbol::fromQualString("attr::mean"),mean);
new_node->f_(c10::Symbol::fromQualString("attr::scale"),scale);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
return new_node;
}
torch::jit::Node* createRandomuniformlike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float high,float low,float seed) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::randomuniformlike, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->f_(c10::Symbol::fromQualString("attr::high"),high);
new_node->f_(c10::Symbol::fromQualString("attr::low"),low);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
return new_node;
}
torch::jit::Node* createReciprocal(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reciprocal, args);
return new_node;
}
torch::jit::Node* createReducel1(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducel1, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducel2(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducel2, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducelogsum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducelogsum, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducelogsumexp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducelogsumexp, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducemax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducemax, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducemean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducemean, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducemin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducemin, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReduceprod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reduceprod, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducesum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducesum, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createReducesumsquare(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reducesumsquare, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return new_node;
}
torch::jit::Node* createRelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::relu, args);
return new_node;
}
torch::jit::Node* createReshape(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::reshape, args);
return new_node;
}
torch::jit::Node* createSelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float gamma) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::selu, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::gamma"),gamma);
return new_node;
}
torch::jit::Node* createShape(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::shape, args);
return new_node;
}
torch::jit::Node* createSigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sigmoid, args);
return new_node;
}
torch::jit::Node* createSize(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::size, args);
return new_node;
}
torch::jit::Node* createSoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::softmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return new_node;
}
torch::jit::Node* createSoftplus(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::softplus, args);
return new_node;
}
torch::jit::Node* createSoftsign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::softsign, args);
return new_node;
}
torch::jit::Node* createSpacetodepth(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::spacetodepth, args);
new_node->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
return new_node;
}
torch::jit::Node* createSplit(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,int64_t axis,const std::vector<int64_t> & split) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::split, args, num_outputs);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->is_(c10::Symbol::fromQualString("attr::split"),split);
return new_node;
}
torch::jit::Node* createSqrt(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::sqrt, args);
return new_node;
}
torch::jit::Node* createSqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::squeeze, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return new_node;
}
torch::jit::Node* createTanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::tanh, args);
return new_node;
}
torch::jit::Node* createTile(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::tile, args);
return new_node;
}
torch::jit::Node* createTranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & perm) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::transpose, args);
new_node->is_(c10::Symbol::fromQualString("attr::perm"),perm);
return new_node;
}
torch::jit::Node* createUnsqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = createAndInsertNode(graph, symbols::popart::unsqueeze, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return new_node;
}

