// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 PopParse.py to regenerate
// clang-format off

torch::jit::Node* createGroupnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t num_groups,float epsilon) {
 torch::jit::Node *new_node = graph->create(symbols::popart::groupnormalization, args);
new_node->i_(c10::Symbol::fromQualString("attr::num_groups"),num_groups);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSubsample(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = graph->create(symbols::popart::subsample, args);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createPrinttensor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t print_gradient) {
 torch::jit::Node *new_node = graph->create(symbols::popart::printtensor, args);
new_node->i_(c10::Symbol::fromQualString("attr::print_gradient"),print_gradient);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createScale(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float scale) {
 torch::jit::Node *new_node = graph->create(symbols::popart::scale, args);
new_node->f_(c10::Symbol::fromQualString("attr::scale"),scale);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLstm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t outputFullSequence) {
 torch::jit::Node *new_node = graph->create(symbols::popart::lstm, args, 2);
new_node->i_(c10::Symbol::fromQualString("attr::outputFullSequence"),outputFullSequence);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createGelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::gelu, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createDetach(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,bool pass_through_creation) {
 torch::jit::Node *new_node = graph->create(symbols::popart::detach, args);
new_node->i_(c10::Symbol::fromQualString("attr::pass_through_creation"),pass_through_creation);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReplicatedallreduce(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::replicatedallreduce, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createL1loss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const float lamda,std::int32_t reduction) {
 torch::jit::Node *new_node = graph->create(symbols::popart::l1loss, args);
new_node->f_(c10::Symbol::fromQualString("attr::lamda"),lamda);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createNllloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction,std::int32_t ignoreIndex) {
 torch::jit::Node *new_node = graph->create(symbols::popart::nllloss, args);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
new_node->i_(c10::Symbol::fromQualString("attr::ignoreIndex"),ignoreIndex);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createIdentityloss(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction) {
 torch::jit::Node *new_node = graph->create(symbols::popart::identityloss, args);
new_node->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAcosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::acosh, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAsinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::asinh, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAtanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::atanh, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createBatchnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float epsilon,float momentum) {
 torch::jit::Node *new_node = graph->create(symbols::popart::batchnormalization, args);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
new_node->f_(c10::Symbol::fromQualString("attr::momentum"),momentum);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createCompress(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::compress, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createCosh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::cosh, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createErf(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::erf, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createEyelike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,int64_t k) {
 torch::jit::Node *new_node = graph->create(symbols::popart::eyelike, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->i_(c10::Symbol::fromQualString("attr::k"),k);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createFlatten(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::flatten, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createGemm(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta,int64_t transA,int64_t transB) {
 torch::jit::Node *new_node = graph->create(symbols::popart::gemm, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
new_node->i_(c10::Symbol::fromQualString("attr::transA"),transA);
new_node->i_(c10::Symbol::fromQualString("attr::transB"),transB);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createGreater(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::greater, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createIsnan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::isnan, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLess(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::less, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMatmul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::matmul, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMaxunpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = graph->create(symbols::popart::maxunpool, args);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMeanvariancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = graph->create(symbols::popart::meanvariancenormalization, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createNonzero(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::nonzero, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createOnehot(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::onehot, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createPrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::prelu, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createScatter(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::scatter, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createShrink(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float bias,float lambd) {
 torch::jit::Node *new_node = graph->create(symbols::popart::shrink, args);
new_node->f_(c10::Symbol::fromQualString("attr::bias"),bias);
new_node->f_(c10::Symbol::fromQualString("attr::lambd"),lambd);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::sign, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSinh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::sinh, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createWhere(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::where, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createExpand(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::expand, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::max, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,int64_t storage_order,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = graph->create(symbols::popart::maxpool, args);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->i_(c10::Symbol::fromQualString("attr::storage_order"),storage_order);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::mean, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::min, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::sum, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAcos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::acos, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAdd(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::add, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLogical_and(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::logical_and, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAsin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::asin, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAtan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::atan, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t count_include_pad,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = graph->create(symbols::popart::averagepool, args);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->i_(c10::Symbol::fromQualString("attr::count_include_pad"),count_include_pad);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createCos(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::cos, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createDiv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::div, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createDropout(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float ratio) {
 torch::jit::Node *new_node = graph->create(symbols::popart::dropout, args);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->f_(c10::Symbol::fromQualString("attr::ratio"),ratio);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createEqual(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::equal, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMul(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::mul, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMultinomial(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t dtype,int64_t sample_size,float seed) {
 torch::jit::Node *new_node = graph->create(symbols::popart::multinomial, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->i_(c10::Symbol::fromQualString("attr::sample_size"),sample_size);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLogical_or(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::logical_or, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createPow(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::pow, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::sin, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSub(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::sub, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createTan(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::tan, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLogical_xor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::logical_xor, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createAbs(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::abs, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createArgmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::argmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createArgmin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::argmin, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createCeil(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::ceil, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createClip(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float max,float min) {
 torch::jit::Node *new_node = graph->create(symbols::popart::clip, args);
new_node->f_(c10::Symbol::fromQualString("attr::max"),max);
new_node->f_(c10::Symbol::fromQualString("attr::min"),min);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createConcat(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::concat, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createConv(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = graph->create(symbols::popart::conv, args);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createConvtranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & output_padding,const std::vector<int64_t> & output_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = graph->create(symbols::popart::convtranspose, args);
new_node->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
new_node->i_(c10::Symbol::fromQualString("attr::group"),group);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->is_(c10::Symbol::fromQualString("attr::output_padding"),output_padding);
new_node->is_(c10::Symbol::fromQualString("attr::output_shape"),output_shape);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createDepthtospace(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize) {
 torch::jit::Node *new_node = graph->create(symbols::popart::depthtospace, args);
new_node->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createElu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = graph->create(symbols::popart::elu, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createExp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::exp, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createFloor(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::floor, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createGather(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::gather, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createGlobalaveragepool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::globalaveragepool, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createGloballppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t p) {
 torch::jit::Node *new_node = graph->create(symbols::popart::globallppool, args);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createGlobalmaxpool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::globalmaxpool, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createHardsigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta) {
 torch::jit::Node *new_node = graph->create(symbols::popart::hardsigmoid, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createHardmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::hardmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createIdentity(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::identity, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createInstancenormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float epsilon) {
 torch::jit::Node *new_node = graph->create(symbols::popart::instancenormalization, args);
new_node->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLrn(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t size,float alpha,float beta,float bias) {
 torch::jit::Node *new_node = graph->create(symbols::popart::lrn, args);
new_node->i_(c10::Symbol::fromQualString("attr::size"),size);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::beta"),beta);
new_node->f_(c10::Symbol::fromQualString("attr::bias"),bias);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLeakyrelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *new_node = graph->create(symbols::popart::leakyrelu, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLog(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::log, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLogsoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::logsoftmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLpnormalization(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t p) {
 torch::jit::Node *new_node = graph->create(symbols::popart::lpnormalization, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLppool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t p,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *new_node = graph->create(symbols::popart::lppool, args);
new_node->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
new_node->i_(c10::Symbol::fromQualString("attr::p"),p);
new_node->is_(c10::Symbol::fromQualString("attr::pads"),pads);
new_node->is_(c10::Symbol::fromQualString("attr::strides"),strides);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createMaxroipool(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pooled_shape,float spatial_scale) {
 torch::jit::Node *new_node = graph->create(symbols::popart::maxroipool, args);
new_node->is_(c10::Symbol::fromQualString("attr::pooled_shape"),pooled_shape);
new_node->f_(c10::Symbol::fromQualString("attr::spatial_scale"),spatial_scale);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createNeg(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::neg, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createLogical_not(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::logical_not, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createRandomnormallike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float mean,float scale,float seed) {
 torch::jit::Node *new_node = graph->create(symbols::popart::randomnormallike, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->f_(c10::Symbol::fromQualString("attr::mean"),mean);
new_node->f_(c10::Symbol::fromQualString("attr::scale"),scale);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createRandomuniformlike(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float high,float low,float seed) {
 torch::jit::Node *new_node = graph->create(symbols::popart::randomuniformlike, args);
new_node->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
new_node->f_(c10::Symbol::fromQualString("attr::high"),high);
new_node->f_(c10::Symbol::fromQualString("attr::low"),low);
new_node->f_(c10::Symbol::fromQualString("attr::seed"),seed);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReciprocal(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reciprocal, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducel1(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducel1, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducel2(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducel2, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducelogsum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducelogsum, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducelogsumexp(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducelogsumexp, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducemax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducemax, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducemean(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducemean, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducemin(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducemin, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReduceprod(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reduceprod, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducesum(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducesum, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReducesumsquare(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reducesumsquare, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
new_node->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createRelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::relu, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createReshape(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::reshape, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSelu(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,float alpha,float gamma) {
 torch::jit::Node *new_node = graph->create(symbols::popart::selu, args);
new_node->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
new_node->f_(c10::Symbol::fromQualString("attr::gamma"),gamma);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createShape(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::shape, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSigmoid(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::sigmoid, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSize(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::size, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSlice(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & ends,const std::vector<int64_t> & starts,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = graph->create(symbols::popart::slice, args);
new_node->is_(c10::Symbol::fromQualString("attr::ends"),ends);
new_node->is_(c10::Symbol::fromQualString("attr::starts"),starts);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSoftmax(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::softmax, args);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSoftplus(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::softplus, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSoftsign(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::softsign, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSpacetodepth(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize) {
 torch::jit::Node *new_node = graph->create(symbols::popart::spacetodepth, args);
new_node->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSplit(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,int64_t axis,const std::vector<int64_t> & split) {
 torch::jit::Node *new_node = graph->create(symbols::popart::split, args, num_outputs);
new_node->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
new_node->is_(c10::Symbol::fromQualString("attr::split"),split);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSqrt(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::sqrt, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createSqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = graph->create(symbols::popart::squeeze, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createTanh(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::tanh, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createTile(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *new_node = graph->create(symbols::popart::tile, args);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createTopk(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,int64_t k,int64_t axis) {
 torch::jit::Node *new_node = graph->create(symbols::popart::topk, args, 2);
new_node->i_(c10::Symbol::fromQualString("attr::k"),k);
new_node->i_(c10::Symbol::fromQualString("attr::axis"),axis);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createTranspose(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & perm) {
 torch::jit::Node *new_node = graph->create(symbols::popart::transpose, args);
new_node->is_(c10::Symbol::fromQualString("attr::perm"),perm);
graph->insertNode(new_node);
return new_node;
}
torch::jit::Node* createUnsqueeze(torch::jit::Graph *graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *new_node = graph->create(symbols::popart::unsqueeze, args);
new_node->is_(c10::Symbol::fromQualString("attr::axes"),axes);
graph->insertNode(new_node);
return new_node;
}

