// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 PopParse.py to regenerate
// clang-format off

torch::jit::Node* Create_groupnormalization(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t num_groups,float epsilon) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::groupnormalization, args);
newNode->i_(c10::Symbol::fromQualString("attr::num_groups"),num_groups);
newNode->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
return newNode;
}
torch::jit::Node* Create_subsample(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & strides) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::subsample, args);
newNode->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return newNode;
}
torch::jit::Node* Create_printtensor(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t print_gradient) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::printtensor, args);
newNode->i_(c10::Symbol::fromQualString("attr::print_gradient"),print_gradient);
return newNode;
}
torch::jit::Node* Create_scale(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float scale) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::scale, args);
newNode->f_(c10::Symbol::fromQualString("attr::scale"),scale);
return newNode;
}
torch::jit::Node* Create_lstm(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t outputFullSequence) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::lstm, args, 2);
newNode->i_(c10::Symbol::fromQualString("attr::outputFullSequence"),outputFullSequence);
return newNode;
}
torch::jit::Node* Create_gelu(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::gelu, args);
return newNode;
}
torch::jit::Node* Create_detach(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,bool pass_through_creation) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::detach, args);
newNode->i_(c10::Symbol::fromQualString("attr::pass_through_creation"),pass_through_creation);
return newNode;
}
torch::jit::Node* Create_replicatedallreduce(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::replicatedallreduce, args);
return newNode;
}
torch::jit::Node* Create_l1loss(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const float lamda,std::int32_t reduction) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::l1loss, args);
newNode->f_(c10::Symbol::fromQualString("attr::lamda"),lamda);
newNode->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
return newNode;
}
torch::jit::Node* Create_nllloss(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction,std::int32_t ignoreIndex) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::nllloss, args);
newNode->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
newNode->i_(c10::Symbol::fromQualString("attr::ignoreIndex"),ignoreIndex);
return newNode;
}
torch::jit::Node* Create_identityloss(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::int32_t reduction) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::identityloss, args);
newNode->i_(c10::Symbol::fromQualString("attr::reduction"),reduction);
return newNode;
}
torch::jit::Node* Create_acosh(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::acosh, args);
return newNode;
}
torch::jit::Node* Create_asinh(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::asinh, args);
return newNode;
}
torch::jit::Node* Create_atanh(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::atanh, args);
return newNode;
}
torch::jit::Node* Create_batchnormalization(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float epsilon,float momentum) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::batchnormalization, args);
newNode->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
newNode->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
newNode->f_(c10::Symbol::fromQualString("attr::momentum"),momentum);
return newNode;
}
torch::jit::Node* Create_compress(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::int32_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::compress, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_cosh(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::cosh, args);
return newNode;
}
torch::jit::Node* Create_erf(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::erf, args);
return newNode;
}
torch::jit::Node* Create_eyelike(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,int64_t k) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::eyelike, args);
newNode->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
newNode->i_(c10::Symbol::fromQualString("attr::k"),k);
return newNode;
}
torch::jit::Node* Create_flatten(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::flatten, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_gemm(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta,int64_t transA,int64_t transB) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::gemm, args);
newNode->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
newNode->f_(c10::Symbol::fromQualString("attr::beta"),beta);
newNode->i_(c10::Symbol::fromQualString("attr::transA"),transA);
newNode->i_(c10::Symbol::fromQualString("attr::transB"),transB);
return newNode;
}
torch::jit::Node* Create_greater(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::greater, args);
return newNode;
}
torch::jit::Node* Create_isnan(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::isnan, args);
return newNode;
}
torch::jit::Node* Create_less(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::less, args);
return newNode;
}
torch::jit::Node* Create_matmul(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::matmul, args);
return newNode;
}
torch::jit::Node* Create_maxunpool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::maxunpool, args);
newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
newNode->is_(c10::Symbol::fromQualString("attr::pads"),pads);
newNode->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return newNode;
}
torch::jit::Node* Create_meanvariancenormalization(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::meanvariancenormalization, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return newNode;
}
torch::jit::Node* Create_nonzero(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::nonzero, args);
return newNode;
}
torch::jit::Node* Create_onehot(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::onehot, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_prelu(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::prelu, args);
return newNode;
}
torch::jit::Node* Create_scatter(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::scatter, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_shrink(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float bias,float lambd) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::shrink, args);
newNode->f_(c10::Symbol::fromQualString("attr::bias"),bias);
newNode->f_(c10::Symbol::fromQualString("attr::lambd"),lambd);
return newNode;
}
torch::jit::Node* Create_sign(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::sign, args);
return newNode;
}
torch::jit::Node* Create_sinh(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::sinh, args);
return newNode;
}
torch::jit::Node* Create_where(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::where, args);
return newNode;
}
torch::jit::Node* Create_expand(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::expand, args);
return newNode;
}
torch::jit::Node* Create_max(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::max, args);
return newNode;
}
torch::jit::Node* Create_maxpool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,int64_t storage_order,const std::vector<int64_t> & strides) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::maxpool, args);
newNode->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
newNode->is_(c10::Symbol::fromQualString("attr::pads"),pads);
newNode->i_(c10::Symbol::fromQualString("attr::storage_order"),storage_order);
newNode->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return newNode;
}
torch::jit::Node* Create_mean(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::mean, args);
return newNode;
}
torch::jit::Node* Create_min(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::min, args);
return newNode;
}
torch::jit::Node* Create_sum(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::sum, args);
return newNode;
}
torch::jit::Node* Create_acos(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::acos, args);
return newNode;
}
torch::jit::Node* Create_add(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::add, args);
return newNode;
}
torch::jit::Node* Create_logical_and(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::logical_and, args);
return newNode;
}
torch::jit::Node* Create_asin(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::asin, args);
return newNode;
}
torch::jit::Node* Create_atan(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::atan, args);
return newNode;
}
torch::jit::Node* Create_averagepool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t count_include_pad,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::averagepool, args);
newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
newNode->i_(c10::Symbol::fromQualString("attr::count_include_pad"),count_include_pad);
newNode->is_(c10::Symbol::fromQualString("attr::pads"),pads);
newNode->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return newNode;
}
torch::jit::Node* Create_cos(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::cos, args);
return newNode;
}
torch::jit::Node* Create_div(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::div, args);
return newNode;
}
torch::jit::Node* Create_dropout(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,float ratio) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::dropout, args);
newNode->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
newNode->f_(c10::Symbol::fromQualString("attr::ratio"),ratio);
return newNode;
}
torch::jit::Node* Create_equal(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::equal, args);
return newNode;
}
torch::jit::Node* Create_mul(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::mul, args);
return newNode;
}
torch::jit::Node* Create_multinomial(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t dtype,int64_t sample_size,float seed) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::multinomial, args);
newNode->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
newNode->i_(c10::Symbol::fromQualString("attr::sample_size"),sample_size);
newNode->f_(c10::Symbol::fromQualString("attr::seed"),seed);
return newNode;
}
torch::jit::Node* Create_logical_or(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::logical_or, args);
return newNode;
}
torch::jit::Node* Create_pow(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::pow, args);
return newNode;
}
torch::jit::Node* Create_sin(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::sin, args);
return newNode;
}
torch::jit::Node* Create_sub(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::sub, args);
return newNode;
}
torch::jit::Node* Create_tan(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::tan, args);
return newNode;
}
torch::jit::Node* Create_logical_xor(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::logical_xor, args);
return newNode;
}
torch::jit::Node* Create_abs(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::abs, args);
return newNode;
}
torch::jit::Node* Create_argmax(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::argmax, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_argmin(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::argmin, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_ceil(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::ceil, args);
return newNode;
}
torch::jit::Node* Create_clip(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float max,float min) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::clip, args);
newNode->f_(c10::Symbol::fromQualString("attr::max"),max);
newNode->f_(c10::Symbol::fromQualString("attr::min"),min);
return newNode;
}
torch::jit::Node* Create_concat(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::concat, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_conv(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::conv, args);
newNode->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
newNode->i_(c10::Symbol::fromQualString("attr::group"),group);
newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
newNode->is_(c10::Symbol::fromQualString("attr::pads"),pads);
newNode->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return newNode;
}
torch::jit::Node* Create_convtranspose(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & dilations,int64_t group,const std::vector<int64_t> & kernel_shape,const std::vector<int64_t> & output_padding,const std::vector<int64_t> & output_shape,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::convtranspose, args);
newNode->is_(c10::Symbol::fromQualString("attr::dilations"),dilations);
newNode->i_(c10::Symbol::fromQualString("attr::group"),group);
newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
newNode->is_(c10::Symbol::fromQualString("attr::output_padding"),output_padding);
newNode->is_(c10::Symbol::fromQualString("attr::output_shape"),output_shape);
newNode->is_(c10::Symbol::fromQualString("attr::pads"),pads);
newNode->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return newNode;
}
torch::jit::Node* Create_depthtospace(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::depthtospace, args);
newNode->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
return newNode;
}
torch::jit::Node* Create_elu(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::elu, args);
newNode->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return newNode;
}
torch::jit::Node* Create_exp(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::exp, args);
return newNode;
}
torch::jit::Node* Create_floor(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::floor, args);
return newNode;
}
torch::jit::Node* Create_gather(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::gather, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_globalaveragepool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::globalaveragepool, args);
return newNode;
}
torch::jit::Node* Create_globallppool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t p) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::globallppool, args);
newNode->i_(c10::Symbol::fromQualString("attr::p"),p);
return newNode;
}
torch::jit::Node* Create_globalmaxpool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::globalmaxpool, args);
return newNode;
}
torch::jit::Node* Create_hardsigmoid(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float alpha,float beta) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::hardsigmoid, args);
newNode->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
newNode->f_(c10::Symbol::fromQualString("attr::beta"),beta);
return newNode;
}
torch::jit::Node* Create_hardmax(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::hardmax, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_identity(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::identity, args);
return newNode;
}
torch::jit::Node* Create_instancenormalization(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float epsilon) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::instancenormalization, args);
newNode->f_(c10::Symbol::fromQualString("attr::epsilon"),epsilon);
return newNode;
}
torch::jit::Node* Create_lrn(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t size,float alpha,float beta,float bias) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::lrn, args);
newNode->i_(c10::Symbol::fromQualString("attr::size"),size);
newNode->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
newNode->f_(c10::Symbol::fromQualString("attr::beta"),beta);
newNode->f_(c10::Symbol::fromQualString("attr::bias"),bias);
return newNode;
}
torch::jit::Node* Create_leakyrelu(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float alpha) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::leakyrelu, args);
newNode->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
return newNode;
}
torch::jit::Node* Create_log(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::log, args);
return newNode;
}
torch::jit::Node* Create_logsoftmax(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::logsoftmax, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_lpnormalization(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis,int64_t p) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::lpnormalization, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
newNode->i_(c10::Symbol::fromQualString("attr::p"),p);
return newNode;
}
torch::jit::Node* Create_lppool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & kernel_shape,int64_t p,const std::vector<int64_t> & pads,const std::vector<int64_t> & strides) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::lppool, args);
newNode->is_(c10::Symbol::fromQualString("attr::kernel_shape"),kernel_shape);
newNode->i_(c10::Symbol::fromQualString("attr::p"),p);
newNode->is_(c10::Symbol::fromQualString("attr::pads"),pads);
newNode->is_(c10::Symbol::fromQualString("attr::strides"),strides);
return newNode;
}
torch::jit::Node* Create_maxroipool(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & pooled_shape,float spatial_scale) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::maxroipool, args);
newNode->is_(c10::Symbol::fromQualString("attr::pooled_shape"),pooled_shape);
newNode->f_(c10::Symbol::fromQualString("attr::spatial_scale"),spatial_scale);
return newNode;
}
torch::jit::Node* Create_neg(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::neg, args);
return newNode;
}
torch::jit::Node* Create_logical_not(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::logical_not, args);
return newNode;
}
torch::jit::Node* Create_randomnormallike(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float mean,float scale,float seed) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::randomnormallike, args);
newNode->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
newNode->f_(c10::Symbol::fromQualString("attr::mean"),mean);
newNode->f_(c10::Symbol::fromQualString("attr::scale"),scale);
newNode->f_(c10::Symbol::fromQualString("attr::seed"),seed);
return newNode;
}
torch::jit::Node* Create_randomuniformlike(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::int32_t dtype,float high,float low,float seed) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::randomuniformlike, args);
newNode->i_(c10::Symbol::fromQualString("attr::dtype"),dtype);
newNode->f_(c10::Symbol::fromQualString("attr::high"),high);
newNode->f_(c10::Symbol::fromQualString("attr::low"),low);
newNode->f_(c10::Symbol::fromQualString("attr::seed"),seed);
return newNode;
}
torch::jit::Node* Create_reciprocal(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reciprocal, args);
return newNode;
}
torch::jit::Node* Create_reducel1(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducel1, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducel2(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducel2, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducelogsum(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducelogsum, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducelogsumexp(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducelogsumexp, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducemax(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducemax, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducemean(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducemean, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducemin(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducemin, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reduceprod(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reduceprod, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducesum(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducesum, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_reducesumsquare(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,std::vector<int64_t> axes,int64_t keepdims) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reducesumsquare, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
newNode->i_(c10::Symbol::fromQualString("attr::keepdims"),keepdims);
return newNode;
}
torch::jit::Node* Create_relu(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::relu, args);
return newNode;
}
torch::jit::Node* Create_reshape(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::reshape, args);
return newNode;
}
torch::jit::Node* Create_selu(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,float alpha,float gamma) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::selu, args);
newNode->f_(c10::Symbol::fromQualString("attr::alpha"),alpha);
newNode->f_(c10::Symbol::fromQualString("attr::gamma"),gamma);
return newNode;
}
torch::jit::Node* Create_shape(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::shape, args);
return newNode;
}
torch::jit::Node* Create_sigmoid(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::sigmoid, args);
return newNode;
}
torch::jit::Node* Create_size(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::size, args);
return newNode;
}
torch::jit::Node* Create_slice(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & ends,const std::vector<int64_t> & starts,const std::vector<int64_t> & axes) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::slice, args);
newNode->is_(c10::Symbol::fromQualString("attr::ends"),ends);
newNode->is_(c10::Symbol::fromQualString("attr::starts"),starts);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return newNode;
}
torch::jit::Node* Create_softmax(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::softmax, args);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_softplus(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::softplus, args);
return newNode;
}
torch::jit::Node* Create_softsign(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::softsign, args);
return newNode;
}
torch::jit::Node* Create_spacetodepth(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t blocksize) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::spacetodepth, args);
newNode->i_(c10::Symbol::fromQualString("attr::blocksize"),blocksize);
return newNode;
}
torch::jit::Node* Create_split(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,unsigned int num_outputs,int64_t axis,const std::vector<int64_t> & split) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::split, args, num_outputs);
newNode->i_(c10::Symbol::fromQualString("attr::num_outputs"),num_outputs);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
newNode->is_(c10::Symbol::fromQualString("attr::split"),split);
return newNode;
}
torch::jit::Node* Create_sqrt(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::sqrt, args);
return newNode;
}
torch::jit::Node* Create_squeeze(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::squeeze, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return newNode;
}
torch::jit::Node* Create_tanh(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::tanh, args);
return newNode;
}
torch::jit::Node* Create_tile(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::tile, args);
return newNode;
}
torch::jit::Node* Create_topk(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,int64_t k,int64_t axis) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::topk, args, 2);
newNode->i_(c10::Symbol::fromQualString("attr::k"),k);
newNode->i_(c10::Symbol::fromQualString("attr::axis"),axis);
return newNode;
}
torch::jit::Node* Create_transpose(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & perm) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::transpose, args);
newNode->is_(c10::Symbol::fromQualString("attr::perm"),perm);
return newNode;
}
torch::jit::Node* Create_unsqueeze(torch::jit::Graph &graph,  const std::vector<torch::jit::Value *>& args,const std::vector<int64_t> & axes) {
 torch::jit::Node *newNode = graph.create(Symbols::popart::unsqueeze, args);
newNode->is_(c10::Symbol::fromQualString("attr::axes"),axes);
return newNode;
}

