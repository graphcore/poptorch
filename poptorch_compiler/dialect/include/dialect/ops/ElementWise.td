// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/*
 * Element wise ops.
 */

class Poptorch_elem_unary<string name> :  Poptorch_Op<name, [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$in1);
    let results = (outs Poptorch_tensor:$result);

    let assemblyFormat = [{
        `(`$in1`)` `(`type($in1)`)` `->` type($result) attr-dict
    }];

    let builders = [OpBuilderDAG<(ins "mlir::Value":$v1),[{
        $_state.addOperands({v1});
        $_state.addTypes(v1.getType());
     }]>
    ];
}

def Poptorch_abs: Poptorch_elem_unary<"abs"> {}
def Poptorch_asin: Poptorch_elem_unary<"asin"> {}
def Poptorch_bitwiseNot: Poptorch_elem_unary<"bitwiseNot"> {}
def Poptorch_cbrt: Poptorch_elem_unary<"cbrt"> {}
def Poptorch_ceil: Poptorch_elem_unary<"ceil"> {}
def Poptorch_cos: Poptorch_elem_unary<"cos"> {}
def Poptorch_countLeadingZeros: Poptorch_elem_unary<"countLeadingZeros"> {}
def Poptorch_erf: Poptorch_elem_unary<"erf"> {}
def Poptorch_exp: Poptorch_elem_unary<"exp"> {}
def Poptorch_expm1: Poptorch_elem_unary<"expm1"> {}
def Poptorch_floor: Poptorch_elem_unary<"floor"> {}
def Poptorch_inv: Poptorch_elem_unary<"inv"> {}
def Poptorch_log: Poptorch_elem_unary<"log"> {}
def Poptorch_log1p: Poptorch_elem_unary<"log1p"> {}
def Poptorch_logicalNot: Poptorch_elem_unary<"logicalNot"> {}
def Poptorch_neg: Poptorch_elem_unary<"neg"> {}
def Poptorch_popcount: Poptorch_elem_unary<"popcount"> {}
def Poptorch_signum: Poptorch_elem_unary<"signum"> {}
def Poptorch_sin: Poptorch_elem_unary<"sin"> {}
def Poptorch_tan: Poptorch_elem_unary<"tan"> {}
def Poptorch_tanh: Poptorch_elem_unary<"tanh"> {}
def Poptorch_round: Poptorch_elem_unary<"round"> {}
def Poptorch_sqrt: Poptorch_elem_unary<"sqrt"> {}
def Poptorch_square: Poptorch_elem_unary<"square"> {}
def Poptorch_sigmoid: Poptorch_elem_unary<"sigmoid"> {}
def Poptorch_rsqrt: Poptorch_elem_unary<"rsqrt"> {}

class Poptorch_elem_unary_in_place<string name> :  Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$in1);
    let assemblyFormat = [{
        `(` `(``->``)` $in1 `)` `(`type($in1)`)` attr-dict
    }];
}

def Poptorch_abs_: Poptorch_elem_unary_in_place<"abs_"> {}
def Poptorch_asin_: Poptorch_elem_unary_in_place<"asin_"> {}
def Poptorch_bitwiseNot_: Poptorch_elem_unary_in_place<"bitwiseNot_"> {}
def Poptorch_cbrt_: Poptorch_elem_unary_in_place<"cbrt_"> {}
def Poptorch_ceil_: Poptorch_elem_unary_in_place<"ceil_"> {}
def Poptorch_cos_: Poptorch_elem_unary_in_place<"cos_"> {}
def Poptorch_countLeadingZeros_: Poptorch_elem_unary_in_place<"countLeadingZeros_"> {}
def Poptorch_erf_: Poptorch_elem_unary_in_place<"erf_"> {}
def Poptorch_exp_: Poptorch_elem_unary_in_place<"exp_"> {}
def Poptorch_expm1_: Poptorch_elem_unary_in_place<"expm1_"> {}
def Poptorch_floor_: Poptorch_elem_unary_in_place<"floor_"> {}
def Poptorch_inv_: Poptorch_elem_unary_in_place<"inv_"> {}
def Poptorch_log_: Poptorch_elem_unary_in_place<"log_"> {}
def Poptorch_log1p_: Poptorch_elem_unary_in_place<"log1p_"> {}
def Poptorch_logicalNot_: Poptorch_elem_unary_in_place<"logicalNot_"> {}
def Poptorch_neg_: Poptorch_elem_unary_in_place<"neg_"> {}
def Poptorch_popcount_: Poptorch_elem_unary_in_place<"popcount_"> {}
def Poptorch_signum_: Poptorch_elem_unary_in_place<"signum_"> {}
def Poptorch_sin_: Poptorch_elem_unary_in_place<"sin_"> {}
def Poptorch_tan_: Poptorch_elem_unary_in_place<"tan_"> {}
def Poptorch_tanh_: Poptorch_elem_unary_in_place<"tanh_"> {}
def Poptorch_round_: Poptorch_elem_unary_in_place<"round_"> {}
def Poptorch_sqrt_: Poptorch_elem_unary_in_place<"sqrt_"> {}
def Poptorch_square_: Poptorch_elem_unary_in_place<"square_"> {}
def Poptorch_sigmoid_: Poptorch_elem_unary_in_place<"sigmoid_"> {}
def Poptorch_rsqrt_: Poptorch_elem_unary_in_place<"rsqrt_"> {}


/*
 * Binary ops.
 */
class Poptorch_elem_binary<string name> :  Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let results = (outs Poptorch_tensor:$result);

    let assemblyFormat = [{
        `(`$in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` `->` type($result) attr-dict
    }];

    let builders = [OpBuilderDAG<(ins "mlir::Value":$v1, "mlir::Value":$v2),[{
        $_state.addOperands({v1, v2});
        $_state.addTypes(inferType(v1, v2));
     }]>
    ];

    // TODO: Will need to match the full shape inference rules.
    let extraClassDeclaration = [{
        static mlir::Type inferType(mlir::Value v1, mlir::Value v2) {
            // Get the types of the inputs.
            mlir::Type t1 = v1.getType();
            mlir::Type t2 = v2.getType();

            // Assume both are tensors for now.
            mlir::RankedTensorType t1_tensor = t1.cast<mlir::RankedTensorType>();
            mlir::RankedTensorType t2_tensor = t2.cast<mlir::RankedTensorType>();

            // Get element type out
            mlir::Type e1 = t1_tensor.getElementType();
            return mlir::RankedTensorType::get(t1_tensor.getShape(), e1);
        }
    }];
}

def Poptorch_atan2: Poptorch_elem_binary<"atan2"> {}
def Poptorch_bitwiseAnd: Poptorch_elem_binary<"bitwiseand"> {}
def Poptorch_bitwiseOr: Poptorch_elem_binary<"bitwiseor"> {}
def Poptorch_bitwiseXor: Poptorch_elem_binary<"bitwisexor"> {}
def Poptorch_bitwiseXnor: Poptorch_elem_binary<"bitwisexnor"> {}
def Poptorch_div: Poptorch_elem_binary<"div"> {}
def Poptorch_eq: Poptorch_elem_binary<"eq"> {}
def Poptorch_gteq: Poptorch_elem_binary<"gteq"> {}
def Poptorch_gt: Poptorch_elem_binary<"gt"> {}
def Poptorch_lteq: Poptorch_elem_binary<"lteq"> {}
def Poptorch_logicalAnd: Poptorch_elem_binary<"logicaland"> {}
def Poptorch_logicalOr: Poptorch_elem_binary<"logicalor"> {}
def Poptorch_lt: Poptorch_elem_binary<"lt"> {}
def Poptorch_max: Poptorch_elem_binary<"max"> {}
def Poptorch_min: Poptorch_elem_binary<"min"> {}
def Poptorch_neq: Poptorch_elem_binary<"neq"> {}
def Poptorch_pow: Poptorch_elem_binary<"pow"> {}
def Poptorch_rem: Poptorch_elem_binary<"rem"> {}
def Poptorch_shiftLeft: Poptorch_elem_binary<"shiftleft"> {}
def Poptorch_shiftRight: Poptorch_elem_binary<"shiftright"> {}
def Poptorch_shiftRightSignExtend: Poptorch_elem_binary<"shiftrightsignextend"> {}




class Poptorch_elem_binary_with_alpha<string name> :  Poptorch_elem_binary<name> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2, DefaultValuedAttr<F32Attr, "1.0">:$alpha);
    let results = (outs Poptorch_tensor:$result);


    let builders = [OpBuilderDAG<(ins "mlir::Value":$v1, "mlir::Value":$v2, "float":$alpha),[{
        $_state.addOperands({v1, v2});

        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));

        $_state.addTypes(inferType(v1, v2));
     }]>
    ];
}


def Poptorch_add: Poptorch_elem_binary_with_alpha<"add"> {}
def Poptorch_sub: Poptorch_elem_binary_with_alpha<"sub"> {}


// We add mul as a special case because multiply by one is common case in pytorch so we want to fold it.
// TODO: add some attributes here.
def Poptorch_mul: Poptorch_elem_binary<"mul"> {}


class Poptorch_elem_binary_in_place<string name> :  Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let assemblyFormat = [{
        `(` `(``->``)` $in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` attr-dict
    }];
}

def Poptorch_atan2_: Poptorch_elem_binary_in_place<"atan2_"> {}
def Poptorch_bitwiseAnd_: Poptorch_elem_binary_in_place<"bitwiseand_"> {}
def Poptorch_bitwiseOr_: Poptorch_elem_binary_in_place<"bitwiseor_"> {}
def Poptorch_bitwiseXor_: Poptorch_elem_binary_in_place<"bitwisexor_"> {}
def Poptorch_bitwiseXnor_: Poptorch_elem_binary_in_place<"bitwisexnor_"> {}
def Poptorch_div_: Poptorch_elem_binary_in_place<"div_"> {}
def Poptorch_eq_: Poptorch_elem_binary_in_place<"eq_"> {}
def Poptorch_gteq_: Poptorch_elem_binary_in_place<"gteq_"> {}
def Poptorch_gt_: Poptorch_elem_binary_in_place<"gt_"> {}
def Poptorch_lteq_: Poptorch_elem_binary_in_place<"lteq_"> {}
def Poptorch_logicalAnd_: Poptorch_elem_binary_in_place<"logicaland_"> {}
def Poptorch_logicalOr_: Poptorch_elem_binary_in_place<"logicalor_"> {}
def Poptorch_lt_: Poptorch_elem_binary_in_place<"lt_"> {}
def Poptorch_max_: Poptorch_elem_binary_in_place<"max_"> {}
def Poptorch_min_: Poptorch_elem_binary_in_place<"min_"> {}
def Poptorch_neq_: Poptorch_elem_binary_in_place<"neq_"> {}
def Poptorch_pow_: Poptorch_elem_binary_in_place<"pow_"> {}
def Poptorch_rem_: Poptorch_elem_binary_in_place<"rem_"> {}
def Poptorch_shiftLeft_: Poptorch_elem_binary_in_place<"shiftleft_"> {}
def Poptorch_shiftRight_: Poptorch_elem_binary_in_place<"shiftright_"> {}
def Poptorch_shiftRightSignExtend_: Poptorch_elem_binary_in_place<"shiftrightsignextend_"> {}
def Poptorch_mul_: Poptorch_elem_binary_in_place<"mul_"> {}


class Poptorch_elem_binary_in_place_with_alpha<string name> :  Poptorch_elem_binary_in_place<name> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2, DefaultValuedAttr<F32Attr, "1.0">:$alpha);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$v1, "mlir::Value":$v2, "float":$alpha),[{
        $_state.addOperands({v1, v2});
        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));
     }]>
    ];
}

def Poptorch_add_: Poptorch_elem_binary_in_place_with_alpha<"add_"> {}
def Poptorch_sub_: Poptorch_elem_binary_in_place_with_alpha<"sub_"> {}


class Poptorch_scaled<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2, F32Attr:$scale);

    let assemblyFormat = [{
        `(` `(``->``)`$in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` attr-dict
    }];


    let builders = [OpBuilderDAG<(ins "mlir::Value":$v1, "mlir::Value":$v2, "float":$scale),[{
        $_state.addOperands({v1, v2});
        $_state.addAttribute("scale", $_builder.getF32FloatAttr(scale));
     }]>
    ];
}

def Poptorch_scaledadd_: Poptorch_scaled<"scaled_add_"> {}

def Poptorch_scaledsub_: Poptorch_scaled<"scaled_sub_"> {}

class Poptorch_addcX<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$input, Poptorch_tensor:$tensor1, Poptorch_tensor:$tensor2, F32Attr:$value);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$input, "mlir::Value":$tensor1, "mlir::Value":$tensor2, "float":$value),[{
        $_state.addOperands({input, tensor1, tensor2});
        $_state.addAttribute("value", $_builder.getF32FloatAttr(value));
     }]>
    ];
}


class Poptorch_addcOutplaceX<string name> : Poptorch_addcX<name> {
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilderDAG<(ins "mlir::Value":$input, "mlir::Value":$tensor1, "mlir::Value":$tensor2, "float":$value),[{
        $_state.addOperands({input, tensor1, tensor2});
        $_state.addAttribute("value", $_builder.getF32FloatAttr(value));
        $_state.addTypes(input.getType());
     }]>
    ];
}

// aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> (Tensor(a!)
def Poptorch_addcmul : Poptorch_addcOutplaceX<"addcmul"> {}
def Poptorch_addcmul_ : Poptorch_addcX<"addcmul_"> {}


// aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> (Tensor(a!))
def Poptorch_addcdiv : Poptorch_addcOutplaceX<"addcdiv"> {}
def Poptorch_addcdiv_ : Poptorch_addcX<"addcdiv_"> {}

class Poptorch_binary_conditional<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let results = (outs Poptorch_tensor:$result);
    let assemblyFormat = [{
        `(`$in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` `->` type($result) attr-dict
    }];
    let builders = [OpBuilderDAG<(ins "mlir::Value":$v1, "mlir::Value":$v2),[{
        $_state.addOperands({v1, v2});
        mlir::RankedTensorType rtt = v1.getType().cast<mlir::RankedTensorType>();
        $_state.addTypes(mlir::RankedTensorType::get(rtt.getShape(), $_builder.getIntegerType(1, true)));
     }]>
    ];
}

def Poptorch_greater: Poptorch_binary_conditional<"greater"> {}
def Poptorch_less: Poptorch_binary_conditional<"less"> {}

def Poptorch_clamp: Poptorch_Op<"clamp", []> {
  // defining input arguments 
  let arguments = (ins Poptorch_tensor:$self,             
                        OptionalAttr<F64Attr>:$min, 
                        OptionalAttr<F64Attr>:$max); 
  // defining output arguments
  let results = (outs Poptorch_tensor:$result); 

  // builders allow us to tell MIR the types 
  let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                    "std::optional<double>":$min,
                                    "std::optional<double>":$max), 
  
  [{$_state.addOperands(self);
  
  if (min.has_value()){
      $_state.addAttribute("min", $_builder.getF64FloatAttr(min.value()));}

  if (max.has_value()) {
      $_state.addAttribute("max", $_builder.getF64FloatAttr(max.value()));}

  if (min.has_value() && max.has_value()){
       $_state.addAttribute("min", $_builder.getF64FloatAttr(min.value())),
       $_state.addAttribute("max", $_builder.getF64FloatAttr(max.value()));}

  $_state.addTypes(self.getType());
  }]>
  ];
}

def Poptorch_clampTensor: Poptorch_Op<"clampTensor", [AttrSizedOperandSegments]> {
  let arguments = (ins Poptorch_tensor:$self,             
                         Optional<Poptorch_tensor>:$min,
                         Optional<Poptorch_tensor>:$max); 
  // defining output arguments
  let results = (outs Poptorch_tensor:$result); 

  // builders allow us to tell MIR the types 
  let builders = [OpBuilderDAG<(ins "mlir::Value":$self,
                                    "mlir::Value":$min,
                                    "mlir::Value":$max),
  
  [{std::vector<mlir::Value> operands = {self};
  std::vector<std::int32_t> segments = {1, 0, 0};
  if (min){
      operands.push_back(min);
      segments[1]=1;
  }
  if (max){
      operands.push_back(max);
      segments[2]=1;
  }
    $_state.addOperands(operands);
    $_state.addAttribute("operand_segment_sizes", $_builder.getI32VectorAttr(segments));
  
    $_state.addTypes(self.getType());  
    }]>
  ];
}