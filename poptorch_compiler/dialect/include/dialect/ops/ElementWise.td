// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/*
 * Element wise ops.
 */

class Poptorch_elem_unary<string name, list<OpTrait> traits = []>
    : Poptorch_Op<name, traits # [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$in1);
    let results = (outs Poptorch_tensor:$result);

    let assemblyFormat = [{
        `(`$in1`)` `(`type($in1)`)` `->` type($result) attr-dict
    }];

    let builders = [OpBuilder<(ins "mlir::Value":$v1),[{
        $_state.addOperands({v1});
        $_state.addTypes(v1.getType());
     }]>
    ];
}

class Poptorch_elem_unary_backward<string name> : Poptorch_Op<name#"_backward", [SameOperandsAndResultShape]> {
    let arguments = (ins Poptorch_tensor:$grad_output,
                         Poptorch_tensor:$output);
    let results = (outs Poptorch_tensor:$grad_input);

    let builders = [OpBuilder<(ins "mlir::Value":$grad_output,
                                   "mlir::Value":$output),[{
        $_state.addOperands({grad_output, output});
        $_state.addTypes(output.getType());
     }]>
    ];
}

// Trigonometric functions.
def Poptorch_acos: Poptorch_elem_unary<"acos", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_acosh: Poptorch_elem_unary<"acosh", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_asin: Poptorch_elem_unary<"asin", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_asinh: Poptorch_elem_unary<"asinh", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_atan: Poptorch_elem_unary<"atan", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_atanh: Poptorch_elem_unary<"atanh", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_cos: Poptorch_elem_unary<"cos", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_cosh: Poptorch_elem_unary<"cosh", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_sin: Poptorch_elem_unary<"sin", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_sinh: Poptorch_elem_unary<"sinh", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_tan: Poptorch_elem_unary<"tan", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_tanh: Poptorch_elem_unary<"tanh", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}

def Poptorch_abs: Poptorch_elem_unary<"abs"> {
    let arguments = (ins Poptorch_non_boolean_tensor:$in1);
}
def Poptorch_bitwiseNot: Poptorch_elem_unary<"bitwiseNot"> {
    let arguments = (ins Poptorch_integral_tensor:$in1);
    let hasCanonicalizeMethod = 1;
}
def Poptorch_ceil: Poptorch_elem_unary<"ceil"> {
    let arguments = (ins Poptorch_float_tensor:$in1);
}
def Poptorch_countLeadingZeros: Poptorch_elem_unary<"countLeadingZeros"> {}
def Poptorch_erf: Poptorch_elem_unary<"erf", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_exp: Poptorch_elem_unary<"exp", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_expm1: Poptorch_elem_unary<"expm1"> {}
def Poptorch_floor: Poptorch_elem_unary<"floor"> {
    let arguments = (ins Poptorch_float_tensor:$in1);
}
def Poptorch_inv: Poptorch_elem_unary<"inv"> {}
def Poptorch_log: Poptorch_elem_unary<"log", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_log1p: Poptorch_elem_unary<"log1p", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_logicalNot: Poptorch_elem_unary<"logicalNot", [ImplicitCastOperand<0>, ImplicitCastToBool]> {}
def Poptorch_neg: Poptorch_elem_unary<"neg"> {}
def Poptorch_popcount: Poptorch_elem_unary<"popcount"> {}
def Poptorch_signum: Poptorch_elem_unary<"signum"> {
    let hasCanonicalizeMethod = 1;
}
def Poptorch_round: Poptorch_elem_unary<"round"> {
    let arguments = (ins Poptorch_float_tensor:$in1);
}
def Poptorch_trunc: Poptorch_elem_unary<"trunc"> {
    let arguments = (ins Poptorch_float_tensor:$in1);
}
def Poptorch_sqrt: Poptorch_elem_unary<"sqrt", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_rsqrt: Poptorch_elem_unary<"rsqrt", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_square: Poptorch_elem_unary<"square", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_sigmoid: Poptorch_elem_unary<"sigmoid", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}

def Poptorch_tanh_backward: Poptorch_elem_unary_backward<"tanh"> {}
def Poptorch_sigmoid_backward: Poptorch_elem_unary_backward<"sigmoid"> {}

def Poptorch_isnan :  Poptorch_Op<"isnan"> {
    let arguments = (ins Poptorch_tensor:$self);
    let results = (outs Poptorch_tensor:$result);

    let hasCanonicalizeMethod = 1;

    let builders = [OpBuilder<(ins "mlir::Value":$self),[{
        $_state.addOperands({self});

        mlir::RankedTensorType rtt = self.getType().cast<mlir::RankedTensorType>();
        $_state.addTypes(mlir::RankedTensorType::get(rtt.getShape(), $_builder.getIntegerType(1, false)));
     }]>
    ];
}

def Poptorch_erfc : Poptorch_elem_unary<"erfc", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_frac : Poptorch_elem_unary<"frac"> {
    let arguments = (ins Poptorch_float_tensor:$in1);
}
def Poptorch_log10 : Poptorch_elem_unary<"log10", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_log2 : Poptorch_elem_unary<"log2", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_reciprocal : Poptorch_elem_unary<"reciprocal", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}

class Poptorch_elem_unary_in_place<string name, list<OpTrait> traits = []> : Poptorch_Op<name, traits> {
    let arguments = (ins Poptorch_tensor:$in1);
    let assemblyFormat = [{
        `(` `(``->``)` $in1 `)` `(`type($in1)`)` attr-dict
    }];
}

// In-place trigonometric functions.
def Poptorch_acos_: Poptorch_elem_unary_in_place<"acos_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_acosh_: Poptorch_elem_unary_in_place<"acosh_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_asin_: Poptorch_elem_unary_in_place<"asin_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_asinh_: Poptorch_elem_unary_in_place<"asinh_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_atan_: Poptorch_elem_unary_in_place<"atan_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_atanh_: Poptorch_elem_unary_in_place<"atanh_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_cos_: Poptorch_elem_unary_in_place<"cos_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_cosh_: Poptorch_elem_unary_in_place<"cosh_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_sin_: Poptorch_elem_unary_in_place<"sin_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_sinh_: Poptorch_elem_unary_in_place<"sinh_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_tan_: Poptorch_elem_unary_in_place<"tan_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_tanh_: Poptorch_elem_unary_in_place<"tanh_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}

def Poptorch_abs_: Poptorch_elem_unary_in_place<"abs_"> {}
def Poptorch_bitwiseNot_: Poptorch_elem_unary_in_place<"bitwiseNot_"> {
    let arguments = (ins Poptorch_integral_tensor:$in1);
    let hasCanonicalizeMethod = 1;
}
def Poptorch_ceil_: Poptorch_elem_unary_in_place<"ceil_"> {}
def Poptorch_countLeadingZeros_: Poptorch_elem_unary_in_place<"countLeadingZeros_"> {}
def Poptorch_erf_: Poptorch_elem_unary_in_place<"erf_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_erfc_: Poptorch_elem_unary_in_place<"erfc_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_exp_: Poptorch_elem_unary_in_place<"exp_"> {}
def Poptorch_expm1_: Poptorch_elem_unary_in_place<"expm1_"> {}
def Poptorch_floor_: Poptorch_elem_unary_in_place<"floor_"> {}
def Poptorch_inv_: Poptorch_elem_unary_in_place<"inv_"> {}
def Poptorch_log_: Poptorch_elem_unary_in_place<"log_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_log10_: Poptorch_elem_unary_in_place<"log10_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_log1p_: Poptorch_elem_unary_in_place<"log1p_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_frac_ : Poptorch_elem_unary_in_place<"frac_"> {
    let arguments = (ins Poptorch_float_tensor:$in1);
}
def Poptorch_log2_: Poptorch_elem_unary_in_place<"log2_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_logicalNot_: Poptorch_elem_unary_in_place<"logicalNot_", [ImplicitCastOperand<0>, ImplicitCastToBool]> {}
def Poptorch_neg_: Poptorch_elem_unary_in_place<"neg_"> {}
def Poptorch_popcount_: Poptorch_elem_unary_in_place<"popcount_"> {}
def Poptorch_reciprocal_ : Poptorch_elem_unary_in_place<"reciprocal_", [ImplicitCastOperand<0>, ImplicitCastToFloat]> {}
def Poptorch_round_: Poptorch_elem_unary_in_place<"round_"> {
    let arguments = (ins Poptorch_float_tensor:$in1);
}
def Poptorch_rsqrt_: Poptorch_elem_unary_in_place<"rsqrt_"> {}
def Poptorch_sigmoid_: Poptorch_elem_unary_in_place<"sigmoid_"> {}
def Poptorch_signum_: Poptorch_elem_unary_in_place<"signum_"> {}
def Poptorch_sqrt_: Poptorch_elem_unary_in_place<"sqrt_"> {}
def Poptorch_square_: Poptorch_elem_unary_in_place<"square_"> {}

/*
 * Binary ops.
 */
class Poptorch_elem_binary<string name, list<OpTrait> traits = []>
  : Poptorch_Op<name, traits # [ImplicitCastOperand<0>, ImplicitCastOperand<1>]> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let results = (outs Poptorch_tensor:$result);

    let assemblyFormat = [{
        `(`$in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` `->` type($result) attr-dict
    }];

    let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2),[{
        $_state.addOperands({v1, v2});
        $_state.addTypes(inferType(v1, v2));
     }]>
    ];

    let extraClassDeclaration = [{
        static mlir::Type inferType(mlir::Value v1, mlir::Value v2) {
            auto out_shape = broadcast(getShape(v1), getShape(v2));
            return mlir::RankedTensorType::get(out_shape, getElementType(v1));
        }
    }];
}

def Poptorch_atan2: Poptorch_elem_binary<"atan2", [ImplicitCastToFloat]> {}
def Poptorch_bitwiseAnd: Poptorch_elem_binary<"bitwiseAnd"> {
    let arguments = (ins Poptorch_integral_tensor:$in1, Poptorch_integral_tensor:$in2);
    let hasCanonicalizeMethod = 1;
}
def Poptorch_bitwiseOr: Poptorch_elem_binary<"bitwiseOr"> {
    let arguments = (ins Poptorch_integral_tensor:$in1, Poptorch_integral_tensor:$in2);
    let hasCanonicalizeMethod = 1;
}
def Poptorch_bitwiseXor: Poptorch_elem_binary<"bitwiseXor"> {
    let arguments = (ins Poptorch_integral_tensor:$in1, Poptorch_integral_tensor:$in2);
    let hasCanonicalizeMethod = 1;
}
def Poptorch_bitwiseXnor: Poptorch_elem_binary<"bitwiseXnor"> {
    let arguments = (ins Poptorch_integral_tensor:$in1, Poptorch_integral_tensor:$in2);
    let hasCanonicalizeMethod = 1;
}
def Poptorch_div: Poptorch_elem_binary<"div", [ImplicitCastToFloat]> {}
def Poptorch_floor_divide : Poptorch_elem_binary<"floor_divide"> {
    let arguments = (ins Poptorch_non_boolean_tensor:$in1, Poptorch_non_boolean_tensor:$in2);
    let verifier = [{
        emitWarning("floor_divide is deprecated. Use div with rounding_mode='trunc' for equivalent behaviour");
        return mlir::success();
    }];
}
def Poptorch_rem : Poptorch_elem_binary<"rem"> {
    let arguments = (ins Poptorch_non_boolean_tensor:$in1, Poptorch_non_boolean_tensor:$in2);
}
def Poptorch_remainder: Poptorch_elem_binary<"remainder"> {
    let arguments = (ins Poptorch_non_boolean_tensor:$in1, Poptorch_non_boolean_tensor:$in2);
}
def Poptorch_maximum: Poptorch_elem_binary<"maximum"> {
    let hasCanonicalizeMethod = 1;
}
def Poptorch_minimum: Poptorch_elem_binary<"minimum"> {
    let hasCanonicalizeMethod = 1;
}
def Poptorch_pow: Poptorch_elem_binary<"pow"> {
    let arguments = (ins Poptorch_non_boolean_tensor:$in1, Poptorch_non_boolean_tensor:$in2);
}
def Poptorch_shiftLeft: Poptorch_elem_binary<"shiftleft"> {}
def Poptorch_shiftRight: Poptorch_elem_binary<"shiftright"> {}
def Poptorch_shiftRightSignExtend: Poptorch_elem_binary<"shiftrightsignextend"> {}

def Poptorch_div_mode: Poptorch_NotImplementedOp<"div_mode",
        [ImplicitCastOperand<0>, ImplicitCastOperand<1>]> {
    let arguments = (ins Poptorch_tensor:$self, Poptorch_tensor:$other,
                     OptionalAttr<StrAttr>:$rounding_mode);
    let results = (outs Poptorch_tensor:$result);
    let builders = [
        OpBuilder<(ins "mlir::Value":$self, "mlir::Value":$other,
                   "const std::optional<std::string>&":$rounding_mode), [{
            $_state.addOperands({self, other});

            if (rounding_mode.has_value())
                $_state.addAttribute("rounding_mode",
                                     $_builder.getStringAttr(*rounding_mode));

            $_state.addTypes(inferType(self, other));
        }]>
    ];
    let extraClassDeclaration = [{
        static mlir::Type inferType(mlir::Value v1, mlir::Value v2) {
            auto out_shape = broadcast(getShape(v1), getShape(v2));
            return mlir::RankedTensorType::get(out_shape, getElementType(v1));
        }
    }];
}

class Poptorch_elem_binary_with_alpha<string name> :  Poptorch_elem_binary<name> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2, DefaultValuedAttr<F32Attr, "1.0">:$alpha);
    let results = (outs Poptorch_tensor:$result);


    let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2, "float":$alpha),[{
        $_state.addOperands({v1, v2});

        $_state.addAttribute("alpha",$_builder.getF32FloatAttr(alpha));

        $_state.addTypes(inferType(v1, v2));
     }]>
    ];
}


def Poptorch_add: Poptorch_elem_binary_with_alpha<"add"> {
    let hasCanonicalizeMethod = 1;
}
def Poptorch_sub: Poptorch_elem_binary_with_alpha<"sub"> {}


// We add mul as a special case because multiply by one is common case in pytorch so we want to fold it.
// TODO: add some attributes here.
def Poptorch_mul: Poptorch_elem_binary<"mul"> {
    let hasCanonicalizeMethod = 1;
}


class Poptorch_elem_binary_in_place<string name> :  Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let assemblyFormat = [{
        `(` `(``->``)` $in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` attr-dict
    }];
}

def Poptorch_atan2_: Poptorch_elem_binary_in_place<"atan2_"> {}
def Poptorch_bitwiseAnd_: Poptorch_elem_binary_in_place<"bitwiseAnd_"> {}
def Poptorch_bitwiseOr_: Poptorch_elem_binary_in_place<"bitwiseOr_"> {}
def Poptorch_bitwiseXor_: Poptorch_elem_binary_in_place<"bitwiseXor_"> {}
def Poptorch_bitwiseXnor_: Poptorch_elem_binary_in_place<"bitwiseXnor_"> {}
def Poptorch_div_: Poptorch_elem_binary_in_place<"div_"> {}
def Poptorch_eq_: Poptorch_elem_binary_in_place<"eq_"> {}
def Poptorch_gteq_: Poptorch_elem_binary_in_place<"gteq_"> {}
def Poptorch_gt_: Poptorch_elem_binary_in_place<"gt_"> {}
def Poptorch_lteq_: Poptorch_elem_binary_in_place<"lteq_"> {}
def Poptorch_logicalAnd_: Poptorch_elem_binary_in_place<"logicalAnd_"> {}
def Poptorch_logicalOr_: Poptorch_elem_binary_in_place<"logicalOr_"> {}
def Poptorch_lt_: Poptorch_elem_binary_in_place<"lt_"> {}
def Poptorch_maximum_: Poptorch_elem_binary_in_place<"maximum_"> {}
def Poptorch_minimum_: Poptorch_elem_binary_in_place<"minimum_"> {}
def Poptorch_neq_: Poptorch_elem_binary_in_place<"neq_"> {}
def Poptorch_pow_: Poptorch_elem_binary_in_place<"pow_"> {
    let arguments = (ins Poptorch_non_boolean_tensor:$in1, Poptorch_non_boolean_tensor:$in2);
}
def Poptorch_rem_: Poptorch_elem_binary_in_place<"rem_"> {}
def Poptorch_shiftLeft_: Poptorch_elem_binary_in_place<"shiftleft_"> {}
def Poptorch_shiftRight_: Poptorch_elem_binary_in_place<"shiftright_"> {}
def Poptorch_shiftRightSignExtend_: Poptorch_elem_binary_in_place<"shiftrightsignextend_"> {}
def Poptorch_mul_: Poptorch_elem_binary_in_place<"mul_"> {}


class Poptorch_elem_binary_in_place_with_alpha<string name> :  Poptorch_elem_binary_in_place<name> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2, DefaultValuedAttr<F32Attr, "1.0">:$alpha);

    let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2, "float":$alpha),[{
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


    let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2, "float":$scale),[{
        $_state.addOperands({v1, v2});
        $_state.addAttribute("scale", $_builder.getF32FloatAttr(scale));
     }]>
    ];
}

def Poptorch_scaledadd_: Poptorch_scaled<"scaled_add_"> {}

def Poptorch_scaledsub_: Poptorch_scaled<"scaled_sub_"> {}

class Poptorch_addcX<string name> : Poptorch_Op<name, []> {
    let arguments = (ins Poptorch_tensor:$input, Poptorch_tensor:$tensor1, Poptorch_tensor:$tensor2, F32Attr:$value);

    let builders = [OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$tensor1, "mlir::Value":$tensor2, "float":$value),[{
        $_state.addOperands({input, tensor1, tensor2});
        $_state.addAttribute("value", $_builder.getF32FloatAttr(value));
     }]>
    ];
}


class Poptorch_addcOutplaceX<string name> : Poptorch_addcX<name> {
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$input, "mlir::Value":$tensor1, "mlir::Value":$tensor2, "float":$value),[{
        $_state.addOperands({input, tensor1, tensor2});

        auto in_shape = getShape(input);
        auto t1_shape = getShape(tensor1);
        auto t2_shape = getShape(tensor2);

        auto out_shape = broadcast(in_shape, broadcast(t1_shape, t2_shape));

        $_state.addAttribute("value", $_builder.getF32FloatAttr(value));
        $_state.addTypes(mlir::RankedTensorType::get(out_shape, getElementType(input)));

     }]>
    ];
}

// aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> (Tensor(a!)
def Poptorch_addcmul : Poptorch_addcOutplaceX<"addcmul"> {}
def Poptorch_addcmul_ : Poptorch_addcX<"addcmul_"> {}


// aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> (Tensor(a!))
def Poptorch_addcdiv : Poptorch_addcOutplaceX<"addcdiv"> {}
def Poptorch_addcdiv_ : Poptorch_addcX<"addcdiv_"> {}

class Poptorch_elem_binary_scalar<string name, list<OpTrait> traits = []>
        : Poptorch_Op<name, traits> {
    let arguments = (ins Poptorch_tensor:$lhs, F32Attr:$rhs);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$lhs, "float":$rhs),[{
        $_state.addOperands(lhs);
        $_state.addAttribute("rhs", $_builder.getF32FloatAttr(rhs));

        $_state.addTypes(lhs.getType());
     }]>
    ];
}

def Poptorch_pow_Tensor_Scalar_out: Poptorch_elem_binary_scalar<"pow_Tensor_Scalar_out"> {}

class Poptorch_binary_conditional<string name, list<OpTrait> traits = []>
        : Poptorch_Op<name, traits # [ImplicitCastOperand<0>, ImplicitCastOperand<1>]> {
    let arguments = (ins Poptorch_tensor:$in1, Poptorch_tensor:$in2);
    let results = (outs Poptorch_tensor_no_grad:$result);
    let assemblyFormat = [{
        `(`$in1 `,` $in2 `)` `(`type($in1)`,`type($in2)`)` `->` type($result) attr-dict
    }];
    let builders = [OpBuilder<(ins "mlir::Value":$v1, "mlir::Value":$v2),[{
        $_state.addOperands({v1, v2});
        auto output_shape = broadcast(getShape(v1), getShape(v2));
        $_state.addTypes(mlir::RankedTensorType::get(output_shape, $_builder.getIntegerType(1, false)));
     }]>
    ];
}

def Poptorch_eq: Poptorch_binary_conditional<"eq"> {}
def Poptorch_neq: Poptorch_binary_conditional<"neq"> {}
def Poptorch_gteq: Poptorch_binary_conditional<"gteq"> {}
def Poptorch_gt: Poptorch_binary_conditional<"gt"> {}
def Poptorch_lteq: Poptorch_binary_conditional<"lteq"> {}
def Poptorch_lt: Poptorch_binary_conditional<"lt"> {}
def Poptorch_logicalAnd: Poptorch_binary_conditional<"logicalAnd", [ImplicitCastToBool]> {}
def Poptorch_logicalOr: Poptorch_binary_conditional<"logicalOr", [ImplicitCastToBool]> {}
def Poptorch_logicalXor: Poptorch_binary_conditional<"logicalXor", [ImplicitCastToBool]> {}

class Poptorch_NotImplemented_binary_conditional_scalar<string name> : Poptorch_NotImplementedOp<name, []> {
    let arguments = (ins Poptorch_tensor:$lhs, F32Attr:$rhs);
    let results = (outs Poptorch_tensor_no_grad:$result);
    let builders = [OpBuilder<(ins "mlir::Value":$lhs, "float":$rhs),[{
        $_state.addOperands({lhs});
        $_state.addAttribute("rhs", $_builder.getF32FloatAttr(rhs));
        mlir::RankedTensorType rtt = lhs.getType().cast<mlir::RankedTensorType>();
        $_state.addTypes(mlir::RankedTensorType::get(rtt.getShape(), $_builder.getIntegerType(1, false)));
     }]>
    ];
}

def Poptorch_lt_Scalar_out : Poptorch_NotImplemented_binary_conditional_scalar<"lt_Scalar_out"> {}
def Poptorch_le_Scalar_out : Poptorch_NotImplemented_binary_conditional_scalar<"le_Scalar_out"> {}
def Poptorch_gt_Scalar_out : Poptorch_NotImplemented_binary_conditional_scalar<"gt_Scalar_out"> {}
def Poptorch_ge_Scalar_out : Poptorch_NotImplemented_binary_conditional_scalar<"ge_Scalar_out"> {}
def Poptorch_eq_Scalar_out : Poptorch_NotImplemented_binary_conditional_scalar<"eq_Scalar_out"> {}
def Poptorch_ne_Scalar_out : Poptorch_NotImplemented_binary_conditional_scalar<"ne_Scalar_out"> {}

def Poptorch_clamp: Poptorch_Op<"clamp", []> {
  // defining input arguments
  let arguments = (ins Poptorch_tensor:$self,
                        OptionalAttr<F64Attr>:$min,
                        OptionalAttr<F64Attr>:$max);
  // defining output arguments
  let results = (outs Poptorch_tensor:$result);

  // builders allow us to tell MIR the types
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "std::optional<double>":$min,
                                 "std::optional<double>":$max), [{
    $_state.addOperands(self);

    if (min.has_value()){
        $_state.addAttribute("min", $_builder.getF64FloatAttr(min.value()));}

    if (max.has_value()) {
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
  let results = (outs Poptorch_non_boolean_tensor:$result);

  let verifier = [{
    if (!min() && !max()) {
        ERROR("torch.clamp: At least one of 'min' or 'max' must not be None");
    }
    return mlir::success();
  }];

  // builders allow us to tell MIR the types
  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                 "mlir::Value":$min,
                                 "mlir::Value":$max),[{
    std::vector<mlir::Value> operands = {self};
    std::vector<std::int32_t> segments = {1, 0, 0};

    auto self_type = self.getType().cast<mlir::RankedTensorType>();
    auto element_type = self_type.getElementType();
    auto out_shape = getShape(self);

    if (min){
      operands.push_back(min);
      segments[1]=1;

      out_shape = broadcast(out_shape, getShape(min));
    }
    if (max){
      operands.push_back(max);
      segments[2]=1;

      out_shape = broadcast(out_shape, getShape(max));
    }
    $_state.addOperands(operands);
    $_state.addAttribute("operand_segment_sizes", $_builder.getI32VectorAttr(segments));

    $_state.addTypes(mlir::RankedTensorType::get(out_shape, element_type));
    }]>
  ];
}

class Poptorch_clamp_common<string name>: Poptorch_NotImplementedOp<name, []> {
  let arguments = (ins Poptorch_tensor:$self,
                       F64Attr:$value);
  let results = (outs Poptorch_tensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$self, "double":$value), [{
      $_state.addOperands({self});

      $_state.addAttribute("value", $_builder.getF64FloatAttr(value));

      $_state.addTypes(self.getType());
    }]>
  ];
}

def Poptorch_clamp_min: Poptorch_clamp_common<"clamp_min">;
def Poptorch_clamp_max: Poptorch_clamp_common<"clamp_max">;

class Poptorch_clamp_tensor_common<string name>: Poptorch_NotImplementedOp<name, []> {
  let arguments = (ins Poptorch_tensor:$self,
                       Poptorch_tensor:$value);
  let results = (outs Poptorch_tensor:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$self, "mlir::Value":$value), [{
      $_state.addOperands({self, value});

      $_state.addTypes(self.getType());
    }]>
  ];
}

def Poptorch_clamp_min_tensor: Poptorch_clamp_tensor_common<"clamp_min_tensor">;
def Poptorch_clamp_max_tensor: Poptorch_clamp_tensor_common<"clamp_max_tensor">;

def Poptorch_threshold_out: Poptorch_Op<"threshold_out", []> {
  let arguments = (ins Poptorch_non_boolean_tensor:$self,
                       F32Attr:$threshold,
                       F32Attr:$value);
  let results = (outs Poptorch_tensor:$result);

  let builders = [OpBuilder<(ins "mlir::Value":$self,
                                    "float":$threshold,
                                    "float":$value), [{
    $_state.addOperands(self);

    $_state.addAttribute("threshold", $_builder.getF32FloatAttr(threshold));
    $_state.addAttribute("value", $_builder.getF32FloatAttr(value));

    $_state.addTypes(self.getType());
  }]>
  ];
}

/*
 * Scatter ops.
 */

class Poptorch_elem_binary_scatter<string name, list<OpTrait> traits = []> : Poptorch_NotImplementedOp<name, traits> {
    let arguments = (ins Poptorch_tensor:$self, I64Attr:$dim, Poptorch_tensor:$index, Poptorch_tensor:$src);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self, "std::int64_t":$dim, "mlir::Value":$index, "mlir::Value":$src),[{
        $_state.addOperands({self, index, src});
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addTypes(self.getType());
      }]>
    ];
}

def Poptorch_scatter_src_out : Poptorch_elem_binary_scatter<"scatter_src_out"> {}
def Poptorch_scatter_value_out : Poptorch_elem_binary_scatter<"scatter_value_out"> {}
def Poptorch_scatter_add_out : Poptorch_elem_binary_scatter<"scatter_add_out"> {}

class Poptorch_elem_binary_scatter_reduce<string name, list<OpTrait> traits = []> : Poptorch_NotImplementedOp<name, traits> {
    let arguments = (ins Poptorch_tensor:$self,
                         I64Attr:$dim,
                         Poptorch_tensor:$index,
                         Poptorch_tensor:$src,
                         StrAttr:$reduce);
    let results = (outs Poptorch_tensor:$result);

    let builders = [OpBuilder<(ins "mlir::Value":$self,
                                   "std::int64_t":$dim,
                                   "mlir::Value":$index,
                                   "mlir::Value":$src,
                                   "std::string":$reduce),[{
        $_state.addOperands({self, index, src});
        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        $_state.addAttribute("reduce", $_builder.getStringAttr(reduce));
        $_state.addTypes(self.getType());
      }]>
    ];
}

def Poptorch_scatter_reduce_out : Poptorch_elem_binary_scatter_reduce<"scatter_reduce_out"> {}
def Poptorch_scatter_value_reduce_out : Poptorch_elem_binary_scatter_reduce<"scatter_value_reduce_out"> {}

// Note: these are provided by the torch_scatter library
class Poptorch_scatter_minmax<string name, list<OpTrait> traits = []> : Poptorch_NotImplementedOp<name, traits> {
    let arguments = (ins Poptorch_tensor:$src,
                         Poptorch_tensor:$index,
                         I64Attr:$dim,
                         Optional<Poptorch_tensor>:$out,
                         OptionalAttr<I64Attr>:$dim_size);

    let results = (outs Poptorch_tensor:$result, Poptorch_tensor_no_grad:$args);

    let builders = [OpBuilder<(ins "mlir::Value":$src,
                                   "mlir::Value":$index,
                                   "std::int64_t":$dim,
                                   "mlir::Value":$out,
                                   "std::optional<int64_t>":$dim_size),[{
        std::vector<mlir::Value> operands = {src, index};
        std::vector<std::int32_t> segments = {1, 1, 0};

        std::vector<std::int64_t> out_shape = getShape(src);

        dim = convertToPositiveDim(dim, out_shape.size());

        if (out) {
            operands.push_back(out);
            segments[2] = 1;

            out_shape = getShape(out);
        } else if (dim_size.has_value()) {
            out_shape[dim] = *dim_size;
        } else {
            ERROR("You must provide either an output parameter or specify dim_size so the output shape may be inferred");
        }

        $_state.addOperands(operands);
        $_state.addAttribute("operand_segment_sizes", $_builder.getI32VectorAttr(segments));

        $_state.addAttribute("dim", $_builder.getI64IntegerAttr(dim));
        if (dim_size.has_value()) {
            $_state.addAttribute("dim_size", $_builder.getI64IntegerAttr(*dim_size));
            ERROR_ON_MSG(*dim_size != out_shape[dim], "dim_size expected to be the same as out.shape()[dim]");
        }

        $_state.addTypes({mlir::RankedTensorType::get(out_shape, getElementType(src)),
                          mlir::RankedTensorType::get(out_shape, $_builder.getIntegerType(32, true))});
      }]>
    ];
}

def Poptorch_scatter_max : Poptorch_scatter_minmax<"scatter_max"> {}
def Poptorch_scatter_min : Poptorch_scatter_minmax<"scatter_min"> {}
