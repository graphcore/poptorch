// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

// aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases,
// int num_layers, float dropout, bool train, bool bidirectional,
// bool batch_first) -> (Tensor, Tensor, Tensor)
def Poptorch_lstm_input: Poptorch_Op<"lstm_input", [AttrSizedOperandSegments]> {
    let arguments = (ins Poptorch_tensor:$input, Poptorch_tensorlist:$hx,
                         Poptorch_tensorlist:$params, BoolAttr:$has_biases,
                         I64Attr:$num_layers, F32Attr:$dropout,
                         BoolAttr:$train, BoolAttr:$bidrectional,
                         BoolAttr:$batch_first);
    let results = (outs Poptorch_tensor:$output, Poptorch_tensor:$hy,
                        Poptorch_tensor:$cy);

    let builders = [OpBuilder<(ins "mlir::Value":$input,
                                   "llvm::SmallVector<mlir::Value, 4>":$hx,
                                   "llvm::SmallVector<mlir::Value, 4>":$params,
                                   "bool":$has_biases,
                                   "std::int64_t":$num_layers,
                                   "float":$dropout,
                                   "bool":$train,
                                   "bool":$bidrectional,
                                   "bool":$batch_first), [{

        ERROR_ON_MSG(hx.size() != 2, "lstm expects two hidden states");
        mlir::Value h_n = hx[0];
        mlir::Value c_n = hx[1];
        auto h_n_shape = getShape(h_n);
        auto c_n_shape = getShape(c_n);

        bool has_projections = (h_n_shape.back() != c_n_shape.back());

        // For now, we do not support projections or bidirectional
        ERROR_ON_MSG(has_projections,
                     "PopTorch does not yet support LSTMs with projections.");
        ERROR_ON_MSG(bidrectional,
                     "PopTorch does not yet support bidrectional LSTMs.");

        // Dropout must be 0.0
        ERROR_ON_MSG(dropout != 0.0, "PopTorch does not yet support LSTMs with "
                                     "dropout not equal to 0.0.");

        // Check all tensors are the same type
        mlir::Type input_type = getElementType(input);

        auto check_tensors = [&input_type](const std::string &name,
                                const mlir::Value& t) {
            if(!t) { return; };

            mlir::Type t_type = getElementType(t);
            ERROR_ON_MSG(t_type != input_type,
              "Input and " <<  name << " are not the same dtype, " <<
              elementTypeToString(input_type) << " vs "  <<
              elementTypeToString(t_type) << ".");
        };

        for(const mlir::Value &h : hx ) {
            check_tensors("hidden", h);
        }
        for(const mlir::Value  &p : params ) {
            check_tensors("parameter", p);
        }

        // Check the parameter sizes
        int param_divider;
        if (has_biases) {
            if (has_projections) {
                param_divider = 5;
            } else {
                param_divider = 4;
            }
        } else {
            if (has_projections) {
                param_divider = 3;
            } else {
                param_divider = 2;
            }
        }

        ERROR_ON_MSG(params.size()%param_divider != 0,
                     "got an incorrect number of RNN parameters: got " <<
                     params.size() << " which does not divice by {}"
                     <<  param_divider);

        std::vector<mlir::Value> operands = {input, hx[0], hx[1]};
        for(auto &param : params) {
            operands.push_back(param);
        }
        $_state.addOperands(operands);


        // Handle shape inference
        auto input_shape = getShape(input);

        bool batched = input_shape.size() == 3;

        size_t D = bidrectional ? 2 : 1;

        size_t N;
        size_t L;
        size_t H_out;
        size_t H_cell;

        if(batched) {
            // These checks should be already checked by PyTorch, so there are
            // no error messages;
            ERROR_ON(input_shape.size() != 3);
            ERROR_ON(h_n_shape.size() != 3);
            ERROR_ON(c_n_shape.size() != 3);

            if(batch_first) {
                N = input_shape[0];
                L = input_shape[1];
            } else {
                N = input_shape[1];
                L = input_shape[0];
            }

            ERROR_ON(h_n_shape[1] != N);
            ERROR_ON(c_n_shape[1] != N);
            H_out = h_n_shape[2];
            H_cell = c_n_shape[2];
        } else {
            L = input_shape[0];
            H_out = h_n_shape[1];
            H_cell = c_n_shape[1];
        }

        ERROR_ON(h_n_shape[0] != D*num_layers);
        ERROR_ON(c_n_shape[0] != D*num_layers);

        std::vector<int64_t> output_shape(input_shape.size(), L);
        if(batched) {
            if(batch_first) {
                output_shape[0] = N;
            } else {
                output_shape[1] = N;
            }
        }
        output_shape.back() = D*H_out;

        std::vector<int64_t> hy_shape(input_shape.size(), H_out);
        hy_shape[0] = D*num_layers;
        if(batched) {
            hy_shape[1] = N;
        }
        hy_shape.back() = H_out;

        std::vector<int64_t> cy_shape(input_shape.size(), H_cell);
        cy_shape[0] = D*num_layers;
        if(batched) {
            cy_shape[1] = N;
        }

        $_state.addTypes(
            {mlir::RankedTensorType::get(output_shape, input_type),
             mlir::RankedTensorType::get(hy_shape, input_type),
             mlir::RankedTensorType::get(cy_shape, input_type)});

        std::vector<std::int32_t> segments = {1, 2, params.size()};
        $_state.addAttribute("operand_segment_sizes", $_builder.getI32VectorAttr(segments));
        $_state.addAttribute("has_biases", $_builder.getBoolAttr(has_biases));
        $_state.addAttribute("num_layers", $_builder.getI64IntegerAttr(num_layers));
        $_state.addAttribute("dropout", $_builder.getF32FloatAttr(dropout));
        $_state.addAttribute("train", $_builder.getBoolAttr(train));
        $_state.addAttribute("bidrectional", $_builder.getBoolAttr(bidrectional));
        $_state.addAttribute("batch_first", $_builder.getBoolAttr(batch_first));

    }]>];
}

