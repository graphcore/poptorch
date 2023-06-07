# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import List
import torch
import torch_geometric

from poptorch_geometric import TrainingStepper


def aggr_harness(aggr,
                 dim_size,
                 dataloader=None,
                 post_proc=None,
                 sorted_index=False,
                 loss_fn=torch.nn.MSELoss(),
                 num_steps=4,
                 atol=5e-3,
                 rtol=5e-3,
                 equal_nan=False,
                 enable_fp_exception=True):
    class AggrWrapper(torch.nn.Module):
        def __init__(self, aggr, loss_fn, post_proc=None):
            assert hasattr(loss_fn, 'reduction')
            # No support for other reduction types yet
            assert loss_fn.reduction in ('sum', 'mean')

            super().__init__()
            self.aggr = aggr
            self.loss_fn = loss_fn
            self.post_proc = post_proc
            self.mean_reduction_in_loss = (loss_fn.reduction == 'mean')

        def forward(self, *args):
            x = args[0]
            edge_index = args[1]
            nodes_mask = args[2]
            size = args[3]

            broadcast_index = edge_index[1] if sorted_index else edge_index[0]
            aggr_index = edge_index[0] if sorted_index else edge_index[1]

            x_broadcasted = torch.index_select(x, 0, broadcast_index)
            kwargs = {}
            if isinstance(self.aggr,
                          (torch_geometric.nn.aggr.SortAggregation,
                           torch_geometric.nn.aggr.GRUAggregation,
                           torch_geometric.nn.aggr.GraphMultisetTransformer,
                           torch_geometric.nn.aggr.SetTransformerAggregation,
                           torch_geometric.nn.aggr.LSTMAggregation)):
                kwargs["max_num_elements"] = size

            result = self.aggr(x_broadcasted,
                               aggr_index,
                               dim_size=size,
                               **kwargs)

            if self.post_proc is not None:
                if isinstance(result, List):
                    nodes_mask = nodes_mask.repeat(len(result))
                    result = torch.cat(result)
                result = self.post_proc(result)
                # Apply nodes mask, so that the loss may be computed properly
                result[~nodes_mask] = 0

            if self.training:
                if isinstance(result, List):
                    result = torch.cat(result)
                target = torch.ones_like(result)
                target[~nodes_mask] = 0

                loss = self.loss_fn(result, target)
                # In case, the loss function applies mean reduction, the result
                # has to be rescaled by the effective size of the batch
                # (excluding padding).
                if self.mean_reduction_in_loss:
                    real_size = torch.count_nonzero(nodes_mask)
                    loss = loss * size / real_size

                return result, loss

            return result

    model = AggrWrapper(aggr, loss_fn=loss_fn, post_proc=post_proc)

    stepper = TrainingStepper(model,
                              atol=atol,
                              rtol=rtol,
                              equal_nan=equal_nan,
                              enable_fp_exception=enable_fp_exception)

    if dataloader is not None:
        for step, batch in enumerate(dataloader):
            if step == num_steps:
                break

            stepper.run(
                1, (batch.x, batch.edge_index, batch.nodes_mask, dim_size))
