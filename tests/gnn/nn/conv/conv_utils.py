# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import torch
from torch_geometric import seed_everything
from torch_geometric.data import HeteroData

from poptorch_geometric import TrainingStepper


def conv_harness(conv,
                 dataset=None,
                 post_proc=None,
                 loss_fn=torch.nn.MSELoss(),
                 num_steps=4,
                 atol=1e-5,
                 rtol=1e-4,
                 batch=None,
                 training=True):
    class ConvWrapper(torch.nn.Module):
        def __init__(self, conv, loss_fn, post_proc=None):
            super().__init__()
            self.conv = conv
            self.loss_fn = loss_fn
            self.post_proc = post_proc

        def forward(self, *args):
            x = self.conv(*args)
            if self.post_proc is not None:
                x = self.post_proc(x)

            if isinstance(x, tuple):
                x = x[0]

            if self.training:
                target = torch.ones_like(x)
                loss = self.loss_fn(x, target)
                return x, loss

            return x

    model = ConvWrapper(conv, loss_fn=loss_fn, post_proc=post_proc)

    if batch is None and dataset is not None:
        batch = (dataset.x, dataset.edge_index)

    stepper = TrainingStepper(model, atol=atol, rtol=rtol)
    if training:
        stepper.run(num_steps, batch)
    else:
        stepper.run_inference(batch)


def generate_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def random_heterodata(in_channels=None):
    seed_everything(0)

    if in_channels is None:
        in_channels = {'author': 16, 'paper': 12, 'term': 3}

    data = HeteroData()
    data['author'].x = torch.randn(6, in_channels['author'])
    data['paper'].x = torch.randn(5, in_channels['paper'])
    data['term'].x = torch.randn(4, in_channels['term'])

    data[('author', 'author')].edge_index = generate_edge_index(6, 6, 15)
    data[('author', 'paper')].edge_index = generate_edge_index(6, 5, 10)
    data[('paper', 'term')].edge_index = generate_edge_index(5, 4, 8)
    return data, in_channels


def hetero_conv_harness(conv,
                        data,
                        output_key,
                        forward_args=None,
                        loss_fn=torch.nn.MSELoss(),
                        num_steps=4,
                        atol=1e-3,
                        rtol=1e-2,
                        enable_fp_exception=True):

    if forward_args is None:
        forward_args = ['x_dict', 'edge_index_dict']

    class ConvWrapper(torch.nn.Module):
        def __init__(self, conv, loss_fn):
            super().__init__()
            self.conv = conv
            self.loss_fn = loss_fn

        def forward(self, *args):
            out = self.conv(*args)
            out = out[output_key]
            if self.training:
                target = torch.ones_like(out)
                loss = self.loss_fn(out, target)
                return out, loss
            return out

    model = ConvWrapper(conv, loss_fn)

    stepper = TrainingStepper(model,
                              atol=atol,
                              rtol=rtol,
                              enable_fp_exception=enable_fp_exception)
    inputs = [getattr(data, f_arg) for f_arg in forward_args]
    stepper.run(num_steps, inputs)
