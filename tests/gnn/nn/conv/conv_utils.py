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
                 batch=None):
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

    stepper.run(num_steps, batch)


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


class HeteroDataTupleConverter:
    def __init__(self, data):
        self.type_list = []
        if not isinstance(data, dict):
            data = data.to_dict()

        for k1 in data.keys():
            data_k1 = data[k1]
            if not isinstance(data[k1], dict):
                data_k1 = data[k1].to_dict()
            for k2 in data_k1.keys():
                self.type_list.append((k1, k2))

    def to_tuple(self, data):
        items = []
        for k1, k2 in self.type_list:
            item = data[k1][k2]
            if not isinstance(item, torch.Tensor):
                item = torch.tensor(item)
                if item.dim() == 0:
                    item = item.unsqueeze(0)

            items.append(item)

        return tuple(items)

    def from_tuple(self, tensors):
        data = HeteroData()
        for (k1, k2), tensor in zip(self.type_list, tensors):
            data[k1][k2] = tensor

        return data


def hetero_conv_harness(
        conv,
        data,
        output_key,
        forward_args=None,
        loss_fn=torch.nn.MSELoss(),
        num_steps=4,
        atol=1e-3,
        rtol=1e-2,
):

    if forward_args is None:
        forward_args = ['x_dict', 'edge_index_dict']

    class ConvWrapper(torch.nn.Module):
        def __init__(self, conv, converter, loss_fn):
            super().__init__()
            self.conv = conv
            self.converter = converter
            self.loss_fn = loss_fn

        def forward(self, *args):
            data = self.converter.from_tuple(args)
            inputs = (getattr(data, x) for x in forward_args)
            out = self.conv(*inputs)

            out = out[output_key]
            if self.training:
                target = torch.ones_like(out)
                loss = self.loss_fn(out, target)
                return out, loss
            return out

    converter = HeteroDataTupleConverter(data)
    model = ConvWrapper(conv, converter, loss_fn)

    stepper = TrainingStepper(model, atol=atol, rtol=rtol)
    stepper.run(num_steps, converter.to_tuple(data))
