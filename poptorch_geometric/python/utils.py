# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import time
from copy import deepcopy

import torch
from torch.testing import assert_close
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import MessagePassing

import poptorch

DataBatch = type(Batch(_base_cls=Data))
HeteroDataBatch = type(Batch(_base_cls=HeteroData))


class AttributeTypeCache:
    """ A class for caching the result of ``Data.is_edge_attr`` and
    ``Data.is_node_attr`` from PyTorch Geometric
    """

    def __init__(self) -> None:
        self._attr_cache = {'node': set(), 'edge': set(), 'other': set()}

    def _put_key(self, data: Data, key: str):
        """insert into cache the result of calling:
            data.is_node_attr(key)
            data.is_edge_attr(key)
        """
        if data.is_node_attr(key):
            self._attr_cache['node'].add(key)
        elif data.is_edge_attr(key):
            self._attr_cache['edge'].add(key)
        else:
            self._attr_cache['other'].add(key)

    def is_node_attr(self, data: Data, key: str) -> bool:
        """cached lookup for data.is_node_attr(key)"""
        if key in self._attr_cache['node']:
            return True

        if key in self._attr_cache['edge'] or key in self._attr_cache['other']:
            return False

        self._put_key(data, key)
        return self.is_node_attr(data, key)

    def is_edge_attr(self, data: Data, key: str) -> bool:
        """cached lookup for data.is_edge_attr(key)"""
        if key in self._attr_cache['edge']:
            return True

        if key in self._attr_cache['node'] or key in self._attr_cache['other']:
            return False

        self._put_key(data, key)
        return self.is_edge_attr(data, key)


def set_aggregation_dim_size(model: torch.nn.Module, dim_size: int):
    """Sets the dim_size argument used in the aggregate step of message passing

        The dim_size will need to be at least as large as the total number of
        nodes in the batch.
    """

    def set_dim_size_hook(module, inputs):  # pylint: disable=unused-argument
        aggr_kwargs = inputs[-1]
        aggr_kwargs['dim_size'] = dim_size
        return aggr_kwargs

    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.register_aggregate_forward_pre_hook(set_dim_size_hook)


class TrainingStepper:
    """
    Test utility for comparing training runs between IPU and CPU.

    Usage:

        model = ...
        batch = ...
        model.train()
        stepper = TrainingSteper(model)
        stepper.run(10, batch)
    """

    def __init__(self,
                 model,
                 lr=0.001,
                 optimizer=poptorch.optim.Adam,
                 options=None,
                 rtol=None,
                 atol=None,
                 enable_fp_exception=True):
        super().__init__()
        model.train()
        self.lr = lr
        self.rtol = rtol
        self.atol = atol
        self.enable_fp_exception = enable_fp_exception
        self.options = poptorch.Options() if options is None else options
        self.setup_cpu(model, optimizer)
        self.setup_ipu(model, optimizer)
        self.check_parameters()

    def setup_cpu(self, model, optimizer):
        self.cpu_model = deepcopy(model)
        self.optimizer = optimizer(self.cpu_model.parameters(), lr=self.lr)

    def setup_ipu(self, model, optimizer):
        self.ipu_model = deepcopy(model)
        ipu_optimizer = optimizer(self.ipu_model.parameters(), lr=self.lr)
        options = self.options
        if self.enable_fp_exception:
            options.Precision.enableFloatingPointExceptions(True)
        self.training_model = poptorch.trainingModel(self.ipu_model,
                                                     optimizer=ipu_optimizer,
                                                     options=options)

    def check_parameters(self):
        for cpu, ipu in zip(self.cpu_model.named_parameters(),
                            self.ipu_model.named_parameters()):
            name, cpu = cpu
            ipu = ipu[1]
            self.assert_close(actual=ipu, expected=cpu, id=name)

    def cpu_step(self, batch):
        self.optimizer.zero_grad()
        out, loss = self.cpu_model(*batch)
        loss.backward()
        self.optimizer.step()
        return out, loss

    def ipu_step(self, batch, copy_weights=True):
        out, loss = self.training_model(*batch)
        if copy_weights:
            self.training_model.copyWeightsToHost()
        return out, loss

    def run(self, num_steps, batch):
        cpu_loss = torch.empty(num_steps)
        ipu_loss = torch.empty(num_steps)

        for i in range(num_steps):
            cpu_out, cpu_loss[i] = self.cpu_step(batch)
            ipu_out, ipu_loss[i] = self.ipu_step(batch)
            self.assert_close(actual=ipu_out, expected=cpu_out, id="Output")
            self.check_parameters()

        self.assert_close(actual=ipu_loss, expected=cpu_loss, id="loss")

    def assert_close(self, actual, expected, id):
        def msg_fn(msg):
            return f"{id} was not equal:\n\n{msg}\n"

        assert_close(actual=actual,
                     expected=expected,
                     msg=msg_fn,
                     rtol=self.rtol,
                     atol=self.atol)

    def benchmark(self, num_steps, batch, devices=('ipu')):
        results = {}
        if 'ipu' in devices:
            _, _ = self.ipu_step(batch, copy_weights=False)
            t_start = time.perf_counter()
            for _ in range(num_steps):
                _, _ = self.ipu_step(batch, copy_weights=False)
            t_end = time.perf_counter()
            results['ipu_time'] = t_end - t_start
        if 'cpu' in devices:
            _, _ = self.cpu_step(batch)
            t_start_cpu = time.perf_counter()
            for _ in range(num_steps):
                _, _ = self.cpu_step(batch)
            t_end_cpu = time.perf_counter()
            results['cpu_time'] = t_end_cpu - t_start_cpu
        if 'gpu' in devices:
            results['gpu_time'] = None
            raise NotImplementedError('GPU benchmarking currently unsupported')
        return results
