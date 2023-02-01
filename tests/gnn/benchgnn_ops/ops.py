# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import List, Tuple

import torch


class ScatterOp(torch.nn.Module):
    def __init__(self, dim: int, input_shape: torch.Size,
                 index_shape: torch.Size, src_shape: torch.Size) -> None:
        """Scatter Op.

        Args:
            dim (int): the axis along which to index.
            input_shape (torch.Size): the scatter input tensor shape.
            index_shape (torch.Size): the indices shape of elements to scatter.
            src_shape (torch.Size): the source element(s) shape to scatter.
        """
        super().__init__()

        self.dim = dim
        input = torch.randn(*input_shape)
        index = torch.randint(input_shape[dim], index_shape)
        src = torch.randn(*src_shape)
        self.register_buffer('input', input)
        self.register_buffer('index', index)
        self.register_buffer('src', src)
        self.register_buffer('output', self(input, index, src, None)[-1])

    def loop_inputs(self) -> List[torch.Tensor]:
        return [self.input, self.index, self.src, self.output]

    def forward(
            self,
            input: torch.tensor,
            index: torch.tensor,
            src: torch.tensor,
            output: torch.tensor  # pylint: disable=unused-argument
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return input, index, src, torch.scatter(input, self.dim, index, src)


class ScatterReduceOp(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 input_shape: torch.Size,
                 index_shape: torch.Size,
                 src_shape: torch.Size,
                 reduce: str,
                 include_self: bool = True) -> None:
        """ScatterReduce Op.

        Args:
            dim (int): the axis along which to index.
            input_shape (torch.Size): the scatter input tensor shape.
            index_shape (torch.Size): the indices shape of elements to scatter.
            src_shape (torch.Size): the source element(s) shape to scatter.
            reduce (str): the reduction operation to apply for non-unique
                indices. ("sum", "prod", "mean", "amax", "amin")
            include_self (bool, optional): whether elements from the self
                tensor are included in the reduction. (default: :obj:`True`)
        """
        super().__init__()

        self.dim = dim
        self.reduce = reduce
        self.include_self = include_self
        input = torch.randn(*input_shape)
        index = torch.randint(input_shape[dim], index_shape)
        src = torch.randn(*src_shape)
        self.register_buffer('input', input)
        self.register_buffer('index', index)
        self.register_buffer('src', src)
        self.register_buffer('output', self(input, index, src, None)[-1])

    def loop_inputs(self) -> List[torch.Tensor]:
        return [self.input, self.index, self.src, self.output]

    def forward(
            self,
            input: torch.tensor,
            index: torch.tensor,
            src: torch.tensor,
            output: torch.tensor  # pylint: disable=unused-argument
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return input, index, src, input.scatter_reduce(
            self.dim, index, src, self.reduce, include_self=self.include_self)


class IndexReduceOp(torch.nn.Module):
    def __init__(self,
                 dim: int,
                 input_shape: torch.Size,
                 index_shape: torch.Size,
                 src_shape: torch.Size,
                 reduce: str,
                 include_self: bool = True) -> None:
        """IndexReduce Op.

        Args:
            dim: the axis along which to index.
            input_shape: the index reduce input tensor shape.
            index_shape: the indices shape of elements to select from.
            src_shape: the source element(s) shape.
            reduce: the reduction operation to apply ("prod",
                    "mean", "amax", "amin")
            include_self: whether elements from the self tensor are included in
                          the reduction
        """
        super().__init__()

        self.dim = dim
        self.reduce = reduce
        self.include_self = include_self
        input = torch.randn(*input_shape)
        index = torch.randint(input_shape[dim], index_shape)
        src = torch.randn(*src_shape)
        self.register_buffer('input', input)
        self.register_buffer('index', index)
        self.register_buffer('src', src)
        self.register_buffer('output', self(input, index, src, None)[-1])

    def loop_inputs(self) -> List[torch.Tensor]:
        return [self.input, self.index, self.src, self.output]

    def forward(
            self: torch.tensor,
            input: torch.tensor,
            index: torch.tensor,
            src: torch.tensor,
            output: torch.tensor  # pylint: disable=unused-argument
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return input, index, src, input.index_reduce_(
            self.dim, index, src, self.reduce, include_self=self.include_self)


class IndexSelectOp(torch.nn.Module):
    def __init__(self, dim: int, input_shape: torch.Size,
                 index_size: int) -> None:
        """IndexSelect Op.

        Args:
            dim: the axis along which to index.
            input_shape: the input tensor shape.
            index_size: the indices size.
        """

        super().__init__()
        self.dim = dim
        input = torch.randn(*input_shape)
        index = torch.randint(input_shape[dim], (index_size, ))
        self.register_buffer('input', input)
        self.register_buffer('index', index)
        self.register_buffer('output', self(input, index, None)[-1])

    def loop_inputs(self) -> List[torch.Tensor]:
        return [self.input, self.index, self.output]

    def forward(self, input: torch.tensor, index: torch.tensor, _: torch.tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return input, index, input.index_select(dim=self.dim, index=index)


class GatherOp(torch.nn.Module):
    def __init__(self, dim: int, input_shape: torch.Size,
                 index_shape: torch.Size) -> None:
        """Gather Op.

        Args:
            dim: the axis along which to index.
            input_shape: the scatter input tensor shape.
            index_shape: the indices shape of elements to gather.
        """

        super().__init__()
        self.dim = dim
        input = torch.randn(*input_shape)
        index = torch.randint(input_shape[dim], index_shape)
        self.register_buffer('input', input)
        self.register_buffer('index', index)
        self.register_buffer('output', self(input, index, None)[-1])

    def loop_inputs(self) -> List[torch.Tensor]:
        return [self.input, self.index, self.output]

    def forward(self: torch.tensor, input: torch.tensor, index: torch.tensor,
                _: torch.tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return input, index, torch.gather(input, self.dim, index)


bench_ops = {
    'gather': GatherOp,
    'index_reduce': IndexReduceOp,
    'index_select': IndexSelectOp,
    'scatter': ScatterOp,
    'scatter_reduce': ScatterReduceOp
}
