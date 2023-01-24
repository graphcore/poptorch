# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numbers
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from . import utils

try:
    from functools import singledispatchmethod
except ImportError:
    # Workaround for systems with Python 3.7
    from functools import singledispatch, update_wrapper

    def singledispatchmethod(func):
        dispatcher = singledispatch(func)

        def wrapper(*arg, **kw):
            return dispatcher.dispatch(arg[1].__class__)(*arg, **kw)

        wrapper.register = dispatcher.register
        update_wrapper(wrapper, func)
        return wrapper


__all__ = ['Pad']

# Note: this code is being re-organized
# pylint: disable=protected-access
# pylint: disable=too-many-nested-blocks


class Pad(BaseTransform):
    r"""Applies padding to enforce consistent tensor shapes.

    This transform will pad node and edge features up to a maximum allowed
    size in the node or edge feature dimension. By default :obj:`0.0` is used
    as the padding value and can be configured by setting :obj:`node_pad_value`
    and :obj:`edge_pad_value`.

    Note that in case of applying Pad to
    :class:`~HeteroData` input, the :obj:`node_pad_value`
    or :obj:`edge_pad_value` can be a single float or a Dict[<node type>,
    <pad value>]. In the latter case, if the user does not include a particular
    node type in the dict, this node type's attributes will not get padded. In
    contrast, if the user does not include a particular edge type in the dict,
    for which the destination node padding was specified, the default edge
    padding value is used.

    Note that in order to allow for at least one padding node for any padding
    edge, below conditions must be met:

    * if :obj:`max_num_nodes` is a single value, it must be greater than the
    maximum number of nodes of any graph in the dataset;
    * if :obj:`max_num_nodes` is a dictionary, value for every node type must
    be greater than the maximum number of this type nodes of any graph in the
    dataset.

    Args:
        max_num_nodes (int or Dict[str, int]): The maximum number of nodes.
            In heterogeneous graphs, may also take in a dictionary denoting the
            maximum number of nodes for specific node types.
        max_num_edges (int or Dict[Tuple[str, str, str], int], optional):
            The maximum number of edges.
            In heterogeneous graphs, may also take in a dictionary denoting the
            maximum number of edges for specific edge types.
        node_pad_value (float or Dict[str, float], optional): The fill value to
            use for node features (default: :obj:`0.0`).
            In heterogeneous graphs, may also take in a dictionary denoting the
            fill value for specific node types.
        edge_pad_value (float or Dict[Tuple[str, str, str], float], optional):
            The fill value to use for edge features (default: :obj:`0.0`).
            In heterogeneous graphs, may also take in a dictionary denoting the
            fill value for specific edge types.
            Note that in case of :obj:`edge_index` feature the tensors are
            padded with the first padded node index (which represents a set of
            self loops on the padded node).
        train_mask_pad_value (bool, optional): The fill value to use for
            :obj:`train_mask` (default: :obj:`False`).
        test_mask_pad_value (bool, optional): The fill value to use for
            :obj:`test_mask` (default: :obj:`False`).
        include_keys (List[str] or Tuple[str], optional): Keys to include from
            the input data object to apply the padding transformation to.
    """

    def __init__(
            self,
            max_num_nodes: Union[int, Dict[str, int]],
            max_num_edges: Optional[
                Union[int, Dict[Tuple[str, str, str], int]]] = None,
            node_pad_value: Optional[Union[float, Dict[str, float]]] = None,
            edge_pad_value: Optional[
                Union[float, Dict[Tuple[str, str, str], float]]] = None,
            train_mask_pad_value: Optional[bool] = None,
            test_mask_pad_value: Optional[bool] = None,
            include_keys: Optional[Union[List[str], Tuple[str]]] = None):
        super().__init__()

        self._default_pad_value = 0.0
        self.max_num_nodes = self.ParamWithDesc(max_num_nodes)
        self._set_max_num_edges(max_num_nodes, max_num_edges, include_keys)

        node_pad_value = (self._default_pad_value
                          if node_pad_value is None else node_pad_value)
        edge_pad_value = (self._default_pad_value
                          if edge_pad_value is None else edge_pad_value)
        self.node_pad = self.ParamWithDesc(node_pad_value)
        self.edge_pad = self.ParamWithDesc(edge_pad_value)

        self.node_additional_attrs_pad = {
            'train_mask':
            0.0 if train_mask_pad_value is None else train_mask_pad_value,
            'test_mask':
            0.0 if test_mask_pad_value is None else test_mask_pad_value
        }

        self.include_keys = include_keys
        self.attribute_cacher = utils.AttributeTypeCache()

    class ParamWithDesc():
        r"""Wraps a parameter and caches the information whether the field is
        of the number type.

        Args:
            param (object): The parameter to be wrapped.
        """

        def __init__(self, param):
            self.value = param
            self.is_number = isinstance(param, numbers.Number)

    def _set_max_num_edges(
            self, max_num_nodes: Union[int, Dict[str, int]],
            max_num_edges: Union[int, Dict[Tuple[str, str, str], int]],
            include_keys: Union[List[str], Tuple[str]]):
        r"""Sets :obj:`max_num_edges` field based on the provided parameters.

        The field is stored in one of the following forms:
        1. :obj:`ParamWithDesc(None)` - no padding added to edges;
        2. :obj:`ParamWithDesc(int)` - all edges padded to constant size (can
           be both specified by the user or auto computed, based on nodes
           padding);
        3. :obj:`ParamWithDesc(Dict[(src_node, dst_node), ParamWithDesc(int)])`
           - edges from :obj:`src_node` to :obj:`dst_node` of any edge type get
           padded to the specified size;
        4. :obj:`ParamWithDesc(Dict[(src_node, dst_node),
                                    ParamWithDesc(Dict[edge_type, int])])` -
           edges from :obj:`src_node` to :obj:`dst_node` of the specified edge
           type get padded to the specified size.
        """

        def _has_items_method(obj):
            return callable(getattr(obj, 'items', None))

        if include_keys is not None and 'x' not in include_keys:
            # No nodes are going to be added, so no edges are to be padded.
            # Option 1 - ParamWithDesc(None)
            self.max_num_edges = self.ParamWithDesc(None)
        else:
            if max_num_edges is not None:
                self.max_num_edges = self.ParamWithDesc(max_num_edges)
                if self.max_num_edges.is_number:
                    # Option 2 - ParamWithDesc(int)
                    return

                # Option 4 - ParamWithDesc(Dict[(src_node, dst_node),
                #                               ParamWithDesc(
                #                                   Dict[edge_type, int]
                #                               )])
                assert _has_items_method(max_num_edges), \
                    'max_num_edges param has to be of type int or '\
                    'Dict[str, str, str].'
                self.max_num_edges = self.ParamWithDesc(defaultdict())
                for key, val in max_num_edges.items():
                    src_node, edge_type, dst_node = key
                    self.max_num_edges.value[src_node,
                                             dst_node] = self.ParamWithDesc(
                                                 {edge_type: val})
            else:
                if _has_items_method(max_num_nodes):
                    # Option 3 - ParamWithDesc(Dict[(src_node, dst_node),
                    #                               ParamWithDesc(int)])
                    self.max_num_edges = self.ParamWithDesc(defaultdict())
                    for src_key, src_val in max_num_nodes.items():
                        for dst_key, dst_val in max_num_nodes.items():
                            self.max_num_edges.value[
                                src_key, dst_key] = self.ParamWithDesc(
                                    src_val * dst_val)
                else:
                    # Option 2 - ParamWithDesc(int)
                    # Assume fully connected graph
                    self.max_num_edges = self.ParamWithDesc(max_num_nodes *
                                                            max_num_nodes)

    @singledispatchmethod
    def validate(self, data):
        raise ValueError(f'Unsupported data type: {type(data)}')

    @validate.register
    def _(self, data: Data):
        r"""Validates that the input graph does not exceed the constraints
        that:

          * the number of :obj:`nodes + 1` must be <= :obj:`max_num_nodes`;
          * the number of edges must be <= :obj:`max_num_edges`.

        Args:
            data (Data): The data to be validated.
        """
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        def assert_numeric_type(wrapped_param: self.ParamWithDesc,
                                param_name: str):
            assert wrapped_param.is_number, \
                f'If the type of the data argument is '\
                f'Data, {param_name} has to be of integer type but is '\
                f'{type(wrapped_param.value)}.'

        assert_numeric_type(self.max_num_nodes, 'max_num_nodes')

        assert num_nodes + 1 <= self.max_num_nodes.value, \
            f'Too many nodes. Graph has {num_nodes} nodes '\
            f'and requested max_num_nodes of {self.max_num_nodes.value} is '\
            'not large enough to accommodate padding edges. The '\
            'max_num_nodes option must be at least as large as\n\n' \
            '    max(data.num_nodes) + 1\n\n' \
            'for all examples in the dataset so that there is at least one '\
            'padding node to route padding edges through.'

        if self.max_num_edges.value is not None:
            assert_numeric_type(self.edge_pad, 'edge_pad')
            assert_numeric_type(self.max_num_edges, 'max_num_edges')

            assert num_edges <= self.max_num_edges.value, \
                f'Too many edges. Graph has {num_edges} edges defined '\
                f'and max_num_edges is {self.max_num_edges.value}. ' \
                'The max_num_edges option must be at least as large as\n\n' \
                '    max(data.num_edges)\n\n' \
                'for all examples in the dataset.'

        def validate_mask(data, attr_name):
            if hasattr(data, attr_name):
                mask = getattr(data, attr_name)
                pad_dim = data.__cat_dim__(attr_name, mask)
                pad_dim_val = mask.shape[pad_dim]
                assert pad_dim_val == num_nodes, \
                    f'train_mask padding dim {pad_dim}, value {pad_dim_val} ' \
                    f'is not equal to num nodes {num_nodes}'

        validate_mask(data, 'train_mask')
        validate_mask(data, 'test_mask')

    @validate.register
    def _(self, data: HeteroData):
        r"""Validates that the input graph does not exceed the constraints
        that:

          * the number of :obj:`nodes + 1` must be <= :obj:`max_num_nodes`;
          * the number of edges must be <= :obj:`max_num_edges`.

        Args:
            data (HeteroData): The data to be validated.
        """
        for type, _, value in self._nodes_gen(data):
            if not torch.is_tensor(value):
                continue

            # The nodes are stored in a form of tensor of shape
            # ['num of nodes', ... ], so we pick dim = 0.
            num_nodes = value.shape[0]

            if self.max_num_nodes.is_number:
                max_num_nodes = self.max_num_nodes.value
            else:
                if not type in self.max_num_nodes.value:
                    # The type is NOT included in the user's input, which means
                    # the user decided NOT to pad the node type.
                    continue
                max_num_nodes = self.max_num_nodes.value[type]

            assert num_nodes + 1 <= max_num_nodes, \
                f'Too many nodes. Graph {type} has {num_nodes} nodes '\
                f'and requested max_num_nodes of {self.max_num_nodes.value} '\
                'is not large enough to accommodate padding edges. The '\
                'max_num_nodes option must be at least as large as\n\n' \
                '    max(data.num_nodes) + 1\n\n' \
                'for all examples in the dataset so that there is at least '\
                'one padding node to route padding edges through.'

        if self.max_num_edges.value is not None:
            for type, _, value in self._edges_gen(data):
                if not torch.is_tensor(value):
                    continue

                src_node, edge_type, dst_node = type

                # The edges are stored in a form of tensor of shape
                # [2, 'num of edges'], so we pick dim = 1.
                num_edges = value.shape[1]

                if self.max_num_edges.is_number:
                    max_num_edges = self.max_num_edges.value
                else:
                    type_key = (src_node, dst_node)
                    if type_key in self.max_num_edges.value:
                        if self.max_num_edges.value[type_key].is_number:
                            max_num_edges = self.max_num_edges.value[
                                type_key].value
                        else:
                            if edge_type in self.max_num_edges.value[
                                    type_key].value:
                                max_num_edges = self.max_num_edges.value[
                                    type_key].value[edge_type]
                            else:
                                # The type (src_node, edge_type, dst_node)
                                # is NOT included in the user's input, which
                                # means the user decided NOT to pad the
                                # edge type.
                                continue
                    else:
                        # The type (src_node, edge_type, dst_node) is NOT
                        # included in the user's input, which means the user
                        # decided NOT to pad the edge type.
                        continue

                assert num_edges <= max_num_edges, \
                    f'Too many edges. Graph has {num_edges} edges of type '\
                    f'{type} defined and max_num_edges is '\
                    f'{self.max_num_edges.value}. ' \
                    'The max_num_edges option must be at least as large '\
                    'as\n\n' \
                    '    max(data.num_edges)\n\n' \
                    'for all examples in the dataset.'

    def __get_pad_value(self, pad, key) -> float:
        if pad.is_number:
            return pad.value

        if key in pad.value:
            return pad.value[key]

        # If the user does not specify the padding value, the
        # default one is used.
        return self._default_pad_value

    @singledispatchmethod
    def __call__(self, data):
        raise ValueError(f'Unsupported data type: {type(data)}')

    @__call__.register
    def _(self, data: Data) -> Data:
        self.validate(data)
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        num_pad_nodes = self.max_num_nodes.value - num_nodes
        if self.max_num_edges.value is not None:
            num_pad_edges = self.max_num_edges.value - num_edges

        new_data = {}
        args = () if self.include_keys is None else self.include_keys

        for key, value in data(*args):
            if not torch.is_tensor(value):
                continue

            is_additional_node_attr_pad = key in self.node_additional_attrs_pad

            if self.attribute_cacher.is_node_attr(
                    data, key) or is_additional_node_attr_pad:
                length = num_pad_nodes
                node_pad_val = self.__get_pad_value(self.node_pad, key)
                pad_value = self.node_additional_attrs_pad.get(
                    key, node_pad_val)

            elif self.attribute_cacher.is_edge_attr(data, key):
                if self.max_num_edges.value is None:
                    continue
                length = num_pad_edges
                if key == 'edge_index':
                    # Padding edges are self-loops on the first padding node.
                    pad_value = num_nodes
                else:
                    pad_value = self.edge_pad.value
            else:
                continue

            dim = data.__cat_dim__(key, value)
            new_data[key] = self._padInDim(value, dim, length, pad_value)

        for key, new_value in new_data.items():
            data[key] = new_value

        # Set num_nodes as recommended in data.num_nodes property definition.
        data.num_nodes = self.max_num_nodes.value
        return data

    def _edges_gen(self,
                   data: HeteroData) -> Generator[str, str, torch.tensor]:
        for type_key, type_values in data._edge_store_dict.items():
            for feature_key, feature_val in type_values.items():
                yield type_key, feature_key, feature_val

    def _nodes_gen(self,
                   data: HeteroData) -> Generator[str, str, torch.tensor]:
        cond = (lambda _: True) if self.include_keys is None else (
            lambda key: key in self.include_keys)

        for type_key, type_values in data._node_store_dict.items():
            for feature_key, feature_val in type_values.items():
                if cond(feature_key):
                    yield type_key, feature_key, feature_val

    @__call__.register
    def _(self, data: HeteroData) -> HeteroData:
        self.validate(data)

        # Iterate over edges.
        if self.max_num_edges.value is not None:
            for type, key, value in self._edges_gen(data):
                if not torch.is_tensor(value):
                    continue

                # The edges are stored in a form of tensor of shape
                # [2, 'num of edges'], so we pick dim = 1.
                num_edges = value.shape[1]
                if self.max_num_edges.is_number:
                    num_pad_edges = self.max_num_edges.value - num_edges
                else:
                    src_node, edge_type, dst_node = type
                    type_key = (src_node, dst_node)
                    if type_key in self.max_num_edges.value:
                        if self.max_num_edges.value[type_key].is_number:
                            num_pad_edges = self.max_num_edges.value[
                                type_key].value - num_edges
                        else:
                            if edge_type in self.max_num_edges.value[
                                    type_key].value:
                                num_pad_edges = self.max_num_edges.value[
                                    type_key].value[edge_type] - num_edges
                            else:
                                # The user did not specify the current edge
                                # for padding.
                                continue
                    else:
                        # The user did not specify the current edge for
                        # padding.
                        continue

                if key == 'edge_index':
                    # Padding edges are first padded src to first padded
                    # dst node.
                    src_node_type = type[0]
                    src_pad_value = data[src_node_type].num_nodes
                    dst_node_type = type[2]
                    dst_pad_value = data[dst_node_type].num_nodes

                    data._edge_store_dict[type][key] = self._padEdgeIndex(
                        value, num_pad_edges, src_pad_value, dst_pad_value)
                else:
                    pad_value = self.__get_pad_value(self.edge_pad, type)
                    dim = data.__cat_dim__(key, value)
                    data._edge_store_dict[type][key] = self._padInDim(
                        value, dim, num_pad_edges, pad_value)

        # Iterate over nodes.
        for type, key, value in self._nodes_gen(data):
            if not torch.is_tensor(value):
                continue

            num_nodes = value.shape[0]
            if self.max_num_nodes.is_number:
                num_pad_nodes = self.max_num_nodes.value - num_nodes
            elif type in self.max_num_nodes.value.keys():
                num_pad_nodes = self.max_num_nodes.value[type] - num_nodes
            else:
                continue

            length = num_pad_nodes

            pad_value = self.__get_pad_value(self.node_pad, type)

            dim = data.__cat_dim__(key, value)
            data._node_store_dict[type][key] = self._padInDim(
                value, dim, length, pad_value)

        return data

    @staticmethod
    def _padInDim(input: torch.Tensor, dim: int, length: int,
                  pad_value: float) -> torch.Tensor:
        r"""Pads the input tensor in the specified dim with a constant value of
        the given length.
        """
        pads = [0] * (2 * input.ndim)
        pads[-2 * dim - 1] = length
        return F.pad(input, pads, 'constant', pad_value)

    @staticmethod
    def _padEdgeIndex(input: torch.Tensor, length: int, src_pad_value: float,
                      dst_pad_value: float) -> torch.Tensor:
        r"""Pads the edges :obj:`edge_index` feature with values specified
        separately for src and dst nodes.
        """
        pads = [0, length, 0, 0]
        padded = F.pad(input, pads, 'constant', src_pad_value)
        if src_pad_value != dst_pad_value:
            padded[1, input.shape[1]:] = dst_pad_value
        return padded

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}('
        s += f'max_num_nodes={self.max_num_nodes.value}, '
        s += f'max_num_edges={self.max_num_edges.value}, '
        s += f'node_pad_value={self.node_pad.value}, '
        s += f'edge_pad_value={self.edge_pad.value})'
        return s
