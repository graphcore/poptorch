# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import functools
import random
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import nbformat
import torch  # noqa F401
from nbconvert.preprocessors import ExecutePreprocessor
from torch.testing import assert_close
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.data.data import BaseData

assert_equal = functools.partial(assert_close, rtol=0., atol=0.)
DEFAULT_PROCESS_TIMEOUT_SECONDS = 40 * 60
REPO_ROOT = Path(__file__).resolve().parents[1]


def is_data(type_: BaseData):
    if type_ is Data:
        return True
    if type_ is HeteroData:
        return False
    raise f"Wrong data type: {type_}. Should be Data or HeteroData!"


class CalledProcessError(subprocess.CalledProcessError):
    """An error for subprocesses which captures stdout and stderr in the error
    message."""

    def __str__(self) -> str:
        return "{original_message}\n{stdout}\n{stderr}".format(
            original_message=super().__str__(),
            stdout=self.stdout,
            stderr=self.stderr)


def run_command_fail_explicitly(command: Union[str, List[str]], cwd: str,
                                **kwargs) -> str:
    """ Runs a command returning the output or failing with useful information
    Args:
        command: The command to execute, can also be a space separated string.
        cwd: The directory in which the command should be
            launched. If called by a pytest test function or method, this
            probably should be a `tmp_path` fixture.
        **kwargs: Additional keyword arguments are passed to
            `subprocess.check_output`.
    Returns:
        The standard output and error of the command if successfully executed.
    Raises:
        RuntimeError: If the subprocess command executes with a non-zero
            output.
    """
    DEFAULT_KWARGS = {
        "shell": isinstance(command, str) and " " in command,
        "stderr": subprocess.PIPE,
        "timeout": DEFAULT_PROCESS_TIMEOUT_SECONDS,
        "universal_newlines": True,
    }

    try:
        merged_kwargs = {**DEFAULT_KWARGS, **kwargs}
        out = subprocess.check_output(
            command,
            cwd=cwd,
            **merged_kwargs,
        )
    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr
        # type of the stdout stream will depend on the subprocess.
        # The python docs say decoding is to be handled at
        # application level.
        if hasattr(stdout, "decode"):
            stdout = stdout.decode("utf-8", errors="ignore")
        if hasattr(stderr, "decode"):
            stderr = stderr.decode("utf-8", errors="ignore")
        raise CalledProcessError(1, cmd=command, output=stdout,
                                 stderr=stderr) from e
    return out


class ExpectedError(Exception):
    """An error which is expected by the test suite, to be used
    when decorating tests:

        @pytest.mark.xfail(raises=ExpectedError)
        def test_something_that_needs_fixing():
            try:
                broken_fun()
            except Exception as e:
                # check that e matches a condition
                if check_cond(e):
                    raise ExpectedError("") from e
                raise  # otherwise raise the original unexpected error
    """


def run_notebook(notebook_filename, expected_error: str = "", cwd=REPO_ROOT):
    """helper to run notebooks which may or may not be expected to fail"""
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": f"{cwd}"}})
    except Exception as e:
        if (not expected_error) or (expected_error not in str(e)):
            raise
        raise ExpectedError(expected_error) from e


class FakeDatasetEqualGraphs(InMemoryDataset):  #pylint: disable=abstract-method
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects with fixed graph size.

    Args:
        num_graphs (int): The number of graphs.
        num_nodes (int): The number of nodes in a graph.
        num_channels (int): The number of node features.
        edge_dim (int): The number of edge features.
        num_edges (int, optional): The number of edges in a graph.
            (default: :obj:`None`)
    """

    def __init__(self,
                 num_graphs: int,
                 num_nodes: int,
                 num_channels: int,
                 edge_dim: int,
                 num_edges: Optional[int] = None) -> None:
        if num_graphs < 1:
            raise RuntimeError("Can't create dataset with less than 1 graph.")

        super().__init__('.')

        self.num_nodes = num_nodes
        if num_edges is not None:
            self.num_edges = num_edges
        else:
            # Randomize number of edges in graph.
            self.num_edges = random.randint(num_nodes + 1,
                                            num_nodes * (num_nodes - 1))
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        data_list = [self.generate_data() for _ in range(num_graphs)]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        x = torch.rand(self.num_nodes, self.num_channels)
        edge_index = torch.randint(high=self.num_nodes,
                                   size=(2, self.num_edges))
        edge_attr = torch.rand(self.num_edges, self.edge_dim)

        # -100 is the default value of `ignore_index` in `nn.CrossEntropyLoss`.
        y = torch.tensor([-100]).long()

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
