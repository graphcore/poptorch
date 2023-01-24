#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import argparse
import os.path as osp
import torch_geometric as pyg

parser = argparse.ArgumentParser(description="Download external datasets")
parser.add_argument("test_dir", help="Path to the folder containing the tests")

args = parser.parse_args()

qm9root = osp.join(args.test_dir, "gnn", ".datasets", "qm9")
pyg.datasets.QM9(root=qm9root)
