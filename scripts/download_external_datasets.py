#!/usr/bin/env python3
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import argparse
import os.path as osp
import torch_geometric as pyg

parser = argparse.ArgumentParser(description="Download external datasets")
parser.add_argument(
    "external_datasets_dir",
    help="The directory where the external datasets will be downloaded.")

args = parser.parse_args()

pyg.datasets.QM9(root=osp.join(args.external_datasets_dir, "qm9"))
pyg.datasets.Planetoid(osp.join(args.external_datasets_dir, "planetoid"),
                       "Cora")
