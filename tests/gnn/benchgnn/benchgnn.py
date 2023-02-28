# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import argparse
import json
import os.path as osp
import sys
from collections import namedtuple
from itertools import product, starmap
from warnings import warn

import torch
from datasets import DataSets
from models import GAT, GCN, GIN, PNA, RGCN, SAGE
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv
from torch_geometric.transforms import Pad
from utils import all_formats, merge_results, print_results

import poptorch
from poptorch_geometric import TrainingStepper, set_aggregation_dim_size
from poptorch_geometric.dataloader import DataLoader as IPUDataLoader, FixedSizeDataLoader

supported_sets = {
    'Cora': [GCN, GAT, GIN, PNA, SAGE],
    'CiteSeer': [GCN, GAT, GIN, PNA, SAGE],
    'PubMed': [GCN, GAT, GIN, PNA, SAGE],
    'mutag': [RGCN],
    'FakeDataset': [GCN, GAT, GIN, PNA, SAGE],
}

all_models = list(set(m.__name__ for v in supported_sets.values() for m in v))
all_datasets = list(supported_sets.keys())
all_loaders = ['torch', 'poptorch', 'poptorch_fixed_size']
all_transforms = [None, 'Pad']

Config = namedtuple('Config', ['Model', 'ds', 'bs', 'loader', 'transform'])


def run_benchmark(args, configs):
    ipu_opts = poptorch.Options()
    if args['synthetic_data']:
        ipu_opts.enableSyntheticData(True)

    results = []
    for cfg in configs:
        if cfg.transform == 'Pad':
            max_num_nodes = args['max_num_nodes']
            max_num_edges = args['max_num_edges']
            assert max_num_nodes is not None and max_num_edges is not None

            cfg.ds.transform = Pad(max_num_nodes=max_num_nodes,
                                   max_num_edges=max_num_edges)
        if cfg.loader == 'torch':
            loader = DataLoader(cfg.ds, batch_size=cfg.bs, shuffle=False)
        elif cfg.loader == 'poptorch':
            loader = IPUDataLoader(cfg.ds, batch_size=cfg.bs)
        else:
            loader = FixedSizeDataLoader(dataset=cfg.ds,
                                         num_nodes=cfg.ds[0].num_nodes,
                                         batch_size=cfg.bs)

        d = next(iter(loader))

        params = {'loss_fn': torch.nn.MSELoss()}

        if cfg.Model.__name__ != 'GIN':
            params['out_channels'] = cfg.ds.num_classes

        if cfg.Model.__name__ == 'PNA':
            params['degree'] = PNAConv.get_degree_histogram(loader)

        if cfg.Model.__name__ == 'RGCN':
            batch = (d.edge_index, d.edge_type)
            params['in_channels'] = d.num_nodes
            params['num_relations'] = cfg.ds.num_relations
        else:
            batch = (d.x, d.edge_index)
            params['disable_dropout'] = args['check_values']
            params['in_channels'] = cfg.ds.num_features

        model = cfg.Model(**params)

        set_aggregation_dim_size(model, int(d.edge_index.max()) + 1)

        stepper = TrainingStepper(model,
                                  options=ipu_opts,
                                  enable_fp_exception=False)

        if args['check_values']:
            warn(
                'Models run without dropout layers. Turn off '
                'check-values to run the full model.', UserWarning)
            stepper.run(4, batch)

        devices = [dev for dev in ('cpu', 'gpu', 'ipu') if args[dev] is True]

        times = stepper.benchmark(args['iters'], batch, devices=devices)

        result = {
            'model': cfg.Model.__name__,
            'dataset': cfg.ds.name,
            '#features': cfg.ds.num_features,
            '#classes': cfg.ds.num_classes,
            '#nodes': getattr(d, 'num_nodes', d.x.size(0)),
            '#edges': getattr(d, 'num_edges', d.edge_index.size(1)),
            '#iters': args['iters'],
            'bs': cfg.bs,
            'dataloader': cfg.loader,
        }

        result.update(times)
        results.append(result)
    return results


def add_main_arguments(parser):
    main_group = parser.add_argument_group('Main')

    main_group.add_argument('--cfg',
                            type=str,
                            default=None,
                            metavar='file',
                            help="Configuration file")

    main_group.add_argument('--print-cfg',
                            type=str,
                            default=None,
                            metavar='file',
                            help="Show configuration file content")

    main_group.add_argument('--model',
                            nargs='+',
                            default=all_models,
                            help='Models to test')

    main_group.add_argument('--dataset',
                            nargs='+',
                            default=all_datasets,
                            help='Datasets to use for testing')

    main_group.add_argument('--ipu',
                            action='store_true',
                            default=True,
                            help="Run on IPU")

    main_group.add_argument('--cpu',
                            action='store_true',
                            default=False,
                            help="Run on CPU")

    main_group.add_argument('--gpu',
                            action='store_true',
                            default=False,
                            help="Run on GPU")

    main_group.add_argument('--iters',
                            type=int,
                            default=200,
                            help="Number of iterations")

    main_group.add_argument('--bs',
                            nargs='+',
                            default=[1],
                            type=int,
                            help="Number of graphs in batch.")

    main_group.add_argument('--check-values',
                            action='store_true',
                            default=False,
                            help='Run checks to make sure the results are'
                            'correct. Models run without dropout layers.')

    main_group.add_argument(
        '--synthetic-data',
        action='store_true',
        default=False,
        help='Use synthetic data on IPU (no data transfers to '
        'device)')

    main_group.add_argument(
        '--loader',
        nargs='+',
        default=['torch'],
        help=
        'Dataloader, possible values: [torch, poptorch, poptorch_fixed_size]')

    main_group.add_argument(
        '--transform',
        nargs='+',
        default=[None],
        help='Dataloader, possible values: [None, Pad]. Pass the required '
        'transformation parameters, for example: --max-num-nodes=30')

    main_group.add_argument('--fmt',
                            type=str,
                            default='rounded_outline',
                            help=f'Output format, one of: {all_formats}')

    main_group.add_argument(
        '--output',
        type=str,
        default=None,
        help='Store JSON output file with configuration and '
        'results. You can load such file later using '
        '--cfg option.')

    transform_group = parser.add_argument_group('Arguments for Pad transform')
    transform_group.add_argument(
        '--max-num-nodes',
        type=int,
        default=None,
        help='Pad transform argument. The number of nodes after padding')
    transform_group.add_argument(
        '--max-num-edges',
        type=int,
        default=None,
        help='Pad transform argument. The edges of nodes after padding')
    return parser


def get_args():
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description="Whatever comes here "
                                     "...",
                                     add_help=True,
                                     formatter_class=help_formatter)
    parser = add_main_arguments(parser)
    args, unknown = parser.parse_known_args()
    assert len(unknown) == 0, f'Unknown options {unknown}'

    args = vars(args)
    loaded_args = {}

    cfg_file = args['print_cfg'] or args['cfg']
    if cfg_file is not None:
        with open(cfg_file, "r") as infile:
            loaded_args_ = json.load(infile)
            loaded_args.update(loaded_args_)

            # Override some of the loaded args with cmd-line args
            # Can't override those args that define a test set
            overwrite_args = [
                'synthetic_data', 'check_values', 'output', 'cfg', 'print_cfg',
                'ipu', 'cpu', 'gpu'
            ]
            for arg in overwrite_args:
                loaded_args_[arg] = args[arg]
            args = loaded_args_

    assert all(d in all_datasets for d in args['dataset']), 'Unknown dataset'
    assert all(m in all_models for m in args['model']), 'Unknown model'
    assert all(ld in all_loaders
               for ld in args['loader']), 'Unknown dataloader'
    assert all(t in all_transforms
               for t in args['transform']), 'Unknown transform'

    return args, loaded_args


def print_cfg_and_results(args, loaded_args, loaded_results):
    print(f'\nArgs loaded from {args["print_cfg"]}:')
    print(loaded_args)
    print(f'\nResults loaded from {args["print_cfg"]}:')
    print_results(loaded_results, args['fmt'])


def get_tst_configs(args):
    root = osp.join(osp.dirname(osp.realpath(__file__)), 'test_data')
    datasets = DataSets(root)
    datasets = [getattr(datasets, name)() for name in args['dataset']]
    models = [globals()[name] for name in args['model']]
    batch_sizes = args['bs']
    loaders = args['loader']
    transforms = args['transform']

    configs = starmap(
        Config, product(models, datasets, batch_sizes, loaders, transforms))

    def is_supported(cfg):
        return cfg.Model in supported_sets[cfg.ds.name]

    configs = filter(is_supported, configs)
    return configs


def save_cfg_and_results(args, results):
    with open(args['output'], "w") as outfile:
        args['results'] = results
        json.dump(args, outfile, indent=4)


if __name__ == '__main__':
    args, loaded_args = get_args()
    loaded_results = loaded_args.get('results', None)
    loaded_args['results'] = None
    if args['print_cfg']:
        print_cfg_and_results(args, loaded_args, loaded_results)
        sys.exit()

    configs = get_tst_configs(args)

    results = run_benchmark(args, configs)

    if args['output'] is not None:
        save_cfg_and_results(args, results)

    if loaded_results:
        results = merge_results(results, loaded_results)

    print_results(results, args['fmt'])
