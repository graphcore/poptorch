# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import os
from typing import List, Tuple

import jsonargparse
import pandas as pd
import torch
from builder import BenchModelBuilder
from metrics import PerfMetrics, to_data_frame
from ops import bench_ops
from tqdm import tqdm


def prepare_parser() -> jsonargparse.ArgumentParser:
    jsonargparse.set_docstring_parse_options(attribute_docstrings=True)
    jsonargparse.typing.register_type(torch.Size, torch.Size, torch.Size)

    parser = jsonargparse.ArgumentParser(prog='GNN Ops Benchmark')
    parser.add_class_arguments(BenchModelBuilder, 'compile_options')

    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='the random seed to use.')
    parser.add_argument(
        '--num_warmup_rounds',
        type=int,
        default=4,
        help='num_warmup_rounds: initial set of runs to discard.')
    parser.add_argument(
        '--num_sample_rounds',
        type=int,
        default=1,
        help='num_sample_rounds: the number of runs used to average the '
        'runtime.')
    parser.add_argument(
        '--calc_samples_mean',
        type=bool,
        default=True,
        help='calculate mean over collected `num_sample_rounds` measurements.')
    parser.add_argument(
        '--clock',
        type=int,
        default=None,
        help='manually override clock value (Mhz) read by gcipuinfo.')
    parser.add_argument(
        '--common_config',
        type=jsonargparse.typing.Path_fr,
        default=None,
        help='yaml file containing configuration options shared between all '
        'benchmark testcases.')
    parser.add_argument(
        '--config_dir',
        type=jsonargparse.typing.Path_drw,
        default=None,
        help='directory with a set of yaml benchmark test case scenario'
        'files.')
    parser.add_argument(
        '--config_files',
        type=List[jsonargparse.typing.Path_fr],
        default=None,
        help='set of yaml file paths containig benchmark test case scenarios.')

    subcommands = parser.add_subcommands(required=False, dest='operation')

    for command, op_type in bench_ops.items():
        subparser = jsonargparse.ArgumentParser()
        subparser.add_class_arguments(op_type)
        subcommands.add_subcommand(command, subparser)

    return parser


def collect_measurements(config_src: str,
                         testcase_config: jsonargparse.namespace.Namespace
                         ) -> pd.DataFrame:
    torch.manual_seed(testcase_config.seed)
    op_name = testcase_config.operation
    op_params = getattr(testcase_config, op_name)

    operator = bench_ops[op_name](**op_params.as_dict())
    builder = BenchModelBuilder(**testcase_config.compile_options.as_dict())
    compiled_model = builder.create_model(operator)
    metrics = PerfMetrics(config_src, operator,
                          testcase_config.compile_options.num_repeats, op_name,
                          str(op_params), testcase_config.clock)

    for _ in range(testcase_config.num_warmup_rounds):
        _ = compiled_model()

    measurements = []
    for _ in range(testcase_config.num_sample_rounds):
        _ = compiled_model()
        measurements.append(
            metrics.get_measurement(compiled_model.cycleCount()))

    return to_data_frame(measurements, testcase_config.calc_samples_mean)


def run_benchmark(testcases: List[Tuple[str, jsonargparse.namespace.Namespace]]
                  ) -> pd.DataFrame:
    bar = tqdm(range(len(testcases)),
               desc="Benchmarking progress",
               unit="testcase",
               position=3)

    data_frames = []
    for testcase_config in testcases:
        data_frames.append(collect_measurements(*testcase_config))
        bar.update()
        bar.refresh()
    bar.clear()
    bar.close()

    return pd.concat(data_frames, ignore_index=True)


def set_defaults_from_yaml_config(
        parser: jsonargparse.ArgumentParser,
        common_config_path: jsonargparse.typing.Path_fr) -> None:
    common_config_raw = parser.parse_path(common_config_path, defaults=False)
    parser.set_defaults(**dict(common_config_raw.as_flat()._get_kwargs()))  # pylint: disable=protected-access


def set_defaults_from_user_params(parser: jsonargparse.ArgumentParser,
                                  user_params: jsonargparse.namespace.Namespace
                                  ) -> None:

    default_params = user_params.clone()
    if 'operation' in default_params:
        op = default_params['operation']
        del default_params[op]
        del default_params['operation']

    parser.set_defaults(**dict(default_params.as_flat()._get_kwargs()))  # pylint: disable=protected-access


def set_defaults(parser: jsonargparse.ArgumentParser,
                 user_params: jsonargparse.namespace.Namespace) -> None:
    common_config_path = None
    if 'common_config' in user_params:
        common_config_path = user_params.common_config.abs_path
        set_defaults_from_yaml_config(parser, common_config_path)

    set_defaults_from_user_params(parser, user_params)


def get_test_case_config_paths(user_params: jsonargparse.namespace.Namespace
                               ) -> List[str]:
    test_case_config_paths = []

    common_config_path = None
    if 'common_config' in user_params:
        common_config_path = user_params.common_config.abs_path

    def is_valid_path(path: str) -> bool:
        return os.path.isfile(path) and path != common_config_path

    if 'config_dir' in user_params:
        base_dir = user_params.config_dir.abs_path

        for filename in os.listdir(base_dir):
            file_path = os.path.join(base_dir, filename)
            if is_valid_path(file_path):
                test_case_config_paths.append(file_path)

    if 'config_files' in user_params:
        for file_path in user_params.config_files:
            file_abs_path = file_path.abs_path
            if is_valid_path(file_abs_path):
                test_case_config_paths.append(file_abs_path)

    return test_case_config_paths


def parse_test_case_config_files(test_case_config_paths: List[str]
                                 ) -> List[jsonargparse.namespace.Namespace]:
    test_case_configs = []
    for file_path in test_case_config_paths:
        try:
            test_case_configs.append((
                os.path.basename(file_path),
                parser.parse_path(file_path),
            ))
        except Exception as e:
            print(f'Parsing {file_path} failed.')
            raise e
    return test_case_configs


def get_test_case_configs(parser: jsonargparse.ArgumentParser,
                          user_params: jsonargparse.namespace.Namespace
                          ) -> List[jsonargparse.namespace.Namespace]:
    test_case_configs = []

    if 'operation' in user_params:
        test_case_configs.append((
            'cmd',
            parser.parse_args(defaults=True),
        ))

    config_paths = get_test_case_config_paths(user_params)
    test_case_configs.extend(parse_test_case_config_files(config_paths))

    return test_case_configs


if __name__ == "__main__":
    parser = prepare_parser()

    user_params = parser.parse_args(defaults=False)
    set_defaults(parser, user_params)
    test_case_configs = get_test_case_configs(parser, user_params)

    if test_case_configs:
        results = run_benchmark(test_case_configs)
        print(results.to_string())
    else:
        print('No test cases to benchmark. Please check `python3 '
              'benchgnn_ops.py --help`.')
