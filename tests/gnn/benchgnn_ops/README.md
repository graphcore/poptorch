# benchgnn

Benchmark tool for testing performance of GNN operators

## Usage example

Running single benchmark test case scenario from command line:
``python3 benchgnn_ops.py --num_sample_rounds 10 scatter --src_shape [1,12] --input_shape [1,12] --index_shape [1,12] --dim 0``

Running multiple benchmark test case scenarios from yaml configuration files from given directory:
``python3 benchgnn_ops.py --common_config=example_configs/common.yaml --config_dir=example_configs``

Running multiple benchmark test case scenarios from given yaml configuration files:
``python3 benchgnn_ops.py --common_config=example_configs/common.yaml --config_files=[example_configs/scatter_testcase1.yaml,example_configs/scatter_testcase2.yaml]``

Running multiple benchmark test case scenarios - combining all available options:
``python3 benchgnn_ops.py --common_config=example_configs/common.yaml --config_dir=example_configs --config_files=[example_configs/scatter_testcase1.yaml,example_configs/scatter_testcase2.yaml] scatter --src_shape [1,12] --input_shape [1,12] --index_shape [1,12] --dim 0``

Type ``python3 benchgnn_ops.py --help`` to print detailed information about supported options.
