# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

from tabulate import _table_formats, tabulate

all_formats = sorted(list(_table_formats.keys()))


def merge_results(results, prev_results):
    if prev_results:
        keys = prev_results[0].keys()
        time_keys = {'ipu_time', 'gpu_time', 'cpu_time'}.intersection(keys)
        for curr, prev in zip(results, prev_results):
            for key in time_keys:
                curr['prev_' + key.split('_')[0]] = prev[key]
    return results


def include_speedups_ratio(results):
    keys = list(results[0].keys())

    # Calculate speedup over other times
    if 'ipu_time' in keys:
        other = filter(lambda x: x in keys,
                       ('cpu_time', 'prev_cpu', 'prev_gpu', 'prev_ipu'))
        for t in other:
            for res in results:
                res['ipu/' + t] = res[t] / res["ipu_time"]

    return results


def print_results(results, format):
    results = include_speedups_ratio(results)

    content = [list(results[0].keys())]
    prev_model = None
    for res in results:
        curr_model = res['model']
        if prev_model != curr_model:
            if prev_model is not None:
                content.append([])
            prev_model = curr_model
        else:
            res['model'] = ''

        row = [f'{x:.2f}' if isinstance(x, float) else x for x in res.values()]

        content.append(row)

    body = tabulate(content, headers='firstrow', tablefmt=format)
    print('\n', body, sep='')
