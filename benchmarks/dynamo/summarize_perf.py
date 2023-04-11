import logging
import os
import re
from collections import defaultdict

import click
import pandas as pd
from tabulate import tabulate


def gmean(s):
    return s.product() ** (1 / len(s))


def find_csv_files(path):
    """
    Recursively search for all CSV files in directory and subdirectories whose
    name contains a target string.
    """
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("_performance.csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


@click.command()
@click.argument("directory", default="artifacts")
@click.option("--amp", is_flag=True)
@click.option("--float32", is_flag=True)
def main(directory, amp, float32):
    """
    Given a directory containing multiple CSVs from --performance benchmark
    runs, aggregates and generates summary statistics similar to the web UI at
    https://torchci-git-fork-huydhn-add-compilers-bench-74abf8-fbopensource.vercel.app/benchmark/compilers

    This is most useful if you've downloaded CSVs from CI and need to quickly
    look at aggregate stats.  The CSVs are expected to follow exactly the same
    naming convention that is used in CI.

    You may also be interested in
    https://docs.google.com/document/d/1DQQxIgmKa3eF0HByDTLlcJdvefC4GwtsklJUgLs09fQ/edit#
    which explains how to interpret the raw csv data.
    """
    dtypes = ["amp", "float32"]
    if amp and not float32:
        dtypes = ["amp"]
    if float32 and not amp:
        dtypes = ["float32"]

    dfs = defaultdict(list)
    for f in find_csv_files(directory):
        try:
            dfs[os.path.basename(f)].append(pd.read_csv(f))
        except Exception:
            logging.warning("failed parsing %s", f)
            raise

    # dtype -> statistic -> benchmark -> compiler -> value
    results = defaultdict(  # dtype
        lambda: defaultdict(  # statistic
            lambda: defaultdict(dict)  # benchmark  # compiler -> value
        )
    )

    for k, v in sorted(dfs.items()):
        regex = (
            "(inductor_with_cudagraphs|inductor_no_cudagraphs|inductor_dynamic)_"
            "(torchbench|huggingface|timm_models)_"
            "(float32|amp)_"
            "(inference|training)_"
            "cuda_"
            r"performance\.csv"
        )
        m = re.match(regex, k)
        assert m is not None, k
        compiler = m.group(1)
        benchmark = m.group(2)
        dtype = m.group(3)
        mode = m.group(4)

        df = pd.concat(v)
        df = df.dropna().query("speedup != 0")

        statistics = {
            "speedup": gmean(df["speedup"]),
            "comptime": df["compilation_latency"].mean(),
            "memory": gmean(df["compression_ratio"]),
        }

        if dtype not in dtypes:
            continue

        for statistic, v in statistics.items():
            results[f"{dtype} {mode}"][statistic][benchmark][compiler] = v

    descriptions = {
        "speedup": "Geometric mean speedup",
        "comptime": "Mean compilation time",
        "memory": "Peak memory compression ratio",
    }

    for dtype_mode, r in results.items():
        print(f"# {dtype_mode} performance results")
        for statistic, data in r.items():
            print(f"## {descriptions[statistic]}")

            table = []
            for row_name in data[list(data.keys())[0]]:
                row = [row_name]
                for col_name in data:
                    row.append(round(data[col_name][row_name], 2))
                table.append(row)

            headers = list(data.keys())
            print(tabulate(table, headers=headers))
            print()


main()
