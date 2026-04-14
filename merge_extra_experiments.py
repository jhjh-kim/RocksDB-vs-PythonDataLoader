"""
Merge multiple extra_experiments.py output directories produced under
separate `srun -c` allocations and regenerate the paper figures.

Example:
    python merge_extra_experiments.py \
        --run_dirs paper_runs_srun/cpu1 paper_runs_srun/cpu2 paper_runs_srun/cpu4 \
        --output_dir paper_runs_srun/merged
"""

from __future__ import annotations

import argparse
import glob
import os
from types import SimpleNamespace

from extra_experiments import (
    aggregate_payloads,
    apply_paper_style,
    build_manifest,
    ensure_dir,
    load_json,
    plot_metric_by_cpu_budget,
    plot_representative_trace,
    save_json,
)


def collect_run_payloads(run_dirs: list[str]) -> list[dict]:
    payloads = []
    for run_dir in run_dirs:
        raw_dir = os.path.join(run_dir, "raw")
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"Missing raw directory: {raw_dir}")

        raw_paths = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
        if not raw_paths:
            raise FileNotFoundError(f"No raw JSON reports found in: {raw_dir}")

        for raw_path in raw_paths:
            payloads.append(load_json(raw_path))
    return payloads


def infer_batch_size(aggregated: list[dict]) -> int:
    batch_sizes = sorted({int(row["batch_size"]) for row in aggregated})
    if len(batch_sizes) != 1:
        raise ValueError(
            "Expected a single fixed batch size in merged results, "
            f"but found: {batch_sizes}"
        )
    return batch_sizes[0]


def infer_num_workers(aggregated: list[dict]) -> list[int]:
    return sorted({int(row["num_workers"]) for row in aggregated})


def infer_cpu_budgets(aggregated: list[dict]) -> list[int]:
    return sorted({int(row["cpu_budget"]) for row in aggregated})


def main():
    parser = argparse.ArgumentParser(description="Merge srun-separated extra experiment outputs")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="Directories created by run_srun_experiments.sh")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for merged artifacts")
    parser.add_argument("--trace_cpu_budget", type=int, default=None, help="CPU budget for the representative trace")
    parser.add_argument("--trace_num_workers", type=int, default=None, help="Worker count for the representative trace")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    apply_paper_style()

    run_payloads = collect_run_payloads(args.run_dirs)
    aggregated, trace_runs = aggregate_payloads(run_payloads)

    merged_summary = {
        "aggregated_summary": aggregated,
        "source_run_dirs": args.run_dirs,
        "run_count": len(trace_runs),
    }
    save_json(os.path.join(args.output_dir, "aggregated_results.json"), merged_summary)

    plot_metric_by_cpu_budget(
        aggregated,
        args.output_dir,
        metric_mean_field="throughput_mean",
        metric_std_field="throughput_std",
        ylabel="Throughput (samples/s)",
        title_prefix="Worker Scaling Under CPU Constraints",
        stem="fig_throughput_scaling",
    )
    plot_metric_by_cpu_budget(
        aggregated,
        args.output_dir,
        metric_mean_field="p99_latency_ms_mean",
        metric_std_field="p99_latency_ms_std",
        ylabel="P99 batch latency (ms)",
        title_prefix="Tail-Latency Sensitivity Under CPU Constraints",
        stem="fig_p99_latency_scaling",
    )

    helper_args = SimpleNamespace(
        cpu_budgets=infer_cpu_budgets(aggregated),
        num_workers=infer_num_workers(aggregated),
        batch_size=infer_batch_size(aggregated),
        epochs=None,
        repeats=None,
        subjects=None,
        mode=None,
        db_path=None,
        output_dir=args.output_dir,
        trace_cpu_budget=args.trace_cpu_budget,
        trace_num_workers=args.trace_num_workers,
    )
    plot_representative_trace(trace_runs, aggregated, helper_args)
    save_json(os.path.join(args.output_dir, "paper_manifest.json"), build_manifest(helper_args, aggregated))


if __name__ == "__main__":
    main()
