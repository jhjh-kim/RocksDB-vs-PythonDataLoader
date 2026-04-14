"""
Focused experiment suite for a short paper on CPU-constrained worker scaling.

This launcher narrows the study to a single research question:

    How do dataset implementation choice and DataLoader worker count interact
    under limited CPU affinity budgets?

It runs a small, reproducible matrix of experiments and generates three core
paper figures:
  1) throughput vs. num_workers under multiple CPU budgets
  2) P99 latency vs. num_workers under multiple CPU budgets
  3) per-step throughput stability in the most constrained setting

Recommended usage:
    python extra_experiments.py \
        --data_dir /path/to/data \
        --subjects sub-01 \
        --db_path ./rocksdb \
        --cpu_budgets 1 2 4 \
        --num_workers 0 1 2 4 8 16 \
        --batch_size 128 \
        --epochs 3 \
        --repeats 3 \
        --output_dir paper_runs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


VARIANT_ORDER = ["EEGDataset", "RocksDB"]
VARIANT_COLORS = {
    "EEGDataset": "#0B6E4F",
    "RocksDB": "#2E5AAC",
}
VARIANT_MARKERS = {
    "EEGDataset": "o",
    "RocksDB": "D",
}


def apply_paper_style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": "#D7D9E0",
        "grid.alpha": 0.6,
        "grid.linestyle": "--",
        "axes.grid": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
    })


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3:
        return values.copy()

    window = max(3, min(window, len(values)))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()

    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def variant_style(label: str) -> dict:
    return {
        "color": VARIANT_COLORS.get(label, "#555555"),
        "marker": VARIANT_MARKERS.get(label, "o"),
    }


def available_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return []


@contextmanager
def cpu_affinity_scope(cpu_budget: int):
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        yield []
        return

    original = sorted(os.sched_getaffinity(0))
    if not original:
        yield []
        return

    requested = max(1, int(cpu_budget))
    selected = original[:min(requested, len(original))]
    os.sched_setaffinity(0, set(selected))
    try:
        yield selected
    finally:
        os.sched_setaffinity(0, set(original))


def raw_report_path(output_dir: str, cpu_budget: int, repeat_idx: int) -> str:
    return os.path.join(output_dir, "raw", f"cpu{cpu_budget}_rep{repeat_idx}.json")


def raw_log_path(output_dir: str, cpu_budget: int, repeat_idx: int) -> str:
    return os.path.join(output_dir, "logs", f"cpu{cpu_budget}_rep{repeat_idx}.log")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_single_benchmark(args, cpu_budget: int, repeat_idx: int) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    perf_eval_path = os.path.join(script_dir, "perf_eval.py")
    report_path = raw_report_path(args.output_dir, cpu_budget, repeat_idx)
    log_path = raw_log_path(args.output_dir, cpu_budget, repeat_idx)

    cmd = [
        sys.executable,
        perf_eval_path,
        "--data_dir", args.data_dir,
        "--subjects", *args.subjects,
        "--mode", args.mode,
        "--db_path", args.db_path,
        "--batch_sizes", str(args.batch_size),
        "--num_workers", *[str(num_worker) for num_worker in args.num_workers],
        "--epochs", str(args.epochs),
        "--output", report_path,
        "--rocksdb_block_cache_mb", str(args.rocksdb_block_cache_mb),
        "--metrics_sample_stride", str(args.metrics_sample_stride),
        "--skip_exhaustive",
    ]

    if args.no_avg:
        cmd.append("--no_avg")
    if args.warmup:
        cmd.append("--warmup")
    if args.disable_rocksdb_metrics:
        cmd.append("--disable_rocksdb_metrics")

    with cpu_affinity_scope(cpu_budget) as pinned_cpus, open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND:\n")
        log_handle.write(" ".join(cmd) + "\n\n")
        log_handle.write(f"PINNED_CPUS: {pinned_cpus}\n\n")
        log_handle.flush()
        subprocess.run(cmd, cwd=script_dir, stdout=log_handle, stderr=subprocess.STDOUT, check=True)

    payload = load_json(report_path)
    payload.setdefault("metadata", {})
    payload["metadata"]["cpu_budget_requested"] = int(cpu_budget)
    payload["metadata"]["repeat"] = int(repeat_idx)
    payload["metadata"]["batch_size_fixed"] = int(args.batch_size)
    payload["metadata"]["num_workers_sweep"] = list(args.num_workers)
    payload["metadata"]["raw_report_path"] = report_path
    payload["metadata"]["raw_log_path"] = log_path
    return payload


def aggregate_payloads(run_payloads: list[dict]) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    trace_runs = []

    for payload in run_payloads:
        metadata = payload.get("metadata", {})
        cpu_budget = int(metadata["cpu_budget_requested"])
        repeat_idx = int(metadata["repeat"])

        for row in payload.get("summary", []):
            if row["label"] not in VARIANT_ORDER:
                continue
            entry = dict(row)
            entry["cpu_budget"] = cpu_budget
            entry["repeat"] = repeat_idx
            summary_rows.append(entry)

        trace_runs.append({
            "cpu_budget": cpu_budget,
            "repeat": repeat_idx,
            "metadata": metadata,
            "summary": payload.get("summary", []),
            "traces": payload.get("traces", []),
        })

    grouped = defaultdict(list)
    for row in summary_rows:
        key = (row["label"], int(row["cpu_budget"]), int(row["batch_size"]), int(row["num_workers"]))
        grouped[key].append(row)

    aggregated = []
    for (label, cpu_budget, batch_size, num_workers), rows in sorted(grouped.items()):
        throughput_values = np.array([row["throughput_mean"] for row in rows], dtype=float)
        p99_values = np.array([row["batch_latency_p99"] * 1000.0 for row in rows], dtype=float)
        p50_values = np.array([row["batch_latency_p50"] * 1000.0 for row in rows], dtype=float)
        p95_values = np.array([row["batch_latency_p95"] * 1000.0 for row in rows], dtype=float)

        aggregated.append({
            "label": label,
            "cpu_budget": cpu_budget,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "repeats": len(rows),
            "throughput_mean": float(np.mean(throughput_values)),
            "throughput_std": float(np.std(throughput_values)),
            "p50_latency_ms_mean": float(np.mean(p50_values)),
            "p50_latency_ms_std": float(np.std(p50_values)),
            "p95_latency_ms_mean": float(np.mean(p95_values)),
            "p95_latency_ms_std": float(np.std(p95_values)),
            "p99_latency_ms_mean": float(np.mean(p99_values)),
            "p99_latency_ms_std": float(np.std(p99_values)),
        })

    return aggregated, trace_runs


def save_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_metric_by_cpu_budget(
    aggregated: list[dict],
    output_dir: str,
    metric_mean_field: str,
    metric_std_field: str,
    ylabel: str,
    title_prefix: str,
    stem: str,
):
    cpu_budgets = sorted({row["cpu_budget"] for row in aggregated})
    fig, axes = plt.subplots(
        1,
        len(cpu_budgets),
        figsize=(5.0 * len(cpu_budgets), 4.1),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    legend_handles = []
    for ax, cpu_budget in zip(axes, cpu_budgets):
        subset = [row for row in aggregated if row["cpu_budget"] == cpu_budget]
        for label in VARIANT_ORDER:
            rows = sorted(
                [row for row in subset if row["label"] == label],
                key=lambda row: int(row["num_workers"]),
            )
            if not rows:
                continue

            workers = np.array([row["num_workers"] for row in rows], dtype=int)
            means = np.array([row[metric_mean_field] for row in rows], dtype=float)
            stds = np.array([row[metric_std_field] for row in rows], dtype=float)
            style = variant_style(label)
            (line,) = ax.plot(workers, means, label=label, **style)
            legend_handles.append(line)
            ax.fill_between(workers, np.maximum(means - stds, 0.0), means + stds, color=style["color"], alpha=0.15)

        ax.set_title(f"CPU budget = {cpu_budget}")
        ax.set_xlabel("DataLoader workers")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title_prefix, y=1.03, fontsize=13, fontweight="semibold")
    unique_handles = {handle.get_label(): handle for handle in legend_handles}
    fig.legend(unique_handles.values(), unique_handles.keys(), loc="upper center", ncols=max(1, len(unique_handles)))
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), bbox_inches="tight")
    plt.close(fig)


def representative_trace_run(trace_runs: list[dict], aggregated: list[dict], cpu_budget: int, num_workers: int, batch_size: int):
    candidate_runs = []
    target_throughput = None
    for row in aggregated:
        if (
            row["label"] == "RocksDB"
            and row["cpu_budget"] == cpu_budget
            and row["num_workers"] == num_workers
            and row["batch_size"] == batch_size
        ):
            target_throughput = row["throughput_mean"]
            break

    for run in trace_runs:
        if run["cpu_budget"] != cpu_budget:
            continue

        has_target = False
        run_rocksdb_throughput = None
        for row in run["summary"]:
            if (
                row["label"] == "RocksDB"
                and int(row["num_workers"]) == num_workers
                and int(row["batch_size"]) == batch_size
            ):
                has_target = True
                run_rocksdb_throughput = row["throughput_mean"]
                break

        if has_target:
            candidate_runs.append((run, run_rocksdb_throughput))

    if not candidate_runs:
        return None

    if target_throughput is None:
        return candidate_runs[0][0]

    candidate_runs.sort(key=lambda item: abs(item[1] - target_throughput))
    return candidate_runs[0][0]


def plot_representative_trace(trace_runs: list[dict], aggregated: list[dict], args):
    chosen_cpu_budget = args.trace_cpu_budget if args.trace_cpu_budget is not None else min(args.cpu_budgets)
    chosen_num_workers = args.trace_num_workers if args.trace_num_workers is not None else max(args.num_workers)
    run = representative_trace_run(trace_runs, aggregated, chosen_cpu_budget, chosen_num_workers, args.batch_size)
    if run is None:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    for label in VARIANT_ORDER:
        rows = [
            row for row in run["traces"]
            if row["label"] == label
            and int(row["batch_size"]) == args.batch_size
            and int(row["num_workers"]) == chosen_num_workers
            and int(row["epoch"]) == 1
        ]
        if not rows:
            continue

        rows = sorted(rows, key=lambda row: int(row["step_in_epoch"]))
        steps = np.arange(1, len(rows) + 1, dtype=int)
        throughput = np.array([row["throughput_samples_per_sec"] for row in rows], dtype=float)
        smooth_window = max(5, min(21, len(throughput) // 5 if len(throughput) >= 25 else len(throughput)))
        smoothed = smooth_series(throughput, smooth_window)
        style = variant_style(label)

        ax.plot(steps, throughput, color=style["color"], alpha=0.18, linewidth=1.0)
        ax.plot(
            steps,
            smoothed,
            color=style["color"],
            marker=style["marker"],
            markevery=max(1, len(steps) // 10),
            label=label,
        )

    ax.set_title(
        f"Per-Step Throughput Stability (CPU budget = {chosen_cpu_budget}, workers = {chosen_num_workers})"
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (samples/s)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best")
    fig.savefig(os.path.join(args.output_dir, "fig_trace_stability.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.output_dir, "fig_trace_stability.png"), bbox_inches="tight")
    plt.close(fig)


def build_manifest(args, aggregated: list[dict]) -> dict:
    return {
        "paper_focus": "CPU-constrained worker scaling for DataLoader pipelines",
        "core_hypotheses": [
            "H1: Under limited CPU affinity, increasing num_workers does not reliably improve throughput for the baseline Python dataset implementation.",
            "H2: Under the same CPU budget, the RocksDB-backed dataset maintains higher throughput or degrades more slowly as worker count increases.",
            "H3: In oversubscribed settings, the RocksDB-backed dataset shows lower tail latency and more stable per-step throughput.",
        ],
        "recommended_core_plots": [
            {
                "file": "fig_throughput_scaling.pdf",
                "message": "Throughput scaling with worker count under CPU budgets 1, 2, and 4.",
            },
            {
                "file": "fig_p99_latency_scaling.pdf",
                "message": "Tail-latency sensitivity under the same CPU budgets.",
            },
            {
                "file": "fig_trace_stability.pdf",
                "message": "Representative step-level stability in the most constrained configuration.",
            },
        ],
        "experiment_config": {
            "cpu_budgets": args.cpu_budgets,
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "repeats": args.repeats,
            "subjects": args.subjects,
            "mode": args.mode,
            "db_path": args.db_path,
        },
        "aggregated_points": len(aggregated),
    }


def main():
    parser = argparse.ArgumentParser(description="Run focused CPU-constrained extra experiments for the paper")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory containing subject folders")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs, e.g. sub-01 sub-02")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the RocksDB database")
    parser.add_argument("--cpu_budgets", nargs="+", type=int, default=[1, 2, 4], help="Visible CPU budgets to evaluate")
    parser.add_argument("--num_workers", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16], help="Worker counts to sweep")
    parser.add_argument("--batch_size", type=int, default=128, help="Single fixed batch size for the paper experiment")
    parser.add_argument("--epochs", type=int, default=3, help="Measured epochs per run")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated runs per CPU budget")
    parser.add_argument("--output_dir", type=str, default="paper_runs", help="Directory for raw reports and figures")
    parser.add_argument("--rocksdb_block_cache_mb", type=int, default=512, help="Per-process RocksDB block cache size")
    parser.add_argument("--metrics_sample_stride", type=int, default=1, help="RocksDB property sampling stride")
    parser.add_argument("--trace_cpu_budget", type=int, default=None, help="CPU budget used for the representative trace plot")
    parser.add_argument("--trace_num_workers", type=int, default=None, help="Worker count used for the representative trace plot")
    parser.add_argument("--no_avg", action="store_true", help="Pass through to perf_eval.py")
    parser.add_argument("--warmup", action="store_true", help="Pass through to perf_eval.py")
    parser.add_argument("--disable_rocksdb_metrics", action="store_true", help="Pass through to perf_eval.py")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "raw"))
    ensure_dir(os.path.join(args.output_dir, "logs"))
    apply_paper_style()

    print("=== Focused Paper Experiment ===")
    print(f"visible_cpus : {available_cpus()}")
    print(f"cpu_budgets  : {args.cpu_budgets}")
    print(f"num_workers  : {args.num_workers}")
    print(f"batch_size   : {args.batch_size}")
    print(f"epochs       : {args.epochs}")
    print(f"repeats      : {args.repeats}")
    print(f"db_path      : {args.db_path}")
    print()

    run_payloads = []
    for cpu_budget in args.cpu_budgets:
        for repeat_idx in range(1, args.repeats + 1):
            print(f"[run] cpu_budget={cpu_budget}, repeat={repeat_idx}")
            run_payloads.append(run_single_benchmark(args, cpu_budget, repeat_idx))

    aggregated, trace_runs = aggregate_payloads(run_payloads)

    save_json(
        os.path.join(args.output_dir, "aggregated_results.json"),
        {
            "aggregated_summary": aggregated,
            "runs": [
                {
                    "cpu_budget": run["cpu_budget"],
                    "repeat": run["repeat"],
                    "metadata": run["metadata"],
                }
                for run in trace_runs
            ],
        },
    )

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
    plot_representative_trace(trace_runs, aggregated, args)

    save_json(os.path.join(args.output_dir, "paper_manifest.json"), build_manifest(args, aggregated))
    print(f"\nArtifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
