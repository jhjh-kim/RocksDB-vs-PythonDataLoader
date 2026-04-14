"""
Merge manually collected perf_eval.py outputs from separate CPU allocations.

Expected layout:
  perf_eval_runs/
    cpu1/perf_eval.json
    cpu2/perf_eval.json
    cpu4/perf_eval.json

Example:
  python merge_perf_eval_manual.py \
    --input_root perf_eval_runs \
    --cpu_labels cpu1 cpu2 cpu4 \
    --output_dir perf_eval_runs/merged
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

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


def variant_style(label: str) -> dict:
    return {
        "color": VARIANT_COLORS.get(label, "#555555"),
        "marker": VARIANT_MARKERS.get(label, "o"),
    }


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


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def cpu_budget_from_label(label: str) -> int:
    match = re.search(r"(\d+)", label)
    if not match:
        raise ValueError(f"Could not infer CPU budget from label: {label}")
    return int(match.group(1))


def collect_payloads(input_root: str, cpu_labels: list[str]) -> list[dict]:
    payloads = []
    for label in cpu_labels:
        json_path = os.path.join(input_root, label, "perf_eval.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing perf_eval.json: {json_path}")

        payload = load_json(json_path)
        payload.setdefault("metadata", {})
        payload["metadata"]["cpu_label"] = label
        payload["metadata"]["cpu_budget_requested"] = cpu_budget_from_label(label)
        payloads.append(payload)
    return payloads


def aggregate_payloads(payloads: list[dict]) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    trace_runs = []

    for payload in payloads:
        metadata = payload.get("metadata", {})
        cpu_label = metadata["cpu_label"]
        cpu_budget = int(metadata["cpu_budget_requested"])

        for row in payload.get("summary", []):
            if row["label"] not in VARIANT_ORDER:
                continue
            entry = dict(row)
            entry["cpu_label"] = cpu_label
            entry["cpu_budget"] = cpu_budget
            summary_rows.append(entry)

        trace_runs.append({
            "cpu_label": cpu_label,
            "cpu_budget": cpu_budget,
            "metadata": metadata,
            "summary": payload.get("summary", []),
            "traces": payload.get("traces", []),
        })

    return summary_rows, trace_runs


def infer_batch_sizes(summary_rows: list[dict]) -> list[int]:
    return sorted({int(row["batch_size"]) for row in summary_rows})


def infer_num_workers(summary_rows: list[dict]) -> list[int]:
    return sorted({int(row["num_workers"]) for row in summary_rows})


def plot_metric_by_cpu_budget(
    summary_rows: list[dict],
    output_dir: str,
    batch_size: int,
    metric_field: str,
    metric_scale: float,
    ylabel: str,
    title_prefix: str,
    stem: str,
):
    cpu_budgets = sorted({int(row["cpu_budget"]) for row in summary_rows})
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
        subset = [
            row for row in summary_rows
            if int(row["cpu_budget"]) == cpu_budget and int(row["batch_size"]) == batch_size
        ]
        for label in VARIANT_ORDER:
            rows = sorted(
                [row for row in subset if row["label"] == label],
                key=lambda row: int(row["num_workers"]),
            )
            if not rows:
                continue

            workers = np.array([row["num_workers"] for row in rows], dtype=int)
            values = np.array([row[metric_field] * metric_scale for row in rows], dtype=float)
            style = variant_style(label)
            (line,) = ax.plot(workers, values, label=label, **style)
            legend_handles.append(line)

        ax.set_title(f"CPU budget = {cpu_budget}")
        ax.set_xlabel("DataLoader workers")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{title_prefix} (batch size = {batch_size})", y=1.03, fontsize=13, fontweight="semibold")
    unique_handles = {handle.get_label(): handle for handle in legend_handles}
    fig.legend(unique_handles.values(), unique_handles.keys(), loc="upper center", ncols=max(1, len(unique_handles)))
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), bbox_inches="tight")
    plt.close(fig)


def representative_trace_run(trace_runs: list[dict], cpu_budget: int) -> dict | None:
    for run in trace_runs:
        if int(run["cpu_budget"]) == int(cpu_budget):
            return run
    return None


def plot_representative_trace(
    trace_runs: list[dict],
    output_dir: str,
    batch_size: int,
    cpu_budget: int,
    num_workers: int,
):
    run = representative_trace_run(trace_runs, cpu_budget)
    if run is None:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    for label in VARIANT_ORDER:
        rows = [
            row for row in run["traces"]
            if row["label"] == label
            and int(row["batch_size"]) == batch_size
            and int(row["num_workers"]) == num_workers
            and int(row["epoch"]) == 1
        ]
        if not rows:
            continue

        rows = sorted(rows, key=lambda row: int(row["step_in_epoch"]))
        steps = np.arange(1, len(rows) + 1, dtype=int)
        throughput = np.array([row["throughput_samples_per_sec"] for row in rows], dtype=float)
        smoothed = smooth_series(throughput, max(5, min(21, len(throughput) // 5 if len(throughput) >= 25 else len(throughput))))
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

    ax.set_title(f"Per-Step Throughput Stability (CPU budget = {cpu_budget}, workers = {num_workers})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Throughput (samples/s)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best")
    fig.savefig(os.path.join(output_dir, "fig_trace_stability.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "fig_trace_stability.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Merge manually collected perf_eval outputs")
    parser.add_argument("--input_root", type=str, required=True, help="Root directory containing cpu1/cpu2/cpu4 subdirectories")
    parser.add_argument("--cpu_labels", nargs="+", default=["cpu1", "cpu2", "cpu4"], help="Subdirectory labels to merge")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for merged figures and JSON")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size to plot; defaults to the smallest available")
    parser.add_argument("--trace_cpu_budget", type=int, default=1, help="CPU budget used for the representative trace")
    parser.add_argument("--trace_num_workers", type=int, default=None, help="Worker count used for the representative trace")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    apply_paper_style()

    payloads = collect_payloads(args.input_root, args.cpu_labels)
    summary_rows, trace_runs = aggregate_payloads(payloads)
    batch_sizes = infer_batch_sizes(summary_rows)
    batch_size = args.batch_size if args.batch_size is not None else batch_sizes[0]
    num_workers = args.trace_num_workers if args.trace_num_workers is not None else max(infer_num_workers(summary_rows))

    merged_summary = {
        "source_input_root": args.input_root,
        "cpu_labels": args.cpu_labels,
        "batch_size_plotted": batch_size,
        "summary": summary_rows,
    }
    with open(os.path.join(args.output_dir, "merged_perf_eval.json"), "w", encoding="utf-8") as handle:
        json.dump(merged_summary, handle, indent=2)

    plot_metric_by_cpu_budget(
        summary_rows,
        args.output_dir,
        batch_size=batch_size,
        metric_field="throughput_mean",
        metric_scale=1.0,
        ylabel="Throughput (samples/s)",
        title_prefix="Worker Scaling Under CPU Constraints",
        stem="fig_throughput_scaling",
    )
    plot_metric_by_cpu_budget(
        summary_rows,
        args.output_dir,
        batch_size=batch_size,
        metric_field="batch_latency_p99",
        metric_scale=1000.0,
        ylabel="P99 batch latency (ms)",
        title_prefix="Tail-Latency Sensitivity Under CPU Constraints",
        stem="fig_p99_latency_scaling",
    )
    plot_representative_trace(
        trace_runs,
        args.output_dir,
        batch_size=batch_size,
        cpu_budget=args.trace_cpu_budget,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
