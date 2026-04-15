"""
Generate paper-quality figures from the JSON report produced by perf_eval.py.

Supported figures:
  - throughput overview
  - latency overview (P50 / P95 / P99)
  - per-step throughput traces
  - RocksDB block-cache hit-rate traces
  - RocksDB compaction traces

Usage:
    python3 paper_figures.py \
        --input benchmark_report.json \
        --output_dir figures/paper

Focused step-throughput comparison only:
    python3 paper_figures.py \
        --input benchmark_report.json \
        --output_dir figures/paper \
        --focus_batch_size 128 \
        --focus_num_workers 8 \
        --only_focus_plot
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


VARIANT_ORDER = ["EEGDataset", "Exhaustive", "RocksDB"]
VARIANT_COLORS = {
    "EEGDataset": "#0B6E4F",
    "Exhaustive": "#C84C09",
    "RocksDB": "#2E5AAC",
}
VARIANT_MARKERS = {
    "EEGDataset": "o",
    "Exhaustive": "s",
    "RocksDB": "D",
}
FOCUSED_TRACE_VARIANTS = ["EEGDataset", "RocksDB"]
FOCUSED_TRACE_COLORS = {
    "EEGDataset": "#2E8B57",
    "RocksDB": "#E67E22",
}
FOCUSED_TRACE_DISPLAY_NAMES = {
    "EEGDataset": "EEGDataset",
    "RocksDB": "RocksDBDataset",
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


def load_report(path: str) -> tuple[list[dict], list[dict], dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload, [], {}

    return payload.get("summary", []), payload.get("traces", []), payload.get("metadata", {})


def ordered_variants(records: list[dict]) -> list[str]:
    seen = {record["label"] for record in records}
    ordered = [variant for variant in VARIANT_ORDER if variant in seen]
    remainder = sorted(seen - set(ordered))
    return ordered + remainder


def unique_sorted(records: list[dict], field: str) -> list:
    return sorted({record[field] for record in records})


def save_figure(fig, output_dir: str, stem: str, formats: list[str]):
    for ext in formats:
        fig.savefig(os.path.join(output_dir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)


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


def group_by_config(records: list[dict]) -> dict[tuple[int, int], list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(int(record["batch_size"]), int(record["num_workers"]))].append(record)
    return dict(grouped)


def variant_style(variant: str) -> dict:
    return {
        "color": VARIANT_COLORS.get(variant, "#555555"),
        "marker": VARIANT_MARKERS.get(variant, "o"),
    }


def focused_trace_style(variant: str) -> dict:
    return {
        "color": FOCUSED_TRACE_COLORS.get(variant, "#555555"),
        "marker": VARIANT_MARKERS.get(variant, "o"),
    }


def plot_throughput_overview(summary: list[dict], output_dir: str, formats: list[str]):
    if not summary:
        return

    batch_sizes = unique_sorted(summary, "batch_size")
    variants = ordered_variants(summary)
    fig, axes = plt.subplots(
        1,
        len(batch_sizes),
        figsize=(5.2 * len(batch_sizes), 4.0),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    legend_handles = []
    for ax, batch_size in zip(axes, batch_sizes):
        subset = [record for record in summary if int(record["batch_size"]) == batch_size]
        for variant in variants:
            variant_rows = sorted(
                [record for record in subset if record["label"] == variant],
                key=lambda record: int(record["num_workers"]),
            )
            if not variant_rows:
                continue

            workers = np.array([row["num_workers"] for row in variant_rows], dtype=int)
            throughput = np.array([row["throughput_mean"] for row in variant_rows], dtype=float)
            error = np.array([row["throughput_std"] for row in variant_rows], dtype=float)
            style = variant_style(variant)

            (line,) = ax.plot(workers, throughput, label=variant, **style)
            legend_handles.append(line)
            lower = np.maximum(throughput - error, 0.0)
            upper = throughput + error
            ax.fill_between(workers, lower, upper, color=style["color"], alpha=0.15)

        ax.set_title(f"Batch size = {batch_size}")
        ax.set_xlabel("DataLoader workers")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].set_ylabel("Throughput (samples/s)")
    fig.suptitle("Read Throughput Across Worker Counts", y=1.03, fontsize=13, fontweight="semibold")
    unique_handles = {handle.get_label(): handle for handle in legend_handles}
    fig.legend(unique_handles.values(), unique_handles.keys(), loc="upper center", ncols=max(1, len(unique_handles)))
    save_figure(fig, output_dir, "fig_throughput_overview", formats)


def plot_latency_overview(summary: list[dict], output_dir: str, formats: list[str]):
    if not summary:
        return

    batch_sizes = unique_sorted(summary, "batch_size")
    variants = ordered_variants(summary)
    latency_fields = [
        ("batch_latency_p50", "P50 latency (ms)"),
        ("batch_latency_p95", "P95 latency (ms)"),
        ("batch_latency_p99", "P99 latency (ms)"),
    ]

    for batch_size in batch_sizes:
        subset = [record for record in summary if int(record["batch_size"]) == batch_size]
        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0), sharex=True, constrained_layout=True)
        legend_handles = []

        for ax, (field, ylabel) in zip(axes, latency_fields):
            for variant in variants:
                variant_rows = sorted(
                    [record for record in subset if record["label"] == variant],
                    key=lambda record: int(record["num_workers"]),
                )
                if not variant_rows:
                    continue

                workers = np.array([row["num_workers"] for row in variant_rows], dtype=int)
                latency_ms = np.array([row[field] * 1000.0 for row in variant_rows], dtype=float)
                style = variant_style(variant)
                (line,) = ax.plot(workers, latency_ms, label=variant, **style)
                legend_handles.append(line)

            ax.set_title(ylabel.replace(" (ms)", ""))
            ax.set_xlabel("DataLoader workers")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(f"Batch Latency Quantiles (batch size = {batch_size})", y=1.03, fontsize=13, fontweight="semibold")
        unique_handles = {handle.get_label(): handle for handle in legend_handles}
        fig.legend(unique_handles.values(), unique_handles.keys(), loc="upper center", ncols=max(1, len(unique_handles)))
        save_figure(fig, output_dir, f"fig_latency_bs{batch_size}", formats)


def plot_iteration_throughput(traces: list[dict], output_dir: str, formats: list[str]):
    if not traces:
        return

    variants = ordered_variants(traces)
    grouped = group_by_config(traces)
    for (batch_size, num_workers), records in grouped.items():
        fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)

        for variant in variants:
            variant_rows = [record for record in records if record["label"] == variant]
            if not variant_rows:
                continue

            variant_rows = sorted(variant_rows, key=lambda record: int(record["global_step"]))
            steps = np.arange(1, len(variant_rows) + 1, dtype=int)
            throughput = np.array([row["throughput_samples_per_sec"] for row in variant_rows], dtype=float)
            style = variant_style(variant)
            smooth_window = max(5, min(21, len(throughput) // 5 if len(throughput) >= 25 else len(throughput)))
            smoothed = smooth_series(throughput, smooth_window)

            ax.plot(steps, throughput, color=style["color"], alpha=0.18, linewidth=1.0)
            ax.plot(steps, smoothed, label=variant, color=style["color"], marker=style["marker"], markevery=max(1, len(steps) // 10))

        ax.set_title(f"Per-Step Throughput (batch size = {batch_size}, workers = {num_workers})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Throughput (samples/s)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best")
        save_figure(fig, output_dir, f"fig_iteration_throughput_bs{batch_size}_nw{num_workers}", formats)


def plot_focused_iteration_throughput(
    traces: list[dict],
    output_dir: str,
    formats: list[str],
    batch_size: int,
    num_workers: int,
):
    if not traces:
        raise ValueError("No trace records were found in the input report.")

    selected = [
        record for record in traces
        if int(record["batch_size"]) == int(batch_size)
        and int(record["num_workers"]) == int(num_workers)
        and record["label"] in FOCUSED_TRACE_VARIANTS
    ]
    if not selected:
        raise ValueError(
            f"No trace records matched batch_size={batch_size}, num_workers={num_workers} "
            f"for variants {FOCUSED_TRACE_VARIANTS}."
        )
    missing_variants = [
        variant for variant in FOCUSED_TRACE_VARIANTS
        if not any(record["label"] == variant for record in selected)
    ]
    if missing_variants:
        raise ValueError(
            f"Missing focused trace variants for batch_size={batch_size}, num_workers={num_workers}: "
            + ", ".join(missing_variants)
        )

    figure_options = {
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.color": "#D7D9E0",
        "grid.alpha": 0.55,
        "grid.linestyle": "--",
        "lines.linewidth": 2.8,
        "lines.markersize": 7.0,
    }

    with plt.rc_context(figure_options):
        fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

        for variant in FOCUSED_TRACE_VARIANTS:
            rows = sorted(
                [record for record in selected if record["label"] == variant],
                key=lambda record: int(record["global_step"]),
            )
            if not rows:
                continue

            steps = np.arange(1, len(rows) + 1, dtype=int)
            throughput = np.array([row["throughput_samples_per_sec"] for row in rows], dtype=float)
            style = focused_trace_style(variant)
            ax.plot(
                steps,
                throughput,
                color=style["color"],
                marker=style["marker"],
                markevery=max(1, len(steps) // 12),
                label=FOCUSED_TRACE_DISPLAY_NAMES.get(variant, variant),
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Throughput (samples/s)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best")
        save_figure(
            fig,
            output_dir,
            f"fig_iteration_throughput_focus_bs{batch_size}_nw{num_workers}",
            formats,
        )


def plot_rocksdb_hit_rate(traces: list[dict], output_dir: str, formats: list[str]):
    rocksdb_traces = [record for record in traces if record["label"] == "RocksDB"]
    if not rocksdb_traces:
        return

    grouped = group_by_config(rocksdb_traces)
    for (batch_size, num_workers), records in grouped.items():
        records = sorted(records, key=lambda record: int(record["global_step"]))
        rates = np.array([record["rocksdb_block_cache_hit_rate"] for record in records], dtype=float)
        valid = np.isfinite(rates)
        if not np.any(valid):
            continue

        steps = np.arange(1, len(records) + 1, dtype=int)
        rates_pct = rates * 100.0
        smoothed = smooth_series(np.where(valid, rates_pct, np.nanmean(rates_pct[valid])), max(5, min(21, len(rates_pct))))

        fig, ax = plt.subplots(figsize=(8.2, 4.2), constrained_layout=True)
        ax.plot(steps, rates_pct, color=VARIANT_COLORS["RocksDB"], alpha=0.2, linewidth=1.0)
        ax.plot(steps, smoothed, color=VARIANT_COLORS["RocksDB"], label="RocksDB block-cache hit rate")
        ax.set_title(f"RocksDB Block-Cache Hit Rate (batch size = {batch_size}, workers = {num_workers})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Hit rate (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="best")
        save_figure(fig, output_dir, f"fig_block_cache_hit_rate_bs{batch_size}_nw{num_workers}", formats)


def plot_rocksdb_compaction(traces: list[dict], output_dir: str, formats: list[str]):
    rocksdb_traces = [record for record in traces if record["label"] == "RocksDB"]
    if not rocksdb_traces:
        return

    grouped = group_by_config(rocksdb_traces)
    for (batch_size, num_workers), records in grouped.items():
        records = sorted(records, key=lambda record: int(record["global_step"]))
        steps = np.arange(1, len(records) + 1, dtype=int)
        total_time = np.array([record["rocksdb_total_compaction_time_sec"] for record in records], dtype=float)
        pending_bytes = np.array([record["rocksdb_estimate_pending_compaction_bytes"] for record in records], dtype=float)
        observed = np.array([record["rocksdb_compaction_observed"] for record in records], dtype=float)
        running = np.array([record["rocksdb_num_running_compactions"] for record in records], dtype=float)

        has_time = np.any(np.isfinite(total_time))
        has_pending = np.any(np.isfinite(pending_bytes))
        has_running = np.any(np.isfinite(running))
        has_observed = np.any(observed > 0.0)
        if not any([has_time, has_pending, has_running, has_observed]):
            continue

        fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.0), sharex=True, constrained_layout=True)

        if has_time:
            time_plot = np.where(np.isfinite(total_time), total_time, np.nan)
            axes[0].plot(steps, time_plot, color="#8A1538")
        axes[0].set_ylabel("Compaction time (s)")
        axes[0].set_title(f"RocksDB Compaction Trace (batch size = {batch_size}, workers = {num_workers})")

        if has_pending:
            pending_mib = np.where(np.isfinite(pending_bytes), pending_bytes / (1024.0 ** 2), np.nan)
            axes[1].plot(steps, pending_mib, color="#6C6F7F", label="Pending compaction (MiB)")
        if has_running:
            axes[1].plot(steps, np.where(np.isfinite(running), running, np.nan), color="#C84C09", label="Running compactions")
        if has_observed:
            axes[1].fill_between(steps, 0.0, observed, color="#8A1538", alpha=0.18, label="Compaction observed")

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Pending / observed")
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].legend(loc="best")
        save_figure(fig, output_dir, f"fig_compaction_bs{batch_size}_nw{num_workers}", formats)


def plot_rocksdb_summary(summary: list[dict], output_dir: str, formats: list[str]):
    rocksdb_summary = [record for record in summary if record["label"] == "RocksDB"]
    if not rocksdb_summary:
        return

    batch_sizes = unique_sorted(rocksdb_summary, "batch_size")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)

    for batch_size in batch_sizes:
        subset = sorted(
            [record for record in rocksdb_summary if int(record["batch_size"]) == batch_size],
            key=lambda record: int(record["num_workers"]),
        )
        if not subset:
            continue

        workers = np.array([record["num_workers"] for record in subset], dtype=int)
        hit_rate = np.array([record["rocksdb_block_cache_hit_rate_mean"] * 100.0 for record in subset], dtype=float)
        compaction_time = np.array([record["rocksdb_compaction_time_total_sec"] for record in subset], dtype=float)

        axes[0].plot(workers, hit_rate, marker="o", label=f"BS={batch_size}")
        axes[1].plot(workers, compaction_time, marker="o", label=f"BS={batch_size}")

    axes[0].set_title("Mean Block-Cache Hit Rate")
    axes[0].set_xlabel("DataLoader workers")
    axes[0].set_ylabel("Hit rate (%)")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100))

    axes[1].set_title("Total Compaction Time")
    axes[1].set_xlabel("DataLoader workers")
    axes[1].set_ylabel("Time (s)")

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("RocksDB Internal Metrics Summary", y=1.03, fontsize=13, fontweight="semibold")
    axes[0].legend(loc="best")
    save_figure(fig, output_dir, "fig_rocksdb_internal_summary", formats)


def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality figures from perf_eval.py output")
    parser.add_argument("--input", type=str, required=True, help="JSON report produced by perf_eval.py")
    parser.add_argument("--output_dir", type=str, default="figures/paper", help="Directory for figure files")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Output file formats")
    parser.add_argument("--focus_batch_size", type=int, help="Batch size for the focused per-step throughput comparison")
    parser.add_argument("--focus_num_workers", type=int, help="num_workers for the focused per-step throughput comparison")
    parser.add_argument("--only_focus_plot", action="store_true", help="Generate only the focused per-step throughput comparison")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    apply_paper_style()
    summary, traces, metadata = load_report(args.input)

    if (args.focus_batch_size is None) != (args.focus_num_workers is None):
        raise ValueError("Both --focus_batch_size and --focus_num_workers must be provided together.")
    if args.only_focus_plot and args.focus_batch_size is None:
        raise ValueError("--only_focus_plot requires --focus_batch_size and --focus_num_workers.")

    if not args.only_focus_plot:
        plot_throughput_overview(summary, args.output_dir, args.formats)
        plot_latency_overview(summary, args.output_dir, args.formats)
        plot_iteration_throughput(traces, args.output_dir, args.formats)
        plot_rocksdb_summary(summary, args.output_dir, args.formats)
        plot_rocksdb_hit_rate(traces, args.output_dir, args.formats)
        plot_rocksdb_compaction(traces, args.output_dir, args.formats)

    if args.focus_batch_size is not None and args.focus_num_workers is not None:
        plot_focused_iteration_throughput(
            traces,
            args.output_dir,
            args.formats,
            batch_size=args.focus_batch_size,
            num_workers=args.focus_num_workers,
        )

    manifest = {
        "input": args.input,
        "output_dir": args.output_dir,
        "formats": args.formats,
        "metadata": metadata,
        "figures_requested": [],
    }
    if not args.only_focus_plot:
        manifest["figures_requested"].extend([
            "throughput_overview",
            "latency_overview",
            "iteration_throughput",
            "rocksdb_internal_summary",
            "block_cache_hit_rate",
            "compaction",
        ])
    if args.focus_batch_size is not None:
        manifest["figures_requested"].append("iteration_throughput_focus")
    manifest["figures_requested"] = [name for name in manifest["figures_requested"] if name is not None]
    with open(os.path.join(args.output_dir, "figure_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
