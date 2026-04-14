"""
Measure RocksDB read performance together with block cache hit rate,
without touching the legacy benchmark code.

This script benchmarks only the RocksDB-backed dataset and reports:
  - throughput
  - epoch time
  - batch latency (mean / p50 / p95 / p99)
  - block cache hit / miss deltas
  - block cache hit rate during the measured run

Usage:
    python rocksdb_cache_eval.py \
        --subjects sub-01 \
        --data_dir /path/to/data \
        --db_path /path/to/rocksdb \
        --batch_sizes 64 128 256 \
        --num_workers 0 4 8 16 \
        --epochs 3 \
        --output rocksdb_cache_eval.json
"""

import argparse
import importlib
import io
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def import_python_rocksdb():
    """
    Import python-rocksdb even if a local `rocksdb/` directory exists.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = []
    for path in (os.getcwd(), project_root):
        if path and os.path.isdir(os.path.join(path, "rocksdb")):
            candidate_paths.append(path)

    removed_paths = []
    for path in dict.fromkeys(candidate_paths):
        while path in sys.path:
            sys.path.remove(path)
            removed_paths.append(path)

    try:
        sys.modules.pop("rocksdb", None)
        module = importlib.import_module("rocksdb")
    finally:
        for path in reversed(removed_paths):
            sys.path.insert(0, path)

    if not hasattr(module, "DB"):
        module_file = getattr(module, "__file__", None)
        raise ImportError(
            "Failed to import python-rocksdb. "
            f"Imported module file: {module_file!r}"
        )
    return module


rocksdb = import_python_rocksdb()


def resolve_db_path(data_dir: str, mode: str, db_path: str | None) -> str:
    candidates = []
    if db_path:
        candidates.append(db_path)
    else:
        candidates.extend([
            os.path.join(data_dir, f"rocksdb_{mode}"),
            os.path.join(os.getcwd(), f"rocksdb_{mode}"),
            os.path.join(os.getcwd(), "rocksdb"),
        ])

    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            ordered_candidates.append(normalized)

    for candidate in ordered_candidates:
        if os.path.exists(os.path.join(candidate, "CURRENT")):
            return candidate

    raise FileNotFoundError(
        "Could not find a RocksDB database. Checked: " + ", ".join(ordered_candidates)
    )


def decode_property_value(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def coerce_float(value):
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = decode_property_value(value)
    if text is None:
        return float("nan")

    cleaned = text.strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        return float(match.group(0)) if match else float("nan")


class RocksDBDataset(Dataset):
    def __init__(self, db_path: str, subjects: list[str]):
        opts = rocksdb.Options()
        opts.create_if_missing = False
        self.db = rocksdb.DB(db_path, opts, read_only=True)
        self.subjects = subjects

        self.keys = []
        for subj in subjects:
            i = 0
            while True:
                key = f"{subj}:{i}".encode()
                val = self.db.get(key)
                if val is None:
                    break
                self.keys.append(key)
                i += 1

        if not self.keys:
            raise RuntimeError(f"No data found in RocksDB at {db_path} for subjects {subjects}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        raw = self.db.get(key)
        buf = io.BytesIO(raw)
        sample = torch.load(buf, weights_only=False)
        return {"eeg": sample["eeg"], "label": torch.tensor(sample["label"], dtype=torch.long)}

    def get_property(self, name: str):
        for query in (name, name.encode()):
            try:
                value = self.db.get_property(query)
            except Exception:
                continue
            if value is not None:
                return value
        return None

    def get_block_cache_counters(self) -> dict:
        return {
            "block_cache_hit": coerce_float(self.get_property("rocksdb.block-cache-hit")),
            "block_cache_miss": coerce_float(self.get_property("rocksdb.block-cache-miss")),
        }


def benchmark_rocksdb(dataset: Dataset, batch_size: int, num_workers: int,
                      num_epochs: int, label: str) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    total_samples = len(dataset)
    epoch_times = []
    per_batch_latencies = []
    cache_before = dataset.get_block_cache_counters()

    for _ in range(num_epochs):
        batch_latencies = []
        t_epoch_start = time.perf_counter()

        t_batch_start = time.perf_counter()
        for batch in loader:
            t_batch_end = time.perf_counter()
            batch_latencies.append(t_batch_end - t_batch_start)
            _ = batch["eeg"].shape
            _ = batch["label"].shape
            t_batch_start = time.perf_counter()

        t_epoch_end = time.perf_counter()
        epoch_times.append(t_epoch_end - t_epoch_start)
        per_batch_latencies.extend(batch_latencies)

    cache_after = dataset.get_block_cache_counters()

    epoch_times = np.array(epoch_times, dtype=float)
    batch_lats = np.array(per_batch_latencies, dtype=float)
    throughput_per_epoch = total_samples / epoch_times

    delta_hit = float("nan")
    delta_miss = float("nan")
    block_cache_hit_rate = float("nan")
    if math.isfinite(cache_before["block_cache_hit"]) and math.isfinite(cache_after["block_cache_hit"]):
        delta_hit = max(0.0, cache_after["block_cache_hit"] - cache_before["block_cache_hit"])
    if math.isfinite(cache_before["block_cache_miss"]) and math.isfinite(cache_after["block_cache_miss"]):
        delta_miss = max(0.0, cache_after["block_cache_miss"] - cache_before["block_cache_miss"])
    if math.isfinite(delta_hit) and math.isfinite(delta_miss) and (delta_hit + delta_miss) > 0.0:
        block_cache_hit_rate = delta_hit / (delta_hit + delta_miss)

    return {
        "label": label,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": num_epochs,
        "total_samples": total_samples,
        "epoch_time_mean": float(np.mean(epoch_times)),
        "epoch_time_std": float(np.std(epoch_times)),
        "throughput_mean": float(np.mean(throughput_per_epoch)),
        "throughput_std": float(np.std(throughput_per_epoch)),
        "batch_latency_mean": float(np.mean(batch_lats)),
        "batch_latency_p50": float(np.percentile(batch_lats, 50)),
        "batch_latency_p95": float(np.percentile(batch_lats, 95)),
        "batch_latency_p99": float(np.percentile(batch_lats, 99)),
        "block_cache_hit": delta_hit,
        "block_cache_miss": delta_miss,
        "block_cache_hit_rate": block_cache_hit_rate,
    }


def run_experiment(args):
    print("=" * 72)
    print("  RocksDB Read Benchmark with Block Cache Hit Rate")
    print("=" * 72)
    print(f"  data_dir    : {args.data_dir}")
    print(f"  subjects    : {args.subjects}")
    print(f"  mode        : {args.mode}")
    print(f"  db_path     : {args.db_path}")
    print(f"  batch_sizes : {args.batch_sizes}")
    print(f"  num_workers : {args.num_workers}")
    print(f"  epochs      : {args.epochs}")
    print(f"  warmup      : {args.warmup}")
    print("=" * 72)
    print()

    print("[1/1] Opening RocksDBDataset ...")
    t0 = time.perf_counter()
    ds_rdb = RocksDBDataset(args.db_path, args.subjects)
    print(f"       -> {len(ds_rdb)} keys indexed in {time.perf_counter() - t0:.2f}s\n")

    all_results = []

    for bs in args.batch_sizes:
        for nw in args.num_workers:
            print(f"{'─' * 60}")
            print(f"  Config: batch_size={bs}, num_workers={nw}")
            print(f"{'─' * 60}")

            if args.warmup:
                print("  [RocksDB] Warm-up epoch ...")
                benchmark_rocksdb(ds_rdb, bs, nw, num_epochs=1, label="RocksDB_warmup")

            print(f"  [RocksDB] Benchmarking {args.epochs} epochs ...")
            result = benchmark_rocksdb(ds_rdb, bs, nw, num_epochs=args.epochs, label="RocksDB")
            all_results.append(result)

            hit_rate = result["block_cache_hit_rate"]
            hit_rate_str = f"{hit_rate * 100:.2f}%" if math.isfinite(hit_rate) else "n/a"
            print(f"    Throughput : {result['throughput_mean']:>10.1f} ± {result['throughput_std']:>8.1f} samples/s")
            print(f"    Epoch time : {result['epoch_time_mean']:>10.4f} ± {result['epoch_time_std']:>8.4f} s")
            print(
                f"    Batch lat  : mean={result['batch_latency_mean'] * 1000:.3f}ms  "
                f"p50={result['batch_latency_p50'] * 1000:.3f}ms  "
                f"p95={result['batch_latency_p95'] * 1000:.3f}ms  "
                f"p99={result['batch_latency_p99'] * 1000:.3f}ms"
            )
            print(
                f"    Cache hit  : {hit_rate_str}  "
                f"(delta_hit={result['block_cache_hit']:.0f}, delta_miss={result['block_cache_miss']:.0f})"
            )
            print()

    print("\n" + "=" * 112)
    print(f"{'Variant':<10} {'BS':>4} {'NW':>4} {'Throughput (s/s)':>20} {'Epoch (s)':>16} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Cache Hit':>10}")
    print("=" * 112)
    for r in all_results:
        cache_hit_rate = r["block_cache_hit_rate"]
        cache_hit_str = f"{cache_hit_rate * 100:>7.2f}%" if math.isfinite(cache_hit_rate) else f"{'n/a':>8}"
        print(
            f"{r['label']:<10} {r['batch_size']:>4} {r['num_workers']:>4} "
            f"{r['throughput_mean']:>10.1f}±{r['throughput_std']:<8.1f} "
            f"{r['epoch_time_mean']:>7.4f}±{r['epoch_time_std']:<7.4f} "
            f"{r['batch_latency_p50'] * 1000:>10.3f} {r['batch_latency_p95'] * 1000:>10.3f} "
            f"{r['batch_latency_p99'] * 1000:>10.3f} {cache_hit_str:>10}"
        )
    print("=" * 112)

    if args.output:
        payload = {
            "metadata": {
                "data_dir": args.data_dir,
                "subjects": args.subjects,
                "mode": args.mode,
                "db_path": args.db_path,
                "batch_sizes": args.batch_sizes,
                "num_workers": args.num_workers,
                "epochs": args.epochs,
                "warmup": args.warmup,
            },
            "summary": all_results,
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RocksDB throughput/latency with block cache hit rate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory with subject folders")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs (e.g. sub-01 sub-02)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Data split")
    parser.add_argument("--db_path", type=str, default=None, help="RocksDB database path")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[64, 128, 256], help="Batch sizes to test")
    parser.add_argument("--num_workers", nargs="+", type=int, default=[0, 4], help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=3, help="Number of measured epochs")
    parser.add_argument("--warmup", action="store_true", default=False, help="Run 1 warm-up epoch before measuring")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")

    args = parser.parse_args()
    args.db_path = resolve_db_path(args.data_dir, args.mode, args.db_path)
    run_experiment(args)


if __name__ == "__main__":
    main()
