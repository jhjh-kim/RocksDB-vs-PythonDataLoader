"""
Measure RocksDB block cache hit rate correctly under multi-worker DataLoader
execution by collecting counters inside worker processes and aggregating them
through shared memory.

This script is separate from the legacy benchmark on purpose.

Reported metrics:
  - throughput
  - epoch time
  - batch latency (mean / p50 / p95 / p99)
  - aggregate block cache hit / miss deltas across workers
  - aggregate block cache hit rate

Usage:
    python rocksdb_cache_eval_workers.py \
        --data_dir /path/to/data \
        --subjects sub-01 \
        --db_path /path/to/rocksdb \
        --batch_sizes 64 128 256 \
        --num_workers 0 4 8 16 \
        --epochs 3 \
        --output rocksdb_cache_eval_workers.json
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info


def import_python_rocksdb():
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
        self.db_path = db_path
        self.subjects = subjects

        self.db = None
        self.db_pid = None
        self.keys = self._build_keys()
        if not self.keys:
            raise RuntimeError(f"No data found in RocksDB at {db_path} for subjects {subjects}")

        self._access_count = 0
        self._metric_sample_stride = 1
        self._metric_seen = None
        self._metric_start_hit = None
        self._metric_start_miss = None
        self._metric_latest_hit = None
        self._metric_latest_miss = None

    def _open_db(self):
        opts = rocksdb.Options()
        opts.create_if_missing = False
        return rocksdb.DB(self.db_path, opts, read_only=True)

    def _ensure_db(self):
        pid = os.getpid()
        if self.db is None or self.db_pid != pid:
            self.db = self._open_db()
            self.db_pid = pid
            self._access_count = 0
        return self.db

    def _build_keys(self) -> list[bytes]:
        db = self._open_db()
        keys = []
        for subj in self.subjects:
            i = 0
            while True:
                key = f"{subj}:{i}".encode()
                if db.get(key) is None:
                    break
                keys.append(key)
                i += 1
        del db
        return keys

    def __len__(self):
        return len(self.keys)

    def get_property(self, name: str):
        db = self._ensure_db()
        for query in (name, name.encode()):
            try:
                value = db.get_property(query)
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

    def reset_metrics_buffers(self, num_slots: int, sample_stride: int):
        self._metric_sample_stride = max(1, int(sample_stride))
        self._metric_seen = mp.Array("i", [0] * num_slots, lock=False)
        self._metric_start_hit = mp.Array("d", [float("nan")] * num_slots, lock=False)
        self._metric_start_miss = mp.Array("d", [float("nan")] * num_slots, lock=False)
        self._metric_latest_hit = mp.Array("d", [float("nan")] * num_slots, lock=False)
        self._metric_latest_miss = mp.Array("d", [float("nan")] * num_slots, lock=False)
        self._access_count = 0

    def collect_metrics_snapshot(self) -> dict:
        total_hit = 0.0
        total_miss = 0.0

        if self._metric_seen is None:
            return {
                "block_cache_hit": float("nan"),
                "block_cache_miss": float("nan"),
                "block_cache_hit_rate": float("nan"),
            }

        for slot in range(len(self._metric_seen)):
            if not self._metric_seen[slot]:
                continue

            start_hit = float(self._metric_start_hit[slot])
            start_miss = float(self._metric_start_miss[slot])
            latest_hit = float(self._metric_latest_hit[slot])
            latest_miss = float(self._metric_latest_miss[slot])

            if math.isfinite(start_hit) and math.isfinite(latest_hit):
                total_hit += max(0.0, latest_hit - start_hit)
            if math.isfinite(start_miss) and math.isfinite(latest_miss):
                total_miss += max(0.0, latest_miss - start_miss)

        hit_rate = float("nan")
        if (total_hit + total_miss) > 0.0:
            hit_rate = total_hit / (total_hit + total_miss)

        return {
            "block_cache_hit": total_hit,
            "block_cache_miss": total_miss,
            "block_cache_hit_rate": hit_rate,
        }

    def _update_shared_counters(self):
        if self._metric_seen is None:
            return

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        counters = self.get_block_cache_counters()
        hit = counters["block_cache_hit"]
        miss = counters["block_cache_miss"]

        if not self._metric_seen[worker_id]:
            self._metric_start_hit[worker_id] = hit
            self._metric_start_miss[worker_id] = miss
            self._metric_seen[worker_id] = 1

        self._metric_latest_hit[worker_id] = hit
        self._metric_latest_miss[worker_id] = miss

    def __getitem__(self, index):
        db = self._ensure_db()
        raw = db.get(self.keys[index])
        self._access_count += 1

        if self._access_count == 1 or (self._access_count % self._metric_sample_stride) == 0:
            self._update_shared_counters()

        buf = io.BytesIO(raw)
        sample = torch.load(buf, weights_only=False)
        return {"eeg": sample["eeg"], "label": torch.tensor(sample["label"], dtype=torch.long)}


def benchmark_rocksdb(dataset: RocksDBDataset, batch_size: int, num_workers: int,
                      num_epochs: int, label: str, metrics_sample_stride: int) -> dict:
    total_samples = len(dataset)
    epoch_times = []
    per_batch_latencies = []
    total_hit = 0.0
    total_miss = 0.0

    for _ in range(num_epochs):
        dataset.reset_metrics_buffers(max(1, num_workers), metrics_sample_stride)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )

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

        snapshot = dataset.collect_metrics_snapshot()
        if math.isfinite(snapshot["block_cache_hit"]):
            total_hit += snapshot["block_cache_hit"]
        if math.isfinite(snapshot["block_cache_miss"]):
            total_miss += snapshot["block_cache_miss"]

    epoch_times = np.array(epoch_times, dtype=float)
    batch_lats = np.array(per_batch_latencies, dtype=float)
    throughput_per_epoch = total_samples / epoch_times

    block_cache_hit_rate = float("nan")
    if (total_hit + total_miss) > 0.0:
        block_cache_hit_rate = total_hit / (total_hit + total_miss)

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
        "block_cache_hit": total_hit,
        "block_cache_miss": total_miss,
        "block_cache_hit_rate": block_cache_hit_rate,
    }


def run_experiment(args):
    print("=" * 78)
    print("  RocksDB Read Benchmark with Worker-Aggregated Block Cache Hit Rate")
    print("=" * 78)
    print(f"  data_dir              : {args.data_dir}")
    print(f"  subjects              : {args.subjects}")
    print(f"  mode                  : {args.mode}")
    print(f"  db_path               : {args.db_path}")
    print(f"  batch_sizes           : {args.batch_sizes}")
    print(f"  num_workers           : {args.num_workers}")
    print(f"  epochs                : {args.epochs}")
    print(f"  warmup                : {args.warmup}")
    print(f"  metrics_sample_stride : {args.metrics_sample_stride}")
    print("=" * 78)
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
                benchmark_rocksdb(ds_rdb, bs, nw, num_epochs=1, label="RocksDB_warmup", metrics_sample_stride=args.metrics_sample_stride)

            print(f"  [RocksDB] Benchmarking {args.epochs} epochs ...")
            result = benchmark_rocksdb(ds_rdb, bs, nw, num_epochs=args.epochs, label="RocksDB", metrics_sample_stride=args.metrics_sample_stride)
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
                "metrics_sample_stride": args.metrics_sample_stride,
            },
            "summary": all_results,
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RocksDB throughput/latency with worker-aggregated block cache hit rate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory with subject folders")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs (e.g. sub-01 sub-02)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Data split")
    parser.add_argument("--db_path", type=str, default=None, help="RocksDB database path")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[64, 128, 256], help="Batch sizes to test")
    parser.add_argument("--num_workers", nargs="+", type=int, default=[0, 4, 8, 16], help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=3, help="Number of measured epochs")
    parser.add_argument("--warmup", action="store_true", default=False, help="Run 1 warm-up epoch before measuring")
    parser.add_argument("--metrics_sample_stride", type=int, default=1, help="Read worker-local cache counters every N samples")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")

    args = parser.parse_args()
    args.db_path = resolve_db_path(args.data_dir, args.mode, args.db_path)
    run_experiment(args)


if __name__ == "__main__":
    main()
