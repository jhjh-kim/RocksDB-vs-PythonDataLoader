"""
Performance Evaluation: Read Throughput, Latency, and RocksDB Internals

Compares three EEG data loading strategies:
  1) EEGDataset  - pre-loaded into memory, indexed lookup
  2) Exhaustive  - re-reads .pt file from disk on every __getitem__
  3) RocksDB     - key-value lookup via python-rocksdb

The benchmark writes a paper-friendly JSON report with:
  - summary throughput / latency statistics
  - per-step throughput traces
  - RocksDB block-cache / compaction traces (when available)

Usage:
    python perf_eval.py \
        --subjects sub-01 \
        --mode train \
        --batch_sizes 32 64 128 256 \
        --num_workers 0 4 8 16 \
        --epochs 3 \
        --output benchmark_report.json \
        --skip_exhaustive
"""

import argparse
import io
import json
import math
import os
import re
import time

import numpy as np
import rocksdb
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info


CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

TRACE_METRIC_FIELDS = (
    "worker_id",
    "sampled",
    "block_cache_hit",
    "block_cache_miss",
    "block_cache_hit_rate",
    "compaction_pending",
    "num_running_compactions",
    "estimate_pending_compaction_bytes",
    "total_compaction_time_sec",
    "compaction_observed",
)


def safe_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_std(values) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr)) if arr.size else float("nan")


def safe_percentile(values, q: float) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def empty_trace_metrics(worker_id: int = -1, sampled: float = 0.0) -> dict[str, float]:
    metrics = {field: float("nan") for field in TRACE_METRIC_FIELDS}
    metrics["worker_id"] = float(worker_id)
    metrics["sampled"] = float(sampled)
    metrics["compaction_observed"] = 0.0
    return metrics


def decode_property_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def coerce_metric_value(value) -> float:
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


def extract_metric_from_stats(stats_text: str, patterns: list[str], scale: float = 1.0) -> float:
    if not stats_text:
        return float("nan")

    for pattern in patterns:
        match = re.search(pattern, stats_text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", "")) * scale
            except ValueError:
                continue
    return float("nan")


def parse_rocksdb_stats(stats_text: str) -> dict[str, float]:
    parsed = {
        "block_cache_hit": extract_metric_from_stats(
            stats_text,
            [
                r"block cache hit(?: count)?\s*[:=]\s*([0-9][0-9,\.]*)",
                r"BLOCK_CACHE_HIT[^0-9]*([0-9][0-9,\.]*)",
            ],
        ),
        "block_cache_miss": extract_metric_from_stats(
            stats_text,
            [
                r"block cache miss(?: count)?\s*[:=]\s*([0-9][0-9,\.]*)",
                r"BLOCK_CACHE_MISS[^0-9]*([0-9][0-9,\.]*)",
            ],
        ),
        "block_cache_hit_rate": extract_metric_from_stats(
            stats_text,
            [
                r"block cache hit rate\s*[:=]\s*([0-9][0-9,\.]*)\s*%",
                r"block cache hit rate\s*[:=]\s*([0-9][0-9,\.]*)",
            ],
            scale=0.01,
        ),
        "total_compaction_time_sec": extract_metric_from_stats(
            stats_text,
            [
                r"compaction[^.\n]*?([0-9][0-9,\.]*)\s*(?:secs|sec|seconds)",
                r"compaction time(?:\s*\(sec\))?\s*[:=]\s*([0-9][0-9,\.]*)",
            ],
        ),
    }
    return parsed


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_rocksdb_step_metrics(batch_trace_metrics: dict, previous_by_worker: dict[int, dict[str, float]]) -> dict[str, float]:
    step_metrics = {
        "rocksdb_block_cache_hit_rate": float("nan"),
        "rocksdb_delta_block_cache_hit": float("nan"),
        "rocksdb_delta_block_cache_miss": float("nan"),
        "rocksdb_compaction_pending": float("nan"),
        "rocksdb_num_running_compactions": float("nan"),
        "rocksdb_estimate_pending_compaction_bytes": float("nan"),
        "rocksdb_total_compaction_time_sec": float("nan"),
        "rocksdb_compaction_time_delta_sec": float("nan"),
        "rocksdb_compaction_observed": 0.0,
    }

    if not batch_trace_metrics:
        return step_metrics

    worker_ids = tensor_to_numpy(batch_trace_metrics["worker_id"]).astype(int)
    valid_positions = np.where(worker_ids >= 0)[0]
    if valid_positions.size == 0:
        return step_metrics

    delta_hits = 0.0
    delta_misses = 0.0
    delta_compaction_time = 0.0
    total_compaction_time = 0.0
    current_pending = []
    current_running = []
    current_pending_bytes = []
    saw_counter_delta = False
    saw_compaction_time = False

    for worker_id in sorted(set(worker_ids[valid_positions].tolist())):
        worker_positions = np.where(worker_ids == worker_id)[0]
        last_pos = int(worker_positions[-1])
        current = {
            "block_cache_hit": float(tensor_to_numpy(batch_trace_metrics["block_cache_hit"])[last_pos]),
            "block_cache_miss": float(tensor_to_numpy(batch_trace_metrics["block_cache_miss"])[last_pos]),
            "block_cache_hit_rate": float(tensor_to_numpy(batch_trace_metrics["block_cache_hit_rate"])[last_pos]),
            "compaction_pending": float(tensor_to_numpy(batch_trace_metrics["compaction_pending"])[last_pos]),
            "num_running_compactions": float(tensor_to_numpy(batch_trace_metrics["num_running_compactions"])[last_pos]),
            "estimate_pending_compaction_bytes": float(
                tensor_to_numpy(batch_trace_metrics["estimate_pending_compaction_bytes"])[last_pos]
            ),
            "total_compaction_time_sec": float(tensor_to_numpy(batch_trace_metrics["total_compaction_time_sec"])[last_pos]),
        }

        previous = previous_by_worker.get(worker_id)

        if previous is not None:
            if math.isfinite(current["block_cache_hit"]) and math.isfinite(previous["block_cache_hit"]):
                delta_hits += max(0.0, current["block_cache_hit"] - previous["block_cache_hit"])
                saw_counter_delta = True
            if math.isfinite(current["block_cache_miss"]) and math.isfinite(previous["block_cache_miss"]):
                delta_misses += max(0.0, current["block_cache_miss"] - previous["block_cache_miss"])
                saw_counter_delta = True
            if math.isfinite(current["total_compaction_time_sec"]) and math.isfinite(previous["total_compaction_time_sec"]):
                delta_compaction_time += max(0.0, current["total_compaction_time_sec"] - previous["total_compaction_time_sec"])
                saw_compaction_time = True

        if math.isfinite(current["compaction_pending"]):
            current_pending.append(current["compaction_pending"])
        if math.isfinite(current["num_running_compactions"]):
            current_running.append(current["num_running_compactions"])
        if math.isfinite(current["estimate_pending_compaction_bytes"]):
            current_pending_bytes.append(current["estimate_pending_compaction_bytes"])
        if math.isfinite(current["total_compaction_time_sec"]):
            total_compaction_time += current["total_compaction_time_sec"]

        previous_by_worker[worker_id] = current

    if saw_counter_delta and (delta_hits + delta_misses) > 0:
        step_metrics["rocksdb_block_cache_hit_rate"] = delta_hits / (delta_hits + delta_misses)
        step_metrics["rocksdb_delta_block_cache_hit"] = delta_hits
        step_metrics["rocksdb_delta_block_cache_miss"] = delta_misses
    else:
        current_rates = tensor_to_numpy(batch_trace_metrics["block_cache_hit_rate"]).astype(float)
        finite_rates = current_rates[np.isfinite(current_rates)]
        if finite_rates.size:
            step_metrics["rocksdb_block_cache_hit_rate"] = float(np.mean(finite_rates))

    if current_pending:
        step_metrics["rocksdb_compaction_pending"] = float(max(current_pending))
    if current_running:
        step_metrics["rocksdb_num_running_compactions"] = float(max(current_running))
    if current_pending_bytes:
        step_metrics["rocksdb_estimate_pending_compaction_bytes"] = float(max(current_pending_bytes))
    if total_compaction_time > 0.0:
        step_metrics["rocksdb_total_compaction_time_sec"] = float(total_compaction_time)
    if saw_compaction_time:
        step_metrics["rocksdb_compaction_time_delta_sec"] = float(delta_compaction_time)

    compaction_observed = (
        (math.isfinite(step_metrics["rocksdb_compaction_time_delta_sec"]) and step_metrics["rocksdb_compaction_time_delta_sec"] > 0.0)
        or (math.isfinite(step_metrics["rocksdb_num_running_compactions"]) and step_metrics["rocksdb_num_running_compactions"] > 0.0)
        or (math.isfinite(step_metrics["rocksdb_compaction_pending"]) and step_metrics["rocksdb_compaction_pending"] > 0.0)
    )
    step_metrics["rocksdb_compaction_observed"] = 1.0 if compaction_observed else 0.0
    return step_metrics


class EEGDatasetBench(Dataset):
    """
    Mirrors the loading logic of the original EEGDataset in data.py,
    but strips it down to EEG tensors + labels only.
    Data is fully loaded into memory at __init__.
    """

    CHANNELS = CHANNELS

    def __init__(self, data_dir: str, subjects: list[str], mode: str,
                 selected_ch: list[str] | None = None, avg: bool = True):
        self.data_dir = data_dir
        self.subjects = subjects
        self.mode = mode
        self.selected_ch = selected_ch or self.CHANNELS
        self.avg = avg

        self.all_eeg = []
        self.all_labels = []

        for subj in subjects:
            pt_path = os.path.join(data_dir, subj, f"{mode}.pt")
            loaded = torch.load(pt_path, weights_only=False)
            eeg = torch.from_numpy(loaded["eeg"]) if isinstance(loaded["eeg"], np.ndarray) else loaded["eeg"]

            idx = [self.CHANNELS.index(ch) for ch in self.selected_ch]
            eeg = eeg[:, :, idx]

            if avg:
                eeg = eeg.mean(axis=1)
                labels = loaded["label"][:, 0] if loaded["label"].ndim > 1 else loaded["label"]
            else:
                eeg = eeg.reshape(-1, *eeg.shape[2:])
                labels = loaded["label"].reshape(-1) if hasattr(loaded["label"], "reshape") else loaded["label"]

            self.all_eeg.append(eeg.float())
            label_tensor = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels.long()
            self.all_labels.append(label_tensor)

        self.all_eeg = torch.cat(self.all_eeg, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)

    def __len__(self):
        return len(self.all_eeg)

    def __getitem__(self, index):
        return {"eeg": self.all_eeg[index], "label": self.all_labels[index]}


class ExhaustiveSearchDataset(Dataset):
    """
    On every __getitem__, reads the entire .pt file from disk
    and extracts the requested sample.
    """

    CHANNELS = CHANNELS

    def __init__(self, data_dir: str, subjects: list[str], mode: str,
                 selected_ch: list[str] | None = None, avg: bool = True):
        self.data_dir = data_dir
        self.subjects = subjects
        self.mode = mode
        self.selected_ch = selected_ch or self.CHANNELS
        self.avg = avg

        self._subject_sizes = []
        for subj in subjects:
            pt_path = os.path.join(data_dir, subj, f"{mode}.pt")
            loaded = torch.load(pt_path, weights_only=False)
            eeg = loaded["eeg"]
            n_stimuli = eeg.shape[0]
            if avg:
                self._subject_sizes.append(n_stimuli)
            else:
                self._subject_sizes.append(n_stimuli * eeg.shape[1])

        self._total = sum(self._subject_sizes)
        self._cum = np.cumsum([0] + self._subject_sizes)

    def __len__(self):
        return self._total

    def __getitem__(self, index):
        subj_idx = int(np.searchsorted(self._cum[1:], index, side="right"))
        local_idx = index - self._cum[subj_idx]

        subj = self.subjects[subj_idx]
        pt_path = os.path.join(self.data_dir, subj, f"{self.mode}.pt")
        loaded = torch.load(pt_path, weights_only=False)
        eeg = torch.from_numpy(loaded["eeg"]) if isinstance(loaded["eeg"], np.ndarray) else loaded["eeg"]

        ch_idx = [self.CHANNELS.index(ch) for ch in self.selected_ch]
        eeg = eeg[:, :, ch_idx]

        if self.avg:
            eeg = eeg.mean(axis=1)
            labels = loaded["label"][:, 0] if loaded["label"].ndim > 1 else loaded["label"]
        else:
            eeg = eeg.reshape(-1, *eeg.shape[2:])
            labels = loaded["label"].reshape(-1)

        return {
            "eeg": eeg[local_idx].float(),
            "label": torch.tensor(int(labels[local_idx]), dtype=torch.long),
        }


class RocksDBDataset(Dataset):
    """
    Reads samples from RocksDB.

    The actual DB handle is opened lazily in each process so that
    multi-worker DataLoader runs can collect worker-local internal metrics.
    """

    def __init__(
        self,
        db_path: str,
        subjects: list[str],
        block_cache_size_mb: int = 512,
        max_open_files: int = -1,
        collect_internal_metrics: bool = True,
        metrics_sample_stride: int = 1,
    ):
        self.db_path = db_path
        self.subjects = subjects
        self.block_cache_size_mb = block_cache_size_mb
        self.max_open_files = max_open_files
        self.collect_internal_metrics = collect_internal_metrics
        self.metrics_sample_stride = max(1, metrics_sample_stride)

        self._db = None
        self._db_pid = None
        self._access_count = 0
        self._cached_trace_metrics = empty_trace_metrics()
        self._block_cache = None

        self.keys = self._build_keys()
        if not self.keys:
            raise RuntimeError(f"No data found in RocksDB at {db_path} for subjects {subjects}")

    def _make_db_options(self):
        opts = rocksdb.Options()
        opts.create_if_missing = False
        opts.max_open_files = self.max_open_files

        if hasattr(rocksdb, "BlockBasedTableFactory") and hasattr(rocksdb, "LRUCache"):
            self._block_cache = rocksdb.LRUCache(self.block_cache_size_mb * 1024 * 1024)
            opts.table_factory = rocksdb.BlockBasedTableFactory(block_cache=self._block_cache)

        return opts

    def _open_db(self):
        return rocksdb.DB(self.db_path, self._make_db_options(), read_only=True)

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

    def _ensure_db(self):
        pid = os.getpid()
        if self._db is None or self._db_pid != pid:
            self._db = self._open_db()
            self._db_pid = pid
            self._access_count = 0
            self._cached_trace_metrics = empty_trace_metrics()
        return self._db

    def _get_property(self, property_name: str):
        db = self._ensure_db()
        for query in (property_name, property_name.encode()):
            try:
                value = db.get_property(query)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
            if value is not None:
                return value
        return None

    def _snapshot_trace_metrics(self) -> dict[str, float]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        metrics = empty_trace_metrics(worker_id=worker_id, sampled=1.0)

        direct_properties = {
            "block_cache_hit": "rocksdb.block-cache-hit",
            "block_cache_miss": "rocksdb.block-cache-miss",
            "block_cache_hit_rate": "rocksdb.block-cache-hit-rate",
            "compaction_pending": "rocksdb.compaction-pending",
            "num_running_compactions": "rocksdb.num-running-compactions",
            "estimate_pending_compaction_bytes": "rocksdb.estimate-pending-compaction-bytes",
            "total_compaction_time_sec": "rocksdb.total-compaction-time",
        }

        for field, property_name in direct_properties.items():
            metrics[field] = coerce_metric_value(self._get_property(property_name))

        stats_text = decode_property_value(self._get_property("rocksdb.stats")) or ""
        parsed_stats = parse_rocksdb_stats(stats_text)
        for field, value in parsed_stats.items():
            if not math.isfinite(metrics[field]) and math.isfinite(value):
                metrics[field] = value

        if math.isfinite(metrics["block_cache_hit"]) and math.isfinite(metrics["block_cache_miss"]):
            denom = metrics["block_cache_hit"] + metrics["block_cache_miss"]
            metrics["block_cache_hit_rate"] = metrics["block_cache_hit"] / denom if denom > 0.0 else float("nan")
        elif math.isfinite(metrics["block_cache_hit_rate"]) and metrics["block_cache_hit_rate"] > 1.0:
            metrics["block_cache_hit_rate"] = metrics["block_cache_hit_rate"] / 100.0

        compaction_observed = (
            (math.isfinite(metrics["compaction_pending"]) and metrics["compaction_pending"] > 0.0)
            or (math.isfinite(metrics["num_running_compactions"]) and metrics["num_running_compactions"] > 0.0)
            or (math.isfinite(metrics["total_compaction_time_sec"]) and metrics["total_compaction_time_sec"] > 0.0)
        )
        metrics["compaction_observed"] = 1.0 if compaction_observed else 0.0
        return metrics

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        db = self._ensure_db()
        raw = db.get(self.keys[index])
        self._access_count += 1

        buf = io.BytesIO(raw)
        sample = torch.load(buf, weights_only=False)

        trace_metrics = empty_trace_metrics()
        if self.collect_internal_metrics:
            should_sample = self._access_count == 1 or (self._access_count % self.metrics_sample_stride) == 0
            if should_sample:
                self._cached_trace_metrics = self._snapshot_trace_metrics()
            trace_metrics = dict(self._cached_trace_metrics)

        return {
            "eeg": sample["eeg"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "trace_metrics": trace_metrics,
        }


def benchmark_dataloader(dataset: Dataset, batch_size: int, num_workers: int,
                         num_epochs: int, label: str) -> tuple[dict, list[dict]]:
    """
    Run a DataLoader for `num_epochs` epochs and collect summary stats
    plus per-step traces for paper-quality figures.
    """
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
    per_epoch_throughput = []
    step_traces = []
    previous_metrics_by_worker = {}
    global_step = 0
    run_start = time.perf_counter()

    for epoch in range(num_epochs):
        batch_latencies = []
        epoch_start = time.perf_counter()
        t_batch_start = time.perf_counter()

        for step_in_epoch, batch in enumerate(loader, start=1):
            t_batch_end = time.perf_counter()
            batch_latency = t_batch_end - t_batch_start
            batch_latencies.append(batch_latency)
            per_batch_latencies.append(batch_latency)

            batch_samples = int(batch["label"].shape[0])
            throughput = batch_samples / batch_latency if batch_latency > 0.0 else float("nan")

            trace_record = {
                "label": label,
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "epoch": int(epoch + 1),
                "step_in_epoch": int(step_in_epoch),
                "global_step": int(global_step),
                "samples_in_batch": int(batch_samples),
                "batch_latency_sec": float(batch_latency),
                "throughput_samples_per_sec": float(throughput),
                "epoch_elapsed_sec": float(t_batch_end - epoch_start),
                "run_elapsed_sec": float(t_batch_end - run_start),
                "rocksdb_block_cache_hit_rate": float("nan"),
                "rocksdb_delta_block_cache_hit": float("nan"),
                "rocksdb_delta_block_cache_miss": float("nan"),
                "rocksdb_compaction_pending": float("nan"),
                "rocksdb_num_running_compactions": float("nan"),
                "rocksdb_estimate_pending_compaction_bytes": float("nan"),
                "rocksdb_total_compaction_time_sec": float("nan"),
                "rocksdb_compaction_time_delta_sec": float("nan"),
                "rocksdb_compaction_observed": 0.0,
            }

            if isinstance(batch, dict) and "trace_metrics" in batch:
                trace_record.update(build_rocksdb_step_metrics(batch["trace_metrics"], previous_metrics_by_worker))

            step_traces.append(trace_record)
            global_step += 1

            _ = batch["eeg"].shape
            _ = batch["label"].shape
            t_batch_start = time.perf_counter()

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        per_epoch_throughput.append(total_samples / epoch_time if epoch_time > 0.0 else float("nan"))

    batch_lats = np.asarray(per_batch_latencies, dtype=float)
    block_cache_rates = [
        record["rocksdb_block_cache_hit_rate"]
        for record in step_traces
        if math.isfinite(record["rocksdb_block_cache_hit_rate"])
    ]
    compaction_deltas = [
        record["rocksdb_compaction_time_delta_sec"]
        for record in step_traces
        if math.isfinite(record["rocksdb_compaction_time_delta_sec"])
    ]
    pending_compaction_bytes = [
        record["rocksdb_estimate_pending_compaction_bytes"]
        for record in step_traces
        if math.isfinite(record["rocksdb_estimate_pending_compaction_bytes"])
    ]
    running_compactions = [
        record["rocksdb_num_running_compactions"]
        for record in step_traces
        if math.isfinite(record["rocksdb_num_running_compactions"])
    ]

    result = {
        "label": label,
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "num_epochs": int(num_epochs),
        "total_samples": int(total_samples),
        "trace_length": int(len(step_traces)),
        "epoch_time_mean": safe_mean(epoch_times),
        "epoch_time_std": safe_std(epoch_times),
        "throughput_mean": safe_mean(per_epoch_throughput),
        "throughput_std": safe_std(per_epoch_throughput),
        "batch_latency_mean": safe_mean(batch_lats),
        "batch_latency_p50": safe_percentile(batch_lats, 50),
        "batch_latency_p95": safe_percentile(batch_lats, 95),
        "batch_latency_p99": safe_percentile(batch_lats, 99),
        "rocksdb_block_cache_hit_rate_mean": safe_mean(block_cache_rates),
        "rocksdb_compaction_time_total_sec": float(np.sum(compaction_deltas)) if compaction_deltas else float("nan"),
        "rocksdb_compaction_observed": bool(any(record["rocksdb_compaction_observed"] > 0.0 for record in step_traces)),
        "rocksdb_peak_pending_compaction_bytes": max(pending_compaction_bytes) if pending_compaction_bytes else float("nan"),
        "rocksdb_peak_running_compactions": max(running_compactions) if running_compactions else float("nan"),
    }
    return result, step_traces


def run_experiment(args):
    print("=" * 70)
    print("  Read Performance Benchmark: EEGDataset vs Exhaustive vs RocksDB")
    print("=" * 70)
    print(f"  data_dir              : {args.data_dir}")
    print(f"  subjects              : {args.subjects}")
    print(f"  mode                  : {args.mode}")
    print(f"  db_path               : {args.db_path}")
    print(f"  batch_sizes           : {args.batch_sizes}")
    print(f"  num_workers           : {args.num_workers}")
    print(f"  epochs                : {args.epochs}")
    print(f"  warmup                : {args.warmup}")
    print(f"  avg                   : {not args.no_avg}")
    print(f"  skip_exhaustive       : {args.skip_exhaustive}")
    print(f"  rocksdb_block_cache_mb: {args.rocksdb_block_cache_mb}")
    print(f"  metrics_sample_stride : {args.metrics_sample_stride}")
    print("=" * 70)
    print()

    avg = not args.no_avg
    n_variants = 2 if args.skip_exhaustive else 3
    step = 0

    step += 1
    print(f"[{step}/{n_variants}] Loading EEGDatasetBench (pre-loaded into memory) ...")
    t0 = time.perf_counter()
    ds_eeg = EEGDatasetBench(args.data_dir, args.subjects, args.mode, avg=avg)
    print(f"       -> {len(ds_eeg)} samples loaded in {time.perf_counter() - t0:.2f}s\n")

    variants = [("EEGDataset", ds_eeg)]

    if not args.skip_exhaustive:
        step += 1
        print(f"[{step}/{n_variants}] Initializing ExhaustiveSearchDataset ...")
        t0 = time.perf_counter()
        ds_exh = ExhaustiveSearchDataset(args.data_dir, args.subjects, args.mode, avg=avg)
        print(f"       -> {len(ds_exh)} samples indexed in {time.perf_counter() - t0:.2f}s\n")
        variants.append(("Exhaustive", ds_exh))

    step += 1
    print(f"[{step}/{n_variants}] Opening RocksDBDataset ...")
    t0 = time.perf_counter()
    ds_rdb = RocksDBDataset(
        args.db_path,
        args.subjects,
        block_cache_size_mb=args.rocksdb_block_cache_mb,
        max_open_files=args.rocksdb_max_open_files,
        collect_internal_metrics=not args.disable_rocksdb_metrics,
        metrics_sample_stride=args.metrics_sample_stride,
    )
    print(f"       -> {len(ds_rdb)} keys indexed in {time.perf_counter() - t0:.2f}s\n")
    variants.append(("RocksDB", ds_rdb))

    all_results = []
    all_traces = []

    for batch_size in args.batch_sizes:
        for num_workers in args.num_workers:
            print(f"{'─' * 60}")
            print(f"  Config: batch_size={batch_size}, num_workers={num_workers}")
            print(f"{'─' * 60}")

            for name, dataset in variants:
                if args.warmup:
                    print(f"  [{name}] Warm-up epoch ...")
                    benchmark_dataloader(dataset, batch_size, num_workers, num_epochs=1, label=f"{name}_warmup")

                print(f"  [{name}] Benchmarking {args.epochs} epochs ...")
                result, traces = benchmark_dataloader(dataset, batch_size, num_workers, num_epochs=args.epochs, label=name)
                all_results.append(result)
                all_traces.extend(traces)

                print(
                    f"    Throughput : {result['throughput_mean']:>10.1f} ± {result['throughput_std']:>8.1f} samples/s"
                )
                print(
                    f"    Epoch time : {result['epoch_time_mean']:>10.4f} ± {result['epoch_time_std']:>8.4f} s"
                )
                print(
                    f"    Batch lat  : mean={result['batch_latency_mean'] * 1000:.3f}ms  "
                    f"p50={result['batch_latency_p50'] * 1000:.3f}ms  "
                    f"p95={result['batch_latency_p95'] * 1000:.3f}ms  "
                    f"p99={result['batch_latency_p99'] * 1000:.3f}ms"
                )

                if name == "RocksDB":
                    hit_rate = result["rocksdb_block_cache_hit_rate_mean"]
                    hit_rate_str = f"{hit_rate * 100:.2f}%" if math.isfinite(hit_rate) else "n/a"
                    compaction_time = result["rocksdb_compaction_time_total_sec"]
                    compaction_str = f"{compaction_time:.4f}s" if math.isfinite(compaction_time) else "n/a"
                    print(f"    Block cache hit rate : {hit_rate_str}")
                    print(f"    Compaction observed  : {result['rocksdb_compaction_observed']}")
                    print(f"    Compaction time sum  : {compaction_str}")
                print()

    print("\n" + "=" * 120)
    print(
        f"{'Variant':<14} {'BS':>4} {'NW':>4} {'Throughput (s/s)':>20} "
        f"{'Epoch (s)':>16} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10} {'Cache Hit':>11} {'Compact':>10}"
    )
    print("=" * 120)
    for result in all_results:
        hit_rate = result["rocksdb_block_cache_hit_rate_mean"]
        compaction_time = result["rocksdb_compaction_time_total_sec"]
        hit_rate_str = f"{hit_rate * 100:>7.2f}%" if math.isfinite(hit_rate) else f"{'n/a':>8}"
        compaction_str = f"{compaction_time:>7.3f}" if math.isfinite(compaction_time) else f"{'n/a':>7}"
        print(
            f"{result['label']:<14} {result['batch_size']:>4} {result['num_workers']:>4} "
            f"{result['throughput_mean']:>10.1f}±{result['throughput_std']:<8.1f} "
            f"{result['epoch_time_mean']:>7.4f}±{result['epoch_time_std']:<7.4f} "
            f"{result['batch_latency_p50'] * 1000:>10.3f} {result['batch_latency_p95'] * 1000:>10.3f} "
            f"{result['batch_latency_p99'] * 1000:>10.3f} {hit_rate_str:>11} {compaction_str:>10}"
        )
    print("=" * 120)

    payload = {
        "metadata": {
            "data_dir": args.data_dir,
            "subjects": args.subjects,
            "mode": args.mode,
            "db_path": args.db_path,
            "batch_sizes": args.batch_sizes,
            "num_workers": args.num_workers,
            "epochs": args.epochs,
            "avg": avg,
            "skip_exhaustive": args.skip_exhaustive,
            "rocksdb_block_cache_mb": args.rocksdb_block_cache_mb,
            "rocksdb_max_open_files": args.rocksdb_max_open_files,
            "metrics_sample_stride": args.metrics_sample_stride,
            "collect_rocksdb_metrics": not args.disable_rocksdb_metrics,
        },
        "summary": all_results,
        "traces": all_traces,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nResults saved to {args.output}")

    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark read throughput & latency: EEGDataset vs Exhaustive vs RocksDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/gpfs/data/oermannlab/users/jk8865/ATS/data/things-eeg/Preprocessed_data_250Hz_whiten",
        help="Root data directory with subject folders",
    )
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject IDs (e.g. sub-01 sub-02)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Data split")
    parser.add_argument("--db_path", type=str, default=None, help="RocksDB database path")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 32, 128], help="Batch sizes to test")
    parser.add_argument("--num_workers", nargs="+", type=int, default=[0, 4], help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=3, help="Number of measured epochs")
    parser.add_argument("--warmup", action="store_true", default=False, help="Run 1 warm-up epoch before measuring")
    parser.add_argument("--no_avg", action="store_true", help="Do NOT average over trials")
    parser.add_argument("--skip_exhaustive", action="store_true", default=False, help="Skip the exhaustive variant")
    parser.add_argument("--output", type=str, default=None, help="Path to save the JSON report")
    parser.add_argument(
        "--rocksdb_block_cache_mb",
        type=int,
        default=512,
        help="Per-process RocksDB block-cache capacity in MiB for the benchmark",
    )
    parser.add_argument(
        "--rocksdb_max_open_files",
        type=int,
        default=-1,
        help="RocksDB max_open_files for each process (-1 keeps all files open)",
    )
    parser.add_argument(
        "--metrics_sample_stride",
        type=int,
        default=1,
        help="Snapshot RocksDB properties every N sample fetches",
    )
    parser.add_argument(
        "--disable_rocksdb_metrics",
        action="store_true",
        default=False,
        help="Disable collection of RocksDB block-cache / compaction properties",
    )

    args = parser.parse_args()
    if args.db_path is None:
        args.db_path = os.path.join(args.data_dir, f"rocksdb_{args.mode}")

    run_experiment(args)


if __name__ == "__main__":
    main()
