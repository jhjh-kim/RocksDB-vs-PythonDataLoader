"""
Performance Evaluation: Read Throughput & Latency Benchmark

Compares three EEG data loading strategies:
  1) EEGDataset  – pre-loaded into memory, indexed lookup
  2) Exhaustive  – re-reads .pt file from disk on every __getitem__
  3) RocksDB     – key-value lookup via python-rocksdb

Usage:
    python perf_eval.py \
        --subjects sub-01 \
        --mode train \
        --batch_sizes 32 64 128 256 \
        --num_workers 0 4 8 16 \
        --epochs 3 \
        --skip_exhaustive
"""

import argparse
import io
import json
import os
import time
from collections import defaultdict

import numpy as np
import rocksdb
import torch
from torch.utils.data import DataLoader, Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Variant 1 — EEGDataset (pre-loaded, indexed lookup)
# ──────────────────────────────────────────────────────────────────────────────

class EEGDatasetBench(Dataset):
    """
    Mirrors the loading logic of the original EEGDataset in data.py,
    but stripped down to EEG tensors + labels only (no images/text).
    Data is fully loaded into memory at __init__.
    """

    CHANNELS = [
        'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2',
    ]

    def __init__(self, data_dir: str, subjects: list[str], mode: str,
                 selected_ch: list[str] | None = None, avg: bool = True):
        self.data_dir = data_dir
        self.subjects = subjects
        self.mode = mode
        self.selected_ch = selected_ch or self.CHANNELS
        self.avg = avg

        # Pre-load all data
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
            self.all_labels.append(torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels.long())

        self.all_eeg = torch.cat(self.all_eeg, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)

    def __len__(self):
        return len(self.all_eeg)

    def __getitem__(self, index):
        return {"eeg": self.all_eeg[index], "label": self.all_labels[index]}


# ──────────────────────────────────────────────────────────────────────────────
# Variant 2 — Exhaustive Search (re-reads .pt file every call)
# ──────────────────────────────────────────────────────────────────────────────

class ExhaustiveSearchDataset(Dataset):
    """
    On every __getitem__, reads the entire .pt file from disk
    and extracts the requested sample.  This is the worst-case scenario.
    """

    CHANNELS = EEGDatasetBench.CHANNELS

    def __init__(self, data_dir: str, subjects: list[str], mode: str,
                 selected_ch: list[str] | None = None, avg: bool = True):
        self.data_dir = data_dir
        self.subjects = subjects
        self.mode = mode
        self.selected_ch = selected_ch or self.CHANNELS
        self.avg = avg

        # We still need to know total length & per-subject sizes
        self._subject_sizes = []
        for subj in subjects:
            pt_path = os.path.join(data_dir, subj, f"{mode}.pt")
            loaded = torch.load(pt_path, weights_only=False)
            eeg = loaded["eeg"]
            if isinstance(eeg, np.ndarray):
                n_stimuli = eeg.shape[0]
            else:
                n_stimuli = eeg.shape[0]
            if avg:
                self._subject_sizes.append(n_stimuli)
            else:
                n_trials = eeg.shape[1]
                self._subject_sizes.append(n_stimuli * n_trials)

        self._total = sum(self._subject_sizes)
        # Build cumulative offsets
        self._cum = np.cumsum([0] + self._subject_sizes)

    def __len__(self):
        return self._total

    def __getitem__(self, index):
        # Determine which subject file this index belongs to
        subj_idx = int(np.searchsorted(self._cum[1:], index, side="right"))
        local_idx = index - self._cum[subj_idx]

        subj = self.subjects[subj_idx]
        pt_path = os.path.join(self.data_dir, subj, f"{self.mode}.pt")

        # --- THIS IS THE EXPENSIVE PART: re-read from disk every time ---
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

        eeg_sample = eeg[local_idx].float()
        label = int(labels[local_idx])

        return {"eeg": eeg_sample, "label": torch.tensor(label, dtype=torch.long)}


# ──────────────────────────────────────────────────────────────────────────────
# Variant 3 — RocksDB DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class RocksDBDataset(Dataset):
    """
    Reads samples from a RocksDB database.  Each __getitem__ does a
    db.get(key) and deserializes the tensor.
    """

    def __init__(self, db_path: str, subjects: list[str]):
        opts = rocksdb.Options()
        opts.create_if_missing = False
        self.db = rocksdb.DB(db_path, opts, read_only=True)
        self.subjects = subjects

        # Build the list of keys
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


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark Harness
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_dataloader(dataset: Dataset, batch_size: int, num_workers: int,
                         num_epochs: int, label: str) -> dict:
    """
    Run a DataLoader for `num_epochs` epochs and collect timing stats.

    Returns a dict with throughput and latency metrics.
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

    for ep in range(num_epochs):
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
        epoch_time = t_epoch_end - t_epoch_start
        epoch_times.append(epoch_time)
        per_batch_latencies.extend(batch_latencies)

    # Compute stats
    epoch_times = np.array(epoch_times)
    batch_lats = np.array(per_batch_latencies)

    throughput_per_epoch = total_samples / epoch_times  # samples/sec per epoch
    result = {
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
    }
    return result


def run_experiment(args):
    """Run the full benchmark across all variants and configurations."""
    print("=" * 70)
    print("  Read Performance Benchmark: EEGDataset vs Exhaustive vs RocksDB")
    print("=" * 70)
    print(f"  data_dir    : {args.data_dir}")
    print(f"  subjects    : {args.subjects}")
    print(f"  mode        : {args.mode}")
    print(f"  db_path     : {args.db_path}")
    print(f"  batch_sizes : {args.batch_sizes}")
    print(f"  num_workers : {args.num_workers}")
    print(f"  epochs      : {args.epochs}")
    print(f"  warmup      : {args.warmup}")
    print(f"  avg              : {not args.no_avg}")
    print(f"  skip_exhaustive  : {args.skip_exhaustive}")
    print("=" * 70)
    print()

    avg = not args.no_avg
    n_variants = 2 if args.skip_exhaustive else 3
    step = 0

    # ── Construct datasets ──────────────────────────────────────────────
    step += 1
    print(f"[{step}/{n_variants}] Loading EEGDatasetBench (pre-loaded into memory) ...")
    t0 = time.perf_counter()
    ds_eeg = EEGDatasetBench(args.data_dir, args.subjects, args.mode, avg=avg)
    t_load_eeg = time.perf_counter() - t0
    print(f"       → {len(ds_eeg)} samples loaded in {t_load_eeg:.2f}s\n")

    variants = [("EEGDataset", ds_eeg)]

    if not args.skip_exhaustive:
        step += 1
        print(f"[{step}/{n_variants}] Initializing ExhaustiveSearchDataset ...")
        t0 = time.perf_counter()
        ds_exh = ExhaustiveSearchDataset(args.data_dir, args.subjects, args.mode, avg=avg)
        t_load_exh = time.perf_counter() - t0
        print(f"       → {len(ds_exh)} samples indexed in {t_load_exh:.2f}s\n")
        variants.append(("Exhaustive", ds_exh))

    step += 1
    print(f"[{step}/{n_variants}] Opening RocksDBDataset ...")
    t0 = time.perf_counter()
    ds_rdb = RocksDBDataset(args.db_path, args.subjects)
    t_load_rdb = time.perf_counter() - t0
    print(f"       → {len(ds_rdb)} keys indexed in {t_load_rdb:.2f}s\n")
    variants.append(("RocksDB", ds_rdb))

    # ── Run benchmarks ──────────────────────────────────────────────────

    all_results = []

    for bs in args.batch_sizes:
        for nw in args.num_workers:
            print(f"{'─' * 60}")
            print(f"  Config: batch_size={bs}, num_workers={nw}")
            print(f"{'─' * 60}")

            for name, ds in variants:
                # Optional warm-up
                if args.warmup:
                    print(f"  [{name}] Warm-up epoch ...")
                    benchmark_dataloader(ds, bs, nw, num_epochs=1, label=f"{name}_warmup")

                print(f"  [{name}] Benchmarking {args.epochs} epochs ...")
                result = benchmark_dataloader(ds, bs, nw, num_epochs=args.epochs, label=name)
                all_results.append(result)

                print(f"    Throughput : {result['throughput_mean']:>10.1f} ± {result['throughput_std']:>8.1f} samples/s")
                print(f"    Epoch time : {result['epoch_time_mean']:>10.4f} ± {result['epoch_time_std']:>8.4f} s")
                print(f"    Batch lat  : mean={result['batch_latency_mean']*1000:.3f}ms  "
                      f"p50={result['batch_latency_p50']*1000:.3f}ms  "
                      f"p95={result['batch_latency_p95']*1000:.3f}ms  "
                      f"p99={result['batch_latency_p99']*1000:.3f}ms")
                print()

    # ── Summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'Variant':<14} {'BS':>4} {'NW':>4} {'Throughput (s/s)':>20} {'Epoch (s)':>16} {'P50 (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}")
    print("=" * 100)
    for r in all_results:
        print(f"{r['label']:<14} {r['batch_size']:>4} {r['num_workers']:>4} "
              f"{r['throughput_mean']:>10.1f}±{r['throughput_std']:<8.1f} "
              f"{r['epoch_time_mean']:>7.4f}±{r['epoch_time_std']:<7.4f} "
              f"{r['batch_latency_p50']*1000:>10.3f} {r['batch_latency_p95']*1000:>10.3f} {r['batch_latency_p99']*1000:>10.3f}")
    print("=" * 100)

    # ── Save results ────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark read throughput & latency: EEGDataset vs Exhaustive vs RocksDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_dir", type=str, default="/gpfs/data/oermannlab/users/jk8865/ATS/data/things-eeg/Preprocessed_data_250Hz_whiten",
                        help="Root data directory with subject folders")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subject IDs (e.g. sub-01 sub-02)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Data split (default: train)")
    parser.add_argument("--db_path", type=str, default="rocksdb",
                        help="RocksDB database path (default: <data_dir>/rocksdb_<mode>)")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 32, 128],
                        help="Batch sizes to test (default: 1 32 128)")
    parser.add_argument("--num_workers", nargs="+", type=int, default=[0, 4],
                        help="Number of DataLoader workers (default: 0 4)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of measured epochs (default: 3)")
    parser.add_argument("--warmup", action="store_true", default=False,
                        help="Run 1 warm-up epoch before measuring (default: off)")
    parser.add_argument("--no_avg", action="store_true",
                        help="Do NOT average over trials")
    parser.add_argument("--skip_exhaustive", action="store_true", default=False,
                        help="Skip the Exhaustive (worst-case) dataset variant")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (optional)")

    args = parser.parse_args()
    args.db_path = args.db_path or os.path.join(args.data_dir, f"rocksdb_{args.mode}")

    run_experiment(args)


if __name__ == "__main__":
    main()
