"""
Ingest EEG .pt data files into a RocksDB database for benchmarking.

Usage:
    python ingest_to_rocksdb.py --data_dir /path/to/data --subjects sub-01 sub-02 --mode train
"""

import argparse
import io
import os
import time

import numpy as np
import rocksdb
import torch


def serialize_sample(eeg_tensor: torch.Tensor, label: int) -> bytes:
    """Serialize an EEG tensor + label into bytes."""
    buf = io.BytesIO()
    torch.save({"eeg": eeg_tensor, "label": label}, buf)
    return buf.getvalue()


def ingest(data_dir: str, subjects: list[str], mode: str, db_path: str,
           selected_ch: list[str] | None = None, avg: bool = True):
    """Read .pt files and write each sample into RocksDB."""

    channels = [
        'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2',
    ]

    if selected_ch is None:
        selected_ch = channels

    # RocksDB options
    opts = rocksdb.Options()
    opts.create_if_missing = True
    opts.write_buffer_size = 64 * 1024 * 1024  # 64 MB
    opts.max_write_buffer_number = 3
    opts.target_file_size_base = 64 * 1024 * 1024

    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    db = rocksdb.DB(db_path, opts)

    total_samples = 0
    t0 = time.time()

    for subj in subjects:
        pt_path = os.path.join(data_dir, subj, f"{mode}.pt")
        if not os.path.exists(pt_path):
            print(f"  [SKIP] {pt_path} not found")
            continue

        print(f"  Loading {pt_path} ...")
        loaded = torch.load(pt_path, weights_only=False)
        eeg = torch.from_numpy(loaded["eeg"]) if isinstance(loaded["eeg"], np.ndarray) else loaded["eeg"]

        # Channel selection
        selected_idx = [channels.index(ch) for ch in selected_ch]
        eeg = eeg[:, :, selected_idx]

        if avg:
            eeg = eeg.mean(axis=1)  # average over trials → (n_stimuli, C, T)
            labels = loaded["label"][:, 0] if loaded["label"].ndim > 1 else loaded["label"]
        else:
            eeg = eeg.reshape(-1, *eeg.shape[2:])
            labels = loaded["label"].reshape(-1) if hasattr(loaded["label"], "reshape") else loaded["label"]

        n = eeg.shape[0]
        batch = rocksdb.WriteBatch()
        for i in range(n):
            key = f"{subj}:{i}".encode()
            val = serialize_sample(eeg[i].float(), int(labels[i]))
            batch.put(key, val)

        db.write(batch)
        total_samples += n
        print(f"  Wrote {n} samples for {subj}")

    elapsed = time.time() - t0

    # Store metadata
    db.put(b"__meta__:subjects", ",".join(subjects).encode())
    db.put(b"__meta__:total_samples", str(total_samples).encode())
    db.put(b"__meta__:mode", mode.encode())

    del db  # close

    print(f"\nIngestion complete: {total_samples} samples in {elapsed:.2f}s")
    print(f"DB path: {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest EEG data into RocksDB")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory containing subject folders")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subject IDs, e.g. sub-01 sub-02")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Data split mode")
    parser.add_argument("--db_path", type=str, default=None,
                        help="Output RocksDB path (default: <data_dir>/rocksdb_<mode>)")
    parser.add_argument("--avg", action="store_true", default=True,
                        help="Average over trials (default: True)")
    parser.add_argument("--no_avg", action="store_true",
                        help="Do NOT average over trials")

    args = parser.parse_args()

    db_path = args.db_path or os.path.join(args.data_dir, f"rocksdb_{args.mode}")
    avg = not args.no_avg

    print(f"=== RocksDB Ingestion ===")
    print(f"  data_dir : {args.data_dir}")
    print(f"  subjects : {args.subjects}")
    print(f"  mode     : {args.mode}")
    print(f"  avg      : {avg}")
    print(f"  db_path  : {db_path}")
    print()

    ingest(args.data_dir, args.subjects, args.mode, db_path, avg=avg)


if __name__ == "__main__":
    main()
