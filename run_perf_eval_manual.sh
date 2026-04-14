#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_perf_eval_manual.sh \
    --cpu_label cpu1 \
    --data_dir /path/to/data \
    --subjects sub-01 [sub-02 ...] \
    --db_path /path/to/rocksdb \
    [--mode train] \
    [--batch_sizes 64] \
    [--num_workers 0 1 2 4 8] \
    [--epochs 3] \
    [--output_root perf_eval_runs] \
    [--python python3]

This script is intended to be run manually inside a separately allocated
environment, for example:

  srun -c 1 --mem=128GB --pty bash
  bash run_perf_eval_manual.sh --cpu_label cpu1 ...

  srun -c 2 --mem=128GB --pty bash
  bash run_perf_eval_manual.sh --cpu_label cpu2 ...

It writes:
  <output_root>/<cpu_label>/perf_eval.json
  <output_root>/<cpu_label>/perf_eval.log
  <output_root>/<cpu_label>/run_manifest.txt
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CPU_LABEL=""
DATA_DIR=""
DB_PATH=""
MODE="train"
OUTPUT_ROOT="perf_eval_runs"
PYTHON_BIN="python3"
EPOCHS="3"
NO_AVG="0"
WARMUP="0"
SKIP_EXHAUSTIVE="1"
DISABLE_ROCKSDB_METRICS="0"
ROCKSDB_BLOCK_CACHE_MB="256"
METRICS_SAMPLE_STRIDE="64"

declare -a SUBJECTS=()
declare -a BATCH_SIZES=("64")
declare -a NUM_WORKERS=("0" "1" "2" "4" "8")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu_label)
      CPU_LABEL="$2"
      shift 2
      ;;
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --db_path)
      DB_PATH="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --rocksdb_block_cache_mb)
      ROCKSDB_BLOCK_CACHE_MB="$2"
      shift 2
      ;;
    --metrics_sample_stride)
      METRICS_SAMPLE_STRIDE="$2"
      shift 2
      ;;
    --no_avg)
      NO_AVG="1"
      shift
      ;;
    --warmup)
      WARMUP="1"
      shift
      ;;
    --use_exhaustive)
      SKIP_EXHAUSTIVE="0"
      shift
      ;;
    --disable_rocksdb_metrics)
      DISABLE_ROCKSDB_METRICS="1"
      shift
      ;;
    --subjects)
      shift
      SUBJECTS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        SUBJECTS+=("$1")
        shift
      done
      ;;
    --batch_sizes)
      shift
      BATCH_SIZES=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        BATCH_SIZES+=("$1")
        shift
      done
      ;;
    --num_workers)
      shift
      NUM_WORKERS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        NUM_WORKERS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CPU_LABEL" || -z "$DATA_DIR" || -z "$DB_PATH" || "${#SUBJECTS[@]}" -eq 0 ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

RUN_DIR="${OUTPUT_ROOT}/${CPU_LABEL}"
mkdir -p "$RUN_DIR"

JSON_PATH="${RUN_DIR}/perf_eval.json"
LOG_PATH="${RUN_DIR}/perf_eval.log"
MANIFEST_PATH="${RUN_DIR}/run_manifest.txt"

CMD=(
  "$PYTHON_BIN"
  "${SCRIPT_DIR}/perf_eval.py"
  --data_dir "$DATA_DIR"
  --subjects "${SUBJECTS[@]}"
  --mode "$MODE"
  --db_path "$DB_PATH"
  --batch_sizes "${BATCH_SIZES[@]}"
  --num_workers "${NUM_WORKERS[@]}"
  --epochs "$EPOCHS"
  --output "$JSON_PATH"
  --rocksdb_block_cache_mb "$ROCKSDB_BLOCK_CACHE_MB"
  --metrics_sample_stride "$METRICS_SAMPLE_STRIDE"
)

if [[ "$NO_AVG" == "1" ]]; then
  CMD+=(--no_avg)
fi
if [[ "$WARMUP" == "1" ]]; then
  CMD+=(--warmup)
fi
if [[ "$SKIP_EXHAUSTIVE" == "1" ]]; then
  CMD+=(--skip_exhaustive)
fi
if [[ "$DISABLE_ROCKSDB_METRICS" == "1" ]]; then
  CMD+=(--disable_rocksdb_metrics)
fi

{
  echo "CPU_LABEL=${CPU_LABEL}"
  echo "PWD=$(pwd)"
  echo "HOSTNAME=$(hostname)"
  echo "DATE=$(date)"
  echo "VISIBLE_CPUS=$(python3 - <<'PY'
import os
if hasattr(os, "sched_getaffinity"):
    print(sorted(os.sched_getaffinity(0)))
else:
    print("unknown")
PY
)"
  echo "COMMAND="
  printf ' %q' "${CMD[@]}"
  printf '\n'
} > "$MANIFEST_PATH"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "Saving JSON to: ${JSON_PATH}"
echo "Saving log to : ${LOG_PATH}"

"${CMD[@]}" 2>&1 | tee "$LOG_PATH"

echo
echo "Done. Results saved under: ${RUN_DIR}"
