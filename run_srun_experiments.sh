#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  SRUN_ARGS="--partition your-partition --account your-account" \
  bash run_srun_experiments.sh \
    --data_dir /path/to/data \
    --subjects sub-01 [sub-02 ...] \
    --db_path /path/to/rocksdb \
    [--mode train] \
    [--cpu_budgets 1 2 4] \
    [--num_workers 0 1 2 4 8 16] \
    [--batch_size 128] \
    [--epochs 3] \
    [--repeats 3] \
    [--output_root paper_runs_srun] \
    [--python python]

This wrapper launches one srun job per CPU budget, for example:
  srun -c 1 ...
  srun -c 2 ...
  srun -c 4 ...

Optional extra srun flags can be injected with the SRUN_ARGS environment
variable, for example:
  SRUN_ARGS="--partition cpu --account mylab --time 02:00:00"

Each run writes results to:
  <output_root>/cpu1
  <output_root>/cpu2
  <output_root>/cpu4

After all runs complete, the script merges them into:
  <output_root>/merged
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR=""
DB_PATH=""
MODE="train"
OUTPUT_ROOT="paper_runs_srun"
PYTHON_BIN="/gpfs/data/oermannlab/users/jk8865/.conda/thought2txt/bin/python"
BATCH_SIZE="128"
EPOCHS="3"
REPEATS="3"
ROCKSDB_BLOCK_CACHE_MB="512"
METRICS_SAMPLE_STRIDE="1"
TRACE_CPU_BUDGET=""
TRACE_NUM_WORKERS=""
NO_AVG="0"
WARMUP="0"
DISABLE_ROCKSDB_METRICS="0"
SRUN_ARGS_STRING="${SRUN_ARGS:-}"
declare -a SRUN_ARGS_ARRAY=()
if [[ -n "$SRUN_ARGS_STRING" ]]; then
  read -r -a SRUN_ARGS_ARRAY <<< "$SRUN_ARGS_STRING"
fi

declare -a SUBJECTS=()
declare -a CPU_BUDGETS=("1" "2" "4")
declare -a NUM_WORKERS=("0" "1" "2" "4" "8" "16")

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
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
    --trace_cpu_budget)
      TRACE_CPU_BUDGET="$2"
      shift 2
      ;;
    --trace_num_workers)
      TRACE_NUM_WORKERS="$2"
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
    --cpu_budgets)
      shift
      CPU_BUDGETS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CPU_BUDGETS+=("$1")
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

if [[ -z "$DATA_DIR" || -z "$DB_PATH" || "${#SUBJECTS[@]}" -eq 0 ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

declare -a RUN_DIRS=()

for CPU_BUDGET in "${CPU_BUDGETS[@]}"; do
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/cpu${CPU_BUDGET}"
  RUN_DIRS+=("$RUN_OUTPUT_DIR")

  CMD=(
    srun
    "${SRUN_ARGS_ARRAY[@]}"
    -c "$CPU_BUDGET"
    "$PYTHON_BIN"
    "${SCRIPT_DIR}/extra_experiments.py"
    --data_dir "$DATA_DIR"
    --subjects "${SUBJECTS[@]}"
    --mode "$MODE"
    --db_path "$DB_PATH"
    --cpu_budgets "$CPU_BUDGET"
    --num_workers "${NUM_WORKERS[@]}"
    --batch_size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --repeats "$REPEATS"
    --output_dir "$RUN_OUTPUT_DIR"
    --rocksdb_block_cache_mb "$ROCKSDB_BLOCK_CACHE_MB"
    --metrics_sample_stride "$METRICS_SAMPLE_STRIDE"
  )

  if [[ "$TRACE_CPU_BUDGET" == "$CPU_BUDGET" ]]; then
    CMD+=(--trace_cpu_budget "$TRACE_CPU_BUDGET")
  fi
  if [[ -n "$TRACE_NUM_WORKERS" ]]; then
    CMD+=(--trace_num_workers "$TRACE_NUM_WORKERS")
  fi
  if [[ "$NO_AVG" == "1" ]]; then
    CMD+=(--no_avg)
  fi
  if [[ "$WARMUP" == "1" ]]; then
    CMD+=(--warmup)
  fi
  if [[ "$DISABLE_ROCKSDB_METRICS" == "1" ]]; then
    CMD+=(--disable_rocksdb_metrics)
  fi

  echo "============================================================"
  echo "Launching CPU budget ${CPU_BUDGET}"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  echo "============================================================"
  "${CMD[@]}"
done

MERGED_OUTPUT_DIR="${OUTPUT_ROOT}/merged"
MERGE_CMD=(
  "$PYTHON_BIN"
  "${SCRIPT_DIR}/merge_extra_experiments.py"
  --run_dirs "${RUN_DIRS[@]}"
  --output_dir "$MERGED_OUTPUT_DIR"
)

if [[ -n "$TRACE_CPU_BUDGET" ]]; then
  MERGE_CMD+=(--trace_cpu_budget "$TRACE_CPU_BUDGET")
fi
if [[ -n "$TRACE_NUM_WORKERS" ]]; then
  MERGE_CMD+=(--trace_num_workers "$TRACE_NUM_WORKERS")
fi

echo "============================================================"
echo "Merging run directories"
printf 'Command:'
printf ' %q' "${MERGE_CMD[@]}"
printf '\n'
echo "============================================================"
"${MERGE_CMD[@]}"

echo
echo "Done. Merged paper artifacts are in: ${MERGED_OUTPUT_DIR}"
