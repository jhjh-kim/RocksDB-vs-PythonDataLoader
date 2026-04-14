SRUN_ARGS="-p cpu_dev --time 02:00:00 --mem=128GB" \
bash run_srun_experiments.sh \
  --data_dir /gpfs/data/oermannlab/users/jk8865/ATS/data/things-eeg/Preprocessed_data_250Hz_whiten \
  --subjects sub-01 \
  --db_path /gpfs/data/oermannlab/users/jk8865/RocksDB-vs-PythonDataLoader/rocksdb \
  --cpu_budgets 1 2 4 \
  --num_workers 0 1 2 4 8 \
  --batch_size 64 \
  --epochs 3 \
  --repeats 3 \
  --output_root paper_runs_srun \
  --python /gpfs/data/oermannlab/users/jk8865/.conda/thought2txt/bin/python