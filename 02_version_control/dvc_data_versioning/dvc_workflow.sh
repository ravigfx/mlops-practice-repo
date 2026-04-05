#!/usr/bin/env bash
# ============================================================
#  DVC (Data Version Control) — Practice Workflow
#  Mirrors Git but for data, models, and large files
# ============================================================

# ── SETUP ────────────────────────────────────────────────────
# pip install dvc dvc-s3
# git init && dvc init

# ── 1. Track a dataset ───────────────────────────────────────
track_data() {
    mkdir -p data/raw
    # Add your CSV here
    dvc add data/raw/dataset.csv
    git add data/raw/dataset.csv.dvc data/.gitignore
    git commit -m "Track raw dataset with DVC"
}

# ── 2. Define a pipeline (dvc.yaml) ──────────────────────────
# dvc.yaml is the pipeline definition file
create_pipeline_yaml() {
cat > dvc.yaml << 'EOF'
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/dataset.csv
      - src/preprocess.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed/train.csv
      - src/train.py
    params:
      - params.yaml:
        - train.n_estimators
        - train.max_depth
    outs:
      - models/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/test.csv
      - src/evaluate.py
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - plots/confusion_matrix.csv
EOF
echo "dvc.yaml created"
}

# ── 3. params.yaml ───────────────────────────────────────────
create_params_yaml() {
cat > params.yaml << 'EOF'
train:
  n_estimators: 100
  max_depth: 5
  random_state: 42
  test_size: 0.2
EOF
echo "params.yaml created"
}

# ── 4. Run pipeline ──────────────────────────────────────────
run_dvc_pipeline() {
    dvc repro              # runs only changed stages
}

# ── 5. Experiment comparison ─────────────────────────────────
compare_experiments() {
    dvc exp run --set-param train.n_estimators=200
    dvc exp show           # table of all experiments
    dvc exp diff           # diff vs baseline
}

# ── 6. Remote storage setup ──────────────────────────────────
setup_remote_s3() {
    dvc remote add -d myremote s3://your-bucket/dvc-store
    dvc remote modify myremote region us-east-1
    dvc push    # push data to S3
    dvc pull    # pull data from S3
}

echo "DVC workflow scripts ready. Run functions individually."
