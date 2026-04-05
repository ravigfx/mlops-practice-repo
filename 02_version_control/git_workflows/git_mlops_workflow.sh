#!/usr/bin/env bash
# ============================================================
#  Git Workflows for MLOps Teams
#  Covers: branching strategies, commit conventions,
#          pre-commit hooks, tag-based model versioning
# ============================================================

# ── Branching Strategy (Gitflow for ML) ──────────────────────
#
#  main          ← production-ready models & code only
#  develop       ← integration branch
#  feature/xxx   ← new features or experiments
#  experiment/xxx ← ML experiments (may be discarded)
#  hotfix/xxx    ← urgent production fixes
#  release/vX.Y  ← release candidates

setup_repo() {
    git init
    git checkout -b develop
    echo "✅ Repo initialized with develop branch"
}

# ── Conventional Commits for ML Projects ─────────────────────
#
#  feat:     new model feature or capability
#  fix:      bug fix in pipeline or training code
#  data:     data changes (new dataset, schema update)
#  model:    model architecture or hyperparameter changes
#  perf:     performance improvement
#  refactor: code refactoring (no functionality change)
#  test:     add or fix tests
#  ci:       CI/CD pipeline changes
#  docs:     documentation only changes
#  chore:    maintenance (deps, config)
#
#  Examples:
#    git commit -m "model: increase n_estimators from 100 to 200"
#    git commit -m "data: add validation split for time series"
#    git commit -m "feat: add SHAP explainability to serving API"
#    git commit -m "fix: handle null values in feature pipeline"
#    git commit -m "perf: cache feature store calls in batch mode"

# ── Feature Branch Workflow ───────────────────────────────────
new_feature() {
    local name="$1"
    git checkout develop
    git pull origin develop
    git checkout -b "feature/$name"
    echo "✅ Feature branch: feature/$name"
}

new_experiment() {
    local name="$1"
    git checkout develop
    git checkout -b "experiment/$name"
    echo "✅ Experiment branch: experiment/$name"
}

# ── Model Versioning with Git Tags ───────────────────────────
# Tag format: model-<name>-v<MAJOR>.<MINOR>.<PATCH>
# MAJOR: breaking change (new architecture)
# MINOR: improvement (new features, retraining)
# PATCH: bug fix or minor update

tag_model() {
    local model_name="$1"
    local version="$2"
    local message="$3"
    git tag -a "model-${model_name}-v${version}" -m "$message"
    git push origin "model-${model_name}-v${version}"
    echo "✅ Tagged: model-${model_name}-v${version}"
}

# Example:
# tag_model "iris-classifier" "1.2.0" "Retrained with additional 500 samples, +2% accuracy"

# ── Pre-commit Hook (save as .git/hooks/pre-commit) ──────────
write_precommit_hook() {
cat > .git/hooks/pre-commit << 'EOF'
#!/usr/bin/env bash
set -e
echo "Running pre-commit checks..."

# 1. Black formatting
black --check src/ tests/ || { echo "❌ Black check failed. Run: black src/ tests/"; exit 1; }

# 2. Flake8 linting
flake8 src/ tests/ --max-line-length=100 || { echo "❌ Flake8 failed."; exit 1; }

# 3. Unit tests (fast only)
pytest tests/ -m "not slow" -q || { echo "❌ Tests failed."; exit 1; }

# 4. Ensure model files are tracked by DVC, not Git
if git diff --cached --name-only | grep -qE "\.(pkl|joblib|h5|pt|onnx)$"; then
    echo "❌ Model binary files should be tracked by DVC, not Git!"
    echo "   Run: dvc add <model_file>"
    exit 1
fi

echo "✅ All pre-commit checks passed!"
EOF
chmod +x .git/hooks/pre-commit
echo "Pre-commit hook installed"
}

# ── DVC + Git Integration ─────────────────────────────────────
dvc_git_workflow() {
    echo """
    DVC + Git combined workflow:

    1. Stage data change with DVC:
       dvc add data/raw/new_data.csv
       git add data/raw/new_data.csv.dvc

    2. Run pipeline:
       dvc repro

    3. Commit everything together:
       git add dvc.lock metrics/ plots/
       git commit -m 'data: add Q4 training data, retrain model'

    4. Push code to Git, data to remote:
       git push
       dvc push

    5. Tag the release:
       git tag -a v1.3.0 -m 'Q4 model release'
       git push origin v1.3.0
    """
}

# ── Release Checklist ─────────────────────────────────────────
release_checklist() {
    echo """
    ✅ ML Model Release Checklist:
    [ ] All unit tests passing
    [ ] Model accuracy >= threshold (check metrics/eval_metrics.json)
    [ ] No data drift detected in test set
    [ ] Model registered in MLflow with version tag
    [ ] Docker image built and pushed to registry
    [ ] CHANGELOG.md updated
    [ ] Rollback plan documented
    [ ] Monitoring alerts configured
    [ ] Stakeholder sign-off received
    """
}

case "${1:-help}" in
    setup)      setup_repo ;;
    feature)    new_feature "$2" ;;
    experiment) new_experiment "$2" ;;
    tag-model)  tag_model "$2" "$3" "$4" ;;
    pre-commit) write_precommit_hook ;;
    dvc)        dvc_git_workflow ;;
    release)    release_checklist ;;
    *)
        echo "Usage: $0 {setup|feature|experiment|tag-model|pre-commit|dvc|release}"
        ;;
esac
