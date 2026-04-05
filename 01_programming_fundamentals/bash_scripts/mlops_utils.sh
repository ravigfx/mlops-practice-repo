#!/usr/bin/env bash
# ============================================================
#  MLOps Bash Essentials
#  Practice: env setup, data download, model pipeline scripts
# ============================================================
set -euo pipefail   # exit on error, undefined vars, pipe failures

# ── 1. Environment Setup ─────────────────────────────────────
setup_env() {
    echo "🔧 Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Environment ready"
}

# ── 2. Data Download & Validation ────────────────────────────
download_data() {
    local url="$1"
    local dest="${2:-data/raw}"
    mkdir -p "$dest"
    echo "📥 Downloading data from $url..."
    curl -fsSL "$url" -o "$dest/dataset.csv"
    echo "Lines in dataset: $(wc -l < "$dest/dataset.csv")"
}

# ── 3. Run ML Pipeline Stages ────────────────────────────────
run_pipeline() {
    local stage="${1:-all}"
    echo "🚀 Running pipeline stage: $stage"

    case "$stage" in
        preprocess)
            python src/preprocess.py ;;
        train)
            python src/train.py ;;
        evaluate)
            python src/evaluate.py ;;
        all)
            python src/preprocess.py
            python src/train.py
            python src/evaluate.py
            ;;
        *)
            echo "Unknown stage: $stage" && exit 1 ;;
    esac
    echo "✅ Stage '$stage' complete"
}

# ── 4. Docker Helpers ────────────────────────────────────────
docker_build_push() {
    local image_name="$1"
    local tag="${2:-latest}"
    echo "🐳 Building Docker image: $image_name:$tag"
    docker build -t "$image_name:$tag" .
    docker push "$image_name:$tag"
}

# ── 5. Log Monitoring ────────────────────────────────────────
tail_logs() {
    local log_file="${1:-logs/training.log}"
    echo "📋 Tailing $log_file (Ctrl+C to stop)..."
    tail -f "$log_file"
}

# ── 6. Cleanup Artifacts ─────────────────────────────────────
cleanup() {
    echo "🧹 Cleaning up..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true
    echo "✅ Cleanup complete"
}

# ── 7. Health Check ──────────────────────────────────────────
health_check() {
    local url="${1:-http://localhost:8000/health}"
    echo "💊 Checking API health at $url..."
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    if [ "$response" = "200" ]; then
        echo "✅ API is healthy"
    else
        echo "❌ API returned HTTP $response"
        exit 1
    fi
}

# ── CLI Entry Point ───────────────────────────────────────────
main() {
    local cmd="${1:-help}"
    shift || true
    case "$cmd" in
        setup)         setup_env ;;
        download)      download_data "$@" ;;
        pipeline)      run_pipeline "$@" ;;
        docker)        docker_build_push "$@" ;;
        logs)          tail_logs "$@" ;;
        cleanup)       cleanup ;;
        health)        health_check "$@" ;;
        help|*)
            echo "Usage: $0 {setup|download|pipeline|docker|logs|cleanup|health}"
            ;;
    esac
}

main "$@"
