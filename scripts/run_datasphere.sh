#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage: run_datasphere.sh [--config PATH] [--setup-env] [--] [EXTRA ARGS]

Options:
  --config PATH    Путь до файла конфигурации (по умолчанию configs/datasphere.yaml)
  --setup-env      Предварительно выполнить scripts/setup_env.sh
  -h, --help       Показать это сообщение
  --               Передать все оставшиеся аргументы в src/train.py
EOF
}

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
cd "$PROJECT_ROOT"

DEFAULT_CONFIG="configs/datasphere.yaml"
CONFIG="$DEFAULT_CONFIG"
SETUP_ENV=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      shift
      [[ $# -gt 0 ]] || { echo "[ERR] --config требует путь." >&2; exit 1; }
      CONFIG="$1"
      ;;
    --config=*)
      CONFIG="${1#*=}"
      ;;
    --setup-env)
      SETUP_ENV=true
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      if [[ "$CONFIG" == "$DEFAULT_CONFIG" && ! "$1" =~ ^- ]]; then
        CONFIG="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      ;;
  esac
  shift || break
done

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERR] Конфиг '$CONFIG' не найден." >&2
  exit 1
fi

if "$SETUP_ENV"; then
  echo "[INFO] Инициализация окружения через scripts/setup_env.sh"
  bash scripts/setup_env.sh
fi

export DATA_DIR="${DATA_DIR:-/home/jupyter/work/datasets/InterfDataset}"
export RUNS_DIR="${RUNS_DIR:-/home/jupyter/work/runs}"
export TORCH_HOME="${TORCH_HOME:-/home/jupyter/work/.cache/torch}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/jupyter/work/.cache/huggingface}"

mkdir -p "$RUNS_DIR" "$(dirname "$TORCH_HOME")" "$(dirname "$TRANSFORMERS_CACHE")"

echo "[INFO] Используем конфиг: $CONFIG"
echo "[INFO] DATA_DIR=$DATA_DIR"
echo "[INFO] RUNS_DIR=$RUNS_DIR"

python -u src/train.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
