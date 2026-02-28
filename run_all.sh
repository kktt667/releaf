#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install -r requirements.txt

if [[ "${RESET_DEMO_STATE:-1}" == "1" ]]; then
  rm -f "data/recovery_ledger.json" || true
  rm -f "data/simulated_chain.jsonl" || true
  rm -rf "data/proof_artifacts" || true
  mkdir -p "data/proof_artifacts"
  echo "normal" > stage.txt 2>/dev/null || true
  echo "[run_all] Demo state reset."
fi

OVERLAY_PID=""
cleanup() {
  echo "normal" > stage.txt 2>/dev/null || true
  if [[ -n "${OVERLAY_PID}" ]]; then
    kill "${OVERLAY_PID}" >/dev/null 2>&1 || true
  fi
  pkill -f "/plant_overlay" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ "${RUN_OVERLAY:-0}" == "1" ]]; then
  if command -v swiftc >/dev/null 2>&1; then
    pkill -f "/plant_overlay" >/dev/null 2>&1 || true
    if [[ ! -f "plant_overlay" || "plant_overlay.swift" -nt "plant_overlay" ]]; then
      swiftc plant_overlay.swift -o plant_overlay
    fi
    ./plant_overlay > data/overlay.log 2>&1 &
    OVERLAY_PID="$!"
    echo "[run_all] Overlay started (pid ${OVERLAY_PID})."
  else
    echo "[run_all] swiftc not found, skipping overlay."
  fi
else
  echo "[run_all] Swift overlay disabled (single-window mode)."
fi

CMD=(.venv/bin/python camera_tracking.py --demo-mode)
CMD+=(--camera-index "${CAMERA_INDEX:-0}")
CMD+=(--upload-port "${UPLOAD_PORT:-8088}")
CMD+=(--training-folder "${TRAINING_FOLDER:-photos}")
CMD+=(--flowers-folder "${FLOWERS_FOLDER:-flowers}")

if [[ -n "${UPLOAD_HOST_IP:-}" ]]; then
  CMD+=(--upload-host-ip "${UPLOAD_HOST_IP}")
fi

if [[ "${ENABLE_CHAIN:-0}" == "1" ]]; then
  CMD+=(--mint-enabled)
  CMD+=(--wallet-address "${WALLET_ADDRESS:-demo_wallet}")
  CMD+=(--thirdweb-mint-url "${THIRDWEB_MINT_URL:-}")
  CMD+=(--thirdweb-update-url "${THIRDWEB_UPDATE_URL:-}")
  CMD+=(--thirdweb-api-key "${THIRDWEB_API_KEY:-}")
fi

if [[ "${EMERGENCY_MODE:-0}" == "1" ]]; then
  CMD+=(--emergency-mode)
  CMD+=(--wallet-address "${WALLET_ADDRESS:-demo_wallet}")
  if [[ -n "${NFT_STORAGE_API_KEY:-}" ]]; then
    CMD+=(--nft-storage-api-key "${NFT_STORAGE_API_KEY}")
  fi
fi

echo "[run_all] Starting app..."
"${CMD[@]}" "$@"
