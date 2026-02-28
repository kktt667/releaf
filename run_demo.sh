#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install -r requirements.txt

if [[ "${ENABLE_CHAIN:-0}" == "1" ]]; then
  exec .venv/bin/python camera_tracking.py \
    --demo-mode \
    --mint-enabled \
    --wallet-address "${WALLET_ADDRESS:-demo_wallet}" \
    --thirdweb-mint-url "${THIRDWEB_MINT_URL:-}" \
    --thirdweb-update-url "${THIRDWEB_UPDATE_URL:-}" \
    --thirdweb-api-key "${THIRDWEB_API_KEY:-}" \
    "$@"
fi

exec .venv/bin/python camera_tracking.py --demo-mode "$@"
