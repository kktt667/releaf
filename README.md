# Touch Grass Recovery Demo

This workspace already contains the full end-to-end hackathon flow in `camera_tracking.py`:

- State-driven lifecycle (`ONLINE` -> `WARNING` -> `DECAY` -> `RECOVERY_REQUIRED` -> `OUTSIDE_MODE` -> `PROOF_VERIFIED` -> `MINTED`)
- Session-scoped QR upload routes (`/session/<id>`)
- Absence-gated outdoor proof verification
- Local behavior/value ledger at `data/recovery_ledger.json`
- Optional thirdweb/Base Sepolia mint/update bridge
- Demo-ready runtime profile via `--demo-mode`

## Quick Start (One Command)

```bash
bash run_demo.sh
```

The script will:

1. Create `.venv` if needed
2. Install `requirements.txt`
3. Start `camera_tracking.py --demo-mode`

## Chain Mode (Optional)

To enable live mint/update calls:

```bash
export ENABLE_CHAIN=1
export WALLET_ADDRESS="0xyourwallet"
export THIRDWEB_MINT_URL="https://your-thirdweb-mint-endpoint"
export THIRDWEB_UPDATE_URL="https://your-thirdweb-update-endpoint"
export THIRDWEB_API_KEY="your_api_key_if_needed"
bash run_demo.sh
```

If chain calls fail, the app automatically falls back to local-ledger mode and shows `CHAIN_FALLBACK_LOCAL_LEDGER` for demo reliability.

### Chain Smoke Test (No Camera Needed)

Use this to validate your thirdweb mint/update endpoint wiring quickly:

```bash
.venv/bin/python camera_tracking.py \
  --mint-enabled \
  --wallet-address "$WALLET_ADDRESS" \
  --thirdweb-mint-url "$THIRDWEB_MINT_URL" \
  --thirdweb-update-url "$THIRDWEB_UPDATE_URL" \
  --thirdweb-api-key "$THIRDWEB_API_KEY" \
  --chain-smoke-test
```

If updating an existing token, include:

```bash
--chain-smoke-token-id "<token_id>"
```

## Overlay (Swift)

The app now writes a stage file (`stage.txt`) compatible with your overlay states (`normal`, `warning`, `decay`, `recovery`).

Run overlay:

```bash
bash run_overlay.sh
```

## Absolute Emergency Mode (Local Mint + Public Proof)

If you want zero deployment risk for judging, use local mint simulation with optional NFT.Storage proof links:

```bash
EMERGENCY_MODE=1 RUN_OVERLAY=1 bash run_all.sh
```

Outputs are written under `data/proof_artifacts/`:

- local proof image (`*_proof.jpg`)
- metadata JSON (`*_metadata.json`)
- simulated token + tx hash stored in `data/recovery_ledger.json`

If `NFT_STORAGE_API_KEY` is set, metadata is uploaded and a public IPFS proof URL is generated.

## Camera Permission (macOS)

If startup logs show camera access denied:

```bash
tccutil reset Camera
```

Then rerun `bash run_demo.sh` and allow camera access when prompted.

## Demo Runbook

For stage/demo script and NFT minting narrative, see `DEMO_WORKFLOW.md`.
