# TouchGrass Demo Workflow

This is the exact demo flow to show in 60-90 seconds.

## 1) Start clean

```bash
bash run_all.sh
```

`run_all.sh` resets prior state by default (`RESET_DEMO_STATE=1`), so you always begin at healthy baseline.

## 2) Show the product loop

1. **Focus tracking starts only when face + eyes are seen.**
2. Keep looking at screen to trigger `WARNING` then `DECAY`.
3. Press `Y` when prompted (demo mode also auto-starts after ~1s).
4. Scan QR from phone.
5. Upload an outdoor photo from the phone page.
6. Watch `NFT UPGRADED` popup on laptop.
7. Open `http://<laptop-ip>:8088/my-flower` on phone to show token/level/value/proof.

## 3) NFT minting modes

### Demo-safe mode (recommended on stage)

- Uses local mint simulation with optional IPFS proof.
- Set in `.env`:
  - `EMERGENCY_MODE=1`
  - `NFT_STORAGE_API_KEY=<optional>`

What you can say:
- "We mint locally for reliability on stage, and assets are IPFS-backed/chain-ready."
- "Each mint also appends to our local immutable chain (`data/simulated_chain.jsonl`) with hash-linked blocks."

### Live chain mode

- Set in `.env`:
  - `ENABLE_CHAIN=1`
  - `EMERGENCY_MODE=0`
  - `THIRDWEB_MINT_URL=...`
  - `THIRDWEB_UPDATE_URL=...`
  - `THIRDWEB_API_KEY=...`

Then run:

```bash
bash run_all.sh
```

## 4) Submission checklist

- Include:
  - `camera_tracking.py`
  - `run_all.sh`
  - `.env.example`
  - `DEMO_WORKFLOW.md`
  - `README.md`
- Mention:
  - state machine flow
  - sessionized QR proof
  - absence gate
  - ledger value growth
  - NFT upgrade popup + phone wallet view
