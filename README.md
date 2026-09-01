# Releaf — touch grass, verified by camera

Your plant wilts while you stare at your screen. To bring it back you have to go outside and photograph real grass.

And it checks. That's the whole project.

Hackathon build. Python, OpenCV, Flask, a Swift desktop overlay, ~2,600 lines.

---

## The actual problem

Naive version is trivially cheatable. Hold up a green jumper. Point the webcam at a photo of a lawn. Congratulations, you've touched grass.

So the real problem isn't "detect grass" — it's **catch people cheating, with no training budget, no internet, running live on a laptop webcam.**

**Beating the green jumper.** Start with an HSV green mask, which happily accepts a t-shirt. Two more signals do the work: real foliage is a hundred shades of green where dyed fabric is basically one, and grass is full of fine edges where a jumper is smooth. So — hue variance, plus Canny edge density counted *only inside the green mask*.

```python
green_mask = (h >= 28) & (h <= 92) & (s >= 45) & (v >= 35)
hue_variation_score = clip(hue_std / 18.0, 0, 1)   # fabric = low hue spread
texture_score = clip(edge_density / 0.18, 0, 1)    # plants = fine edges
```

A separate outdoor heuristic looks at the whole scene, so your houseplant doesn't count either.

**The bit that actually stops cheating** isn't visual. Recovery needs the webcam to confirm you're *not there* — you can't submit proof from outdoors while sitting at your screen. The photo comes off your phone via a QR-scoped session, so the two devices are necessarily in different places.

**k-NN with a fallback.** No trained model — k-NN over a handful of labelled images, blended with the heuristic. `ml is None` when there are no samples, so it works on first run with zero setup and improves as you add them:

```python
if ml is None:      # no samples yet
    return heur     # fall back to pure heuristic
return clip(0.45 * heur + 0.55 * ml, 0.0, 1.0)
```

Right call for a hackathon. Genuinely weaker than a real CNN, but it degrades predictably instead of mysteriously.

## Lifecycle

```
ONLINE → WARNING → DECAY → RECOVERY_REQUIRED → OUTSIDE_MODE → PROOF_VERIFIED → MINTED
```

Every transition gated on the signals above, all of it written to an append-only ledger.

## Chain stuff that can't break the demo

Verified recoveries mint as NFTs via thirdweb on Base Sepolia. A live demo can't hang on an RPC call, so chain state is a *badge*, not a blocking step: `CHAIN_LIVE → CHAIN_RETRYING → CHAIN_FALLBACK_LOCAL_LEDGER`. Chain slow or dead? Retry, quietly fall back to the local hash-linked ledger, carry on.

## Running it

```bash
bash run_demo.sh    # venv, deps, demo mode — then scan the QR in the window
```

Chain mode is opt-in via env vars (see `.env.example`). Without them everything runs local. `run_all.sh` adds the Swift overlay.

## Known rough edges

`camera_tracking.py` is one 2,600-line file — CV pipeline, HTTP server and chain bridge all in one module because splitting them mid-demo wasn't worth the risk. The classes are already there, they just need to become files.

The bigger one: **I never adversarially tested the thing whose entire premise is resisting cheating.** No numbers on how often a jumper or a monitor-photo gets through. That's the gap that matters and it's what I'd build next.

MIT
