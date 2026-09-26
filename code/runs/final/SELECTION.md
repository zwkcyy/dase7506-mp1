# Final model selection (MP1)

Selection uses the **validation split only**. The test split had not been evaluated for any
candidate when this choice was made.

## Candidates (validation BPB, cache tuned on validation, fine grid λ ∈ {0.03…0.07} × θ ∈ {7, 10, 14})

| candidate | base model | params | cache off | cache on (best λ, θ) | gain |
|---|---|---|---|---|---|
| e7-cache-fine | e7-long3x-drop10 (modern w256 d6, dropout 0.1, 14400 steps, lr 2e-3, EMA 0.999) | 5,344,512 | 1.527374 | **1.504738** (λ 0.05, θ 10) | 0.0226 |
| e8-cache | e8-long3x-d8-drop10 (same recipe, depth 8) | 6,951,168 | 1.550568 | 1.521285 (λ 0.05, θ 10) | 0.0293 |

## Rule

Pick the lower validation BPB; if the two differ by less than 0.003, pick E7 (faster, more
evaluation-time headroom).

## Choice

**e7-cache-fine.** It is 0.0165 BPB better than e8-cache, so the tie-break was not needed.
E8 (depth 8) overfits more over ~33 epochs: its raw-weight validation curve bottoms out at
step 10800 (1.5573) and ends at 1.5651, while E7 ends at 1.5392.

- Source checkpoint: `runs/e7-cache-fine/checkpoint.pt`, copied byte-for-byte to `runs/final/checkpoint.pt`
- Weights from: `runs/e7-long3x-drop10/checkpoint.pt` (EMA weights)
- Cache: `cache_lambda = 0.05`, `cache_theta = 10.0` (stored in the checkpoint config)
- Implementation: `student` (sparse exact within-window cache, commit 2ef7d3d)
- Checkpoint SHA256: `b1a16def50ec0cbc3656f46392c049dde58da3d907bad8dc70152fb73215da0b`
- Checkpoint size: 20.40 MiB
