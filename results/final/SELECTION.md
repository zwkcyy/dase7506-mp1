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

## Resource check before test (validation, 9 rounds, `measure.py --compare` vs `runs/baseline`)

time_ratio_median 3.329 (min 3.293, IQR 3.315–3.361), baseline spread 7.8%, peak RAM 1.837 GiB,
checkpoint 20.40 MiB. All within limits.

## Frozen test evaluation (run once)

Code frozen at commit `fb51697b14f78cc86fc086eb40922a6a2e279fa6`. Official command:
`python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs/final/test.json`

- **Test BPB: 1.519774** (token ppl 23.97, 428,405 targets, 1,292,013 UTF-8 bytes; the official baseline is 2.101260)
- checkpoint_sha256: `b1a16def50ec0cbc3656f46392c049dde58da3d907bad8dc70152fb73215da0b`
- evaluator_sha256: `128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d` (unmodified evaluate.py)
- implementation_sha256 (student.py): `ec9777670bb9d592fabc3ee9378eb1a1c9324f605fc354c4e67a9665ce5f9c45`

Test-split measurement (`measure.py --compare ... --split test --repeats 5`, CPU fp32, 4 threads):

| | rounds (s) | median (s) |
|---|---|---|
| baseline | 14.9 14.8 14.9 15.0 14.9 | 14.9 |
| final | 49.7 49.8 49.4 49.6 50.9 | 49.7 |

- **time_ratio_median 3.329** (per-round 3.33 / 3.36 / 3.32 / 3.31 / 3.41), limit 5
- **peak RAM 1.836 GiB** (whole evaluate.py process tree), limit 4 GiB
- **checkpoint 20.40 MiB**, limit 64 MiB

No further test evaluations are to be run on any model.
