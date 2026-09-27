# MP1: a small GPT trained from scratch on WikiText-2 (HKU DASE7506)

Author: Zhu Wenkang (HKU student ID u3684363), GitHub: zwkcyy

Final predictor: a 5.3M-parameter GPT (RoPE, SwiGLU, RMSNorm, no biases, scaled init) trained for
14,400 steps with dropout 0.1 and an EMA of the weights, plus a sparse within-window neural cache
at evaluation time. It uses the supplied evaluator and data unchanged.

## 1. Results

| | value |
|---|---|
| **Test BPB** (protocol `7506-mp1-wt2-v2`, CPU fp32, 4 threads) | **1.519774** |
| Official baseline test BPB | 2.101260 |
| Validation BPB of the final predictor | 1.504738 |
| Checkpoint | `runs/final/checkpoint.pt`, 20.40 MiB |
| Checkpoint SHA256 | `b1a16def50ec0cbc3656f46392c049dde58da3d907bad8dc70152fb73215da0b` |
| Evaluator SHA256 (`evaluate.py`, unmodified) | `128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d` |
| Implementation SHA256 (`student.py`) | `ec9777670bb9d592fabc3ee9378eb1a1c9324f605fc354c4e67a9665ce5f9c45` |
| Code version used for the test evaluation | commit `fb51697` |

**Resource limits, measured on the test split** (5 alternating rounds):

| limit | measured | allowed |
|---|---|---|
| CPU scoring time / baseline (median of per-round ratios) | **3.329** (rounds 3.33, 3.36, 3.32, 3.31, 3.41) | ≤ 5 |
| Peak RAM of the evaluation process tree | **1.836 GiB** | ≤ 4 GiB |
| Inference asset (checkpoint, uncompressed) | **20.40 MiB** | ≤ 64 MiB |

How these were measured (`code/measure.py`):

- The unmodified `evaluate.py` runs as a subprocess on CPU, fp32, 4 threads. The time is the
  `seconds` field that `evaluate.py` itself reports (scoring only, excluding loading).
- With `--compare`, the baseline and candidate checkpoints run **alternately** (baseline,
  candidate, baseline, …). Each round's ratio is candidate time divided by that round's
  baseline time, and the reported value is the median of these ratios. Alternating keeps slow
  drift in machine speed from favouring either model.
- Peak RAM is polled every 20 ms as the **sum of RSS over the whole process tree**. On Windows
  the venv's `python.exe` is a small launcher that starts the real interpreter as a child
  process, so measuring the launcher alone would report about 5 MB.
- Machine: Intel Core i5-9300HF (4 cores / 8 threads), Windows 11, on AC power, power mode
  "Best performance". Baseline scoring time on this machine: 14.9 s (test), 13.3 s (validation).

## 2. Quick evaluation (no retraining)

All commands run from `code/`, with Python 3.12 and PyTorch 2.7.1.

```bash
cd code
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Download the checkpoint from **https://github.com/zwkcyy/dase7506-mp1/releases/download/v1.0/checkpoint.pt** and save it as `code/runs/final/checkpoint.pt`.
Then check its hash:

```bash
sha256sum runs/final/checkpoint.pt
# Windows PowerShell: (Get-FileHash runs\final\checkpoint.pt -Algorithm SHA256).Hash.ToLower()
# expected: b1a16def50ec0cbc3656f46392c049dde58da3d907bad8dc70152fb73215da0b
```

Run the official evaluation:

```bash
python evaluate.py --checkpoint runs/final/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs/final/test.json
```

Expected output (the `seconds` field depends on the machine):

```text
"bpb": 1.5197736388045109,
"targets": 428405,
"utf8_bytes": 1292013,
"checkpoint_sha256": "b1a16def50ec0cbc3656f46392c049dde58da3d907bad8dc70152fb73215da0b",
"implementation_sha256": "ec9777670bb9d592fabc3ee9378eb1a1c9324f605fc354c4e67a9665ce5f9c45",
"evaluator_sha256": "128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d",
```

The checkpoint records `implementation: student` and a config with `cache_lambda = 0.05` and
`cache_theta = 10.0`, so `evaluate.py` rebuilds the cached predictor from `student.py` with no
extra files. To reproduce the resource measurement (this also needs the baseline checkpoint,
`runs/baseline/checkpoint.pt`, from the official baseline command in `code/README.md`):

```bash
python measure.py --checkpoint runs/baseline/checkpoint.pt --compare runs/final/checkpoint.pt --split test --repeats 5
```

## 3. Full reproduction

The final model is trained from scratch in one run (no parent checkpoint). The exact command
below comes from the `recipe` field of `results/e7-long3x-drop10/metrics.json`:

```bash
python train.py --implementation student --config configs/modern_w256d6_drop10.json \
  --device cuda --precision fp32 --threads 4 --seed 17 --steps 14400 --batch-size 32 \
  --lr 0.002 --min-lr-ratio 0.1 --warmup 100 --weight-decay 0.1 --grad-clip 1.0 \
  --ema 0.999 --eval-every 1800 --run-dir runs/e7-long3x-drop10
```

Expected result: validation BPB 1.5274 with EMA weights (the saved checkpoint), 1.5392 with the raw weights;
117,964,800 training targets; 3,751 s on the GPU below. Next, tune the cache on the validation
split (no retraining; the output checkpoint has the same weights with the best λ and θ written into its config):

```bash
python tune_cache.py --checkpoint runs/e7-long3x-drop10/checkpoint.pt \
  --lambdas 0,0.03,0.04,0.05,0.06,0.07 --thetas 7,10,14 --save-best runs/e7-cache-fine/checkpoint.pt
```

Expected: best `cache_lambda = 0.05`, `cache_theta = 10.0`, validation BPB 1.5047 (cache off
1.5274). `runs/final/checkpoint.pt` is a byte-identical copy of `runs/e7-cache-fine/checkpoint.pt`.
`tune_cache.py` defaults to `--device cuda`; add `--device cpu` if there is no GPU.

The original run used an NVIDIA GeForce GTX 1650 (4 GB) with fp32. Use `--precision fp32`: on this
GPU, `auto` picks bf16, which is emulated and slower. Training on CPU works but is roughly an
order of magnitude slower. CUDA kernels are not bit-for-bit deterministic, so a re-run may
differ slightly in the last digits of the BPB. The published checkpoint is the reference.

## 4. Method (summary; details in the report)

- **Configurable model (`code/student.py`).** One GPT class covers the classroom baseline as a
  special case. With the baseline config it is numerically identical to `model.GPT` (max
  difference 0.0). With default flags the new `train.py` reproduces the original recipe
  exactly (same loss, BPB and checkpoint hash after 100 steps).
- **Architecture:** width 256, 6 layers, 4 heads, context 256, tied embeddings, rotary position
  embeddings (RoPE), SwiGLU feed-forward (hidden 704), RMSNorm, no linear biases, and GPT-2
  style 1/√(2·depth) scaled initialisation of residual projections. 5,344,512 parameters.
- **Training:** AdamW (weight decay 0.1), peak lr 2e-3 with 100 warmup steps and cosine decay to
  10%, gradient clipping 1.0, dropout 0.1, and an exponential moving average of the weights
  (decay 0.999) that is saved as the checkpoint. 14,400 steps × 32 × 256 targets (about 33
  passes over the training split).
- **Within-window neural cache** (after Grave et al., 2017), used only in `predict_log_probs`:
  `p = (1−λ)·p_model + λ·p_cache`. `p_cache` is a softmax over θ·cos(h_t, h_i) for earlier
  positions i < t of the same window, placed on the tokens that followed them. The
  implementation is sparse but exact: it only touches tokens already seen in the window.
  `code/check_cache_equiv.py` checks it against a dense reference (max error 8.6e-6). With the
  sparse version, the cache adds about 0.23× baseline time.

## 5. Experiments

Validation only. The eval time ratio is the median CPU scoring-time ratio against the baseline
(`results/*/measure_*.json`), measured for each architecture: runs sharing an architecture
share its measurement. `e8-cache` was not timed; its value is the depth-8 ratio plus the
measured cache overhead. The table is generated by `results/make_figures.py` from
`results/*.json` (see also `results/run_log.csv`).

| run | config | params | steps | training targets | lr | dropout | EMA | train s | val BPB | val BPB raw | eval time ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | baseline | 1,088,256 | 1200 | 9,830,400 | 0.001 | 0 | - | 815 | 2.0711 | - | 1.00 |
| baseline-gpu-fp32 | baseline | 1,088,256 | 1200 | 9,830,400 | 0.001 | 0 | - | 72 | 2.0711 | - | 1.00 |
| e1-scale | scale_w256d6 | 5,328,896 | 1200 | 9,830,400 | 0.001 | 0 | - | 242 | 1.8653 | - | 2.80 |
| e2-modern | modern_w256d6 | 5,344,512 | 1200 | 9,830,400 | 0.001 | 0 | - | 306 | 1.6878 | - | 3.06 |
| e3-modern-lr2e-3 | modern_w256d6 | 5,344,512 | 1200 | 9,830,400 | 0.002 | 0 | - | 306 | 1.6852 | - | 3.06 |
| e4-long-drop10 | modern_w256d6_drop10 | 5,344,512 | 4800 | 39,321,600 | 0.002 | 0.1 | 0.999 | 1250 | 1.5393 | 1.5453 | 3.06 |
| e5-long-drop20 | modern_w256d6_drop20 | 5,344,512 | 4800 | 39,321,600 | 0.002 | 0.2 | 0.999 | 1247 | 1.5816 | 1.5818 | 3.06 |
| e7-long3x-drop10 | modern_w256d6_drop10 | 5,344,512 | 14400 | 117,964,800 | 0.002 | 0.1 | 0.999 | 3751 | 1.5274 | 1.5392 | 3.12 |
| e8-long3x-d8-drop10 | modern_w256d8_drop10 | 6,951,168 | 14400 | 117,964,800 | 0.002 | 0.1 | 0.999 | 4996 | 1.5506 | 1.5651 | 3.97 |
| e9-long3x-drop05 | modern_w256d6_drop05 | 5,344,512 | 14400 | 117,964,800 | 0.002 | 0.05 | 0.999 | 3867 | 1.6043 | 1.6240 | 3.12 |
| baseline-long3x | baseline | 1,088,256 | 14400 | 117,964,800 | 0.001 | 0 | - | 892 | 1.6795 | - | 1.00 |
| **e7-cache-fine (final)** | e7-long3x-drop10 + cache λ=0.05 θ=10 | 5,344,512 | – | – | – | – | – | 0 (no retraining) | **1.5047** | – | 3.35 |
| e8-cache | e8-long3x-d8-drop10 + cache λ=0.05 θ=10 | 6,951,168 | – | – | – | – | – | 0 (no retraining) | 1.5213 | – | 4.20 (est.) |

Component ablations (each removes one component from E2; 1200 steps, lr 1e-3, seed 17; E2 = 1.6878):

| run | change | params | val BPB | Δ vs E2 |
|---|---|---:|---:|---:|
| abl-no_rope | learned pos (no RoPE) | 5,410,048 | 1.8034 | +0.1157 |
| abl-no_swiglu | GELU MLP (no SwiGLU) | 5,246,208 | 1.7482 | +0.0605 |
| abl-no_scaledinit | no scaled init | 5,344,512 | 1.6922 | +0.0045 |
| abl-with_bias | with bias | 5,360,640 | 1.6856 | -0.0021 |
| abl-no_rms | LayerNorm (no RMSNorm) | 5,347,840 | 1.6834 | -0.0044 |

Main findings:
- At the same training budget, the modern architecture beats plain scaling by 0.18 BPB.
- RoPE and SwiGLU account for almost all of that gain. Differences below about 0.005 are
  within single-seed noise.
- Training longer helps most, but only with enough regularisation. With dropout 0.05 (E9) the
  model overfits after about 9,000 steps, and depth 8 (E8) overfits after about 10,800 steps.
- The baseline trained on the same 118M targets reaches only 1.68, so the improvement does not
  come from training length alone.
- EMA adds about 0.012 BPB and the cache about 0.023 BPB.

Selection rule, applied on validation only (`results/final/SELECTION.md`): take the lower
validation BPB of `e7-cache-fine` and `e8-cache`, preferring E7 if they are within 0.003. E7
won by 0.0165.

Figures (`results/figures/`, generated by `python results/make_figures.py` from the repository root):

| file | content |
|---|---|
| `a_validation_curves.png` | validation BPB vs step for E7, E8, E9, baseline-long3x (every 1800 steps) and E4, E5 (every 600) |
| `b_equal_budget.png` | baseline / E1 / E2 / E3 at the same training budget |
| `c_ablations.png` | change in validation BPB when one component is removed from E2 |
| `d_cache_grid_e7.png` | E7 validation BPB over the λ × θ cache grid |
| `e_quality_vs_cost.png` | validation BPB vs evaluation-time ratio |

`make_figures.py` needs `matplotlib`, used **only for plotting**. It is not an evaluation
dependency and is not in `requirements.txt`.

## 6. Training and search cost

Hardware: Intel Core i5-9300HF (4C/8T) with 24 GB RAM, and an NVIDIA GeForce GTX 1650 (4 GB), on
Windows 11. All GPU training used fp32.

| run(s) | device | steps | train seconds |
|---|---|---:|---:|
| baseline (official) | CPU | 1200 | 815 |
| check-orig, check-new (train.py equivalence check) | CPU | 100 each | 52 + 52 |
| smoke | GPU (bf16) | 10 | 1.5 |
| baseline-gpu (bf16), baseline-gpu-fp32 | GPU | 1200 each | 108 + 72 |
| timing-* (5 evaluation-timing probes) | GPU | 10 each | 2–4 each, 14 total |
| e1-scale, e2-modern, e3-modern-lr2e-3 | GPU | 1200 each | 242 + 306 + 306 |
| abl-* (5 ablations) | GPU | 1200 each | 280–306 each, 1,464 total |
| e4-long-drop10, e5-long-drop20 | GPU | 4800 each | 1,250 + 1,247 |
| e7-long3x-drop10 (final weights) | GPU | 14400 | 3,751 |
| e8-long3x-d8-drop10 | GPU | 14400 | 4,996 |
| e9-long3x-drop05 | GPU | 14400 | 3,867 |
| e9 first attempt (killed at step 2300 by the host because the system ran low on memory; discarded) | GPU | 2300 | ≈ 600 |
| baseline-long3x | GPU | 14400 | 892 |

- **26 training processes in total:** 17 full runs, 8 short check/timing runs (10–100 steps) and
  1 aborted run.
- **GPU training time:** 18,516 s (5.14 h) for completed runs, about 5.3 h including the aborted
  run. **CPU training time:** 919 s.
- **No-retraining steps:** 4 cache-tuned checkpoints (`e6-cache`, `e7-cache`, `e7-cache-fine`,
  `e8-cache`) from 5 `tune_cache.py` grid searches, 143 validation evaluations on the GPU in
  total, a few seconds each. `runs/final` is a copy of `e7-cache-fine`.
- The CPU timing measurements (`measure.py`, 3–9 alternating rounds each) took roughly 2 h of
  wall-clock time in total.

## 7. Data and rule compliance

- **Data:** only the supplied WikiText-2 training split was used for training. There are no
  pretrained weights, no external data, and no retrieval store. The cache stores nothing
  between windows (see below).
- **Model selection used validation only:** architecture, learning rate, dropout, depth, the
  cache λ/θ and the final checkpoint were all chosen on the validation split.
- **Test was evaluated once, after freezing** (commit `fb51697`). The frozen predictor was then
  re-run 5 times on test only to time it (`measure.py`), which the README allows for a frozen
  predictor. Earlier test evaluations were only the course's smoke test and the official
  baseline, both as instructed by the starter README.
- **Evaluator unchanged:** `evaluate.py`, `common.py`, `data/`, `tests/`, `model.py` and
  `configs/baseline.json` are byte-identical to the starter release (evaluator SHA256
  `128bcb2dab0be0d427505bddb4671e3ab3a8f78e114be79a689c0f9029af133d`, tokenizer SHA256
  `020d1bc6aa4449c4f352b2e03d0e0fb4f39287f15297705e421b1fa7d817262e`). `.gitattributes`
  disables line-ending conversion so the data files stay byte-identical.
- **No cross-window state:** the cache is built from the current window's hidden states only.
  Position t uses only `ids[:, :t+1]`, and nothing persists between `predict_log_probs` calls.
  This is checked by `tests/` and by `check_config.py configs/check_cache.json` (causality,
  normalisation, batch independence, no carried state).
- **Logging:** every run's seed, processed training targets, parent checkpoint, code commit and
  timings are in `code/run_log.csv` (copy in `results/run_log.csv`).

## 8. Repository layout

| path | content |
|---|---|
| `code/student.py` | configurable GPT + sparse within-window cache (the submitted implementation) |
| `code/train.py` | training recipe (original behaviour by default; adds lr, warmup, weight decay, EMA, etc.) |
| `code/tune_cache.py` | validation grid search of cache λ/θ; writes a checkpoint with the chosen values |
| `code/measure.py` | resource measurement (alternating baseline/candidate, process-tree RSS) |
| `code/check_config.py`, `code/check_cache_equiv.py` | contract checks for any config; sparse vs dense cache check |
| `code/configs/` | all experiment configs |
| `code/run_log.csv` | full experiment log |
| `results/` | metrics, cache grids, measurements, final test JSON, selection note, figures |
| `code/evaluate.py`, `common.py`, `model.py`, `data/`, `tests/` | unchanged starter files |

## 9. References and reused work

- Course-provided MP1 starter code (`evaluate.py`, `common.py`, `model.py`, `train.py`, data and tokenizer); the student code builds on it.
- J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, Y. Liu. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021. arXiv:2104.09864.
- N. Shazeer. *GLU Variants Improve Transformer.* 2020. arXiv:2002.05202.
- B. Zhang, R. Sennrich. *Root Mean Square Layer Normalization.* NeurIPS 2019.
- A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. *Language Models are Unsupervised Multitask Learners.* 2019 (GPT-2; scaled residual initialisation).
- E. Grave, A. Joulin, N. Usunier. *Improving Neural Language Models with a Continuous Cache.* ICLR 2017.
- S. Merity, C. Xiong, J. Bradbury, R. Socher. *Pointer Sentinel Mixture Models.* 2016. arXiv:1609.07843 (WikiText).
- B. T. Polyak, A. B. Juditsky. *Acceleration of Stochastic Approximation by Averaging.* SIAM J. Control Optim., 1992 (weight averaging).

## 10. AI assistance

AI assistance was used substantially in this project, as the course permits. I disclose it here in full.

- **Claude (Anthropic, claude.ai chat)** read the starter package with me and translated and
  explained the guide. It proposed the overall plan (modernise the architecture, train longer
  with regularisation, add a within-window neural cache as the core mechanism) and the
  experiment design: equal-budget comparisons, component ablations, the cache λ/θ ablation,
  the longer-training baseline control, and the validation-only selection rules. It wrote the
  first versions of `student.py` (configurable model and cache, including the sparse exact
  rewrite of the cache), the recipe options and EMA in `train.py`, and `tune_cache.py`,
  `measure.py`, `check_config.py` and `check_cache_equiv.py`. It also helped interpret results
  and draft this README and the report.
- **Claude Code (Anthropic, in VS Code)** set up the environment and ran every training,
  evaluation and measurement command on my machine. It kept `run_log.csv` and the git history,
  fixed the Windows process-tree memory measurement, and wrote `collect_results.py`,
  `make_figures.py` and the first draft of this README.
- **My role:** I made the decisions at each stage, ran the work on my own hardware, and
  reviewed the code and results. I understand and can explain every part of the
  implementation. Every number in this repository and the report comes from the logged runs
  in `results/`. No result was produced or edited by hand. Model selection used the
  validation split only, and the test split was evaluated once, after the method was frozen.

## 11. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.
