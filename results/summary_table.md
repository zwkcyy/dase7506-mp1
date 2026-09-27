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
| e7-cache-fine | e7-long3x-drop10 + cache λ=0.05 θ=10 | 5,344,512 | – | – | – | – | – | 0 (no retraining) | 1.5047 | – | 3.35 |
| e8-cache | e8-long3x-d8-drop10 + cache λ=0.05 θ=10 | 6,951,168 | – | – | – | – | – | 0 (no retraining) | 1.5213 | – | 4.20 (est.) |

Ablations (1200 steps, lr 1e-3, seed 17; difference vs E2 = 1.6878):

| run | change | params | val BPB | Δ vs E2 |
|---|---|---:|---:|---:|
| abl-no_rope | learned pos (no RoPE) | 5,410,048 | 1.8034 | +0.1157 |
| abl-no_swiglu | GELU MLP (no SwiGLU) | 5,246,208 | 1.7482 | +0.0605 |
| abl-no_scaledinit | no scaled init | 5,344,512 | 1.6922 | +0.0045 |
| abl-with_bias | with bias | 5,360,640 | 1.6856 | -0.0021 |
| abl-no_rms | LayerNorm (no RMSNorm) | 5,347,840 | 1.6834 | -0.0044 |
