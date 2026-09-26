"""MP1 student model: a configurable GPT that contains the classroom baseline as a special case.

Required config keys (same as configs/baseline.json): vocab, width, heads, depth, context.
Optional keys (defaults reproduce the baseline architecture of model.py):
  norm        'layer' (default) | 'rms'       normalisation layers
  mlp         'gelu'  (default) | 'swiglu'    feed-forward block
  pos         'learned' (default) | 'rope'    learned absolute positions or rotary embeddings
  bias        true (default) | false          biases in the linear layers
  mlp_hidden  MLP hidden size; default 4*width for gelu, 8/3*width rounded up to 32 for swiglu
  dropout     residual/embedding dropout, active only in training mode (default 0.0)
  scaled_init false (default) | true          GPT-2 style 1/sqrt(2*depth) init of residual outputs
  rope_base   rotary base frequency (default 10000)
  cache_lambda  weight of the within-window neural cache at evaluation (default 0.0 = off)
  cache_theta   sharpness of the cache's cosine-similarity kernel (default 10.0; 0 = uniform copy)

Within-window neural cache (after Grave et al., 2017, "Improving Neural Language Models
with a Continuous Cache"), used only in predict_log_probs:
  p(w | x_<=t) = (1 - lambda) * p_model(w) + lambda * p_cache(w)
  p_cache(w)   = sum_{i<t, x_{i+1}=w} softmax_i(theta * cos(h_t, h_i))
where h are the final normalised hidden states of the same window. Position t only uses
h_0..h_t and tokens x_1..x_t, i.e. ids[:, :t+1]; position 0 has no history and uses p_model.

Both interfaces of model.GPT are kept: forward() returns logits for training and
predict_log_probs() returns normalised log-probabilities for evaluation. Nothing is
carried between calls, so every evaluation window starts fresh.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

DEFAULTS = dict(norm='layer', mlp='gelu', pos='learned', bias=True, mlp_hidden=None,
                dropout=0.0, scaled_init=False, rope_base=10000.0, cache_lambda=0.0, cache_theta=10.0)


def make_norm(kind, width):
    if kind == 'layer':
        return nn.LayerNorm(width)
    if kind == 'rms':
        return nn.RMSNorm(width, eps=1e-6)
    raise ValueError(f'Unknown norm: {kind}')


class Rotary(nn.Module):
    """Rotary position embedding (Su et al., 2021), rotate-half form. Buffers are not saved."""

    def __init__(self, head_dim, context, base):
        super().__init__()
        if head_dim % 2:
            raise ValueError('RoPE needs an even head dimension.')
        inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angles = torch.outer(torch.arange(context, dtype=torch.float32), inv_freq)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x):  # x: [batch, heads, time, head_dim]
        length = x.shape[-2]
        cos, sin = self.cos[:length].to(x.dtype), self.sin[:length].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width, bias = cfg['width'], cfg['bias']
        self.heads, self.dropout = cfg['heads'], cfg['dropout']
        self.norm1, self.norm2 = make_norm(cfg['norm'], width), make_norm(cfg['norm'], width)
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.proj = nn.Linear(width, width, bias=bias)
        if cfg['mlp'] not in ('gelu', 'swiglu'):
            raise ValueError(f"Unknown mlp: {cfg['mlp']}")
        self.swiglu = cfg['mlp'] == 'swiglu'
        hidden = cfg['mlp_hidden']
        self.up = nn.Linear(width, 2 * hidden if self.swiglu else hidden, bias=bias)
        self.down = nn.Linear(hidden, width, bias=bias)

    def forward(self, x, rotary=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if rotary is not None:
            q, k = rotary(q), rotary(k)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + F.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)), self.dropout, self.training)
        h = self.up(self.norm2(x))
        if self.swiglu:
            gate, value = h.chunk(2, dim=-1)
            h = F.silu(gate) * value
        else:
            h = F.gelu(h)
        return x + F.dropout(self.down(h), self.dropout, self.training)


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        unknown = set(config) - set(DEFAULTS) - {'vocab', 'width', 'heads', 'depth', 'context'}
        if unknown:
            raise ValueError(f'Unknown config keys: {sorted(unknown)}')
        cfg = {**DEFAULTS, **config}
        if cfg['width'] % cfg['heads']:
            raise ValueError('width must be divisible by heads.')
        if cfg['pos'] not in ('learned', 'rope'):
            raise ValueError(f"Unknown pos: {cfg['pos']}")
        if cfg['mlp_hidden'] is None:
            cfg['mlp_hidden'] = 4 * cfg['width'] if cfg['mlp'] == 'gelu' else 32 * math.ceil(8 * cfg['width'] / 3 / 32)
        self.config = dict(config)
        self.context = cfg['context']
        self.dropout = cfg['dropout']
        self.cache_lambda, self.cache_theta = float(cfg['cache_lambda']), float(cfg['cache_theta'])
        if not 0 <= self.cache_lambda < 1:
            raise ValueError('cache_lambda must be in [0, 1).')
        width = cfg['width']
        self.token = nn.Embedding(cfg['vocab'], width)
        self.pos = nn.Embedding(self.context, width) if cfg['pos'] == 'learned' else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg['depth'])])
        self.norm = make_norm(cfg['norm'], width)
        self.head = nn.Linear(width, cfg['vocab'], bias=False)
        self.rotary = Rotary(width // cfg['heads'], self.context, cfg['rope_base']) if cfg['pos'] == 'rope' else None
        self.apply(self.initialize)
        if cfg['scaled_init']:
            for block in self.blocks:
                for layer in (block.proj, block.down):
                    nn.init.normal_(layer.weight, std=.02 / math.sqrt(2 * cfg['depth']))
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, self.dropout, self.training)
        for block in self.blocks:
            x = block(x, self.rotary)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities; position t sees ids[:, :t+1] only."""
        hidden = self.features(ids)
        log_probs = F.log_softmax(self.head(hidden).float(), dim=-1)
        if self.cache_lambda == 0 or ids.shape[1] < 2:
            return log_probs
        return self.mix_cache(log_probs, hidden.float(), ids)

    def mix_cache(self, log_probs, hidden, ids):
        # Sparse but exact: p_cache is non-zero only on tokens seen earlier in the window, so every
        # other vocabulary entry is just (1 - lambda) * p_model, a constant shift in log space.
        batch, length, _ = log_probs.shape
        h = F.normalize(hidden, dim=-1)
        keys, values = h[:, :-1], ids[:, 1:]                   # key i predicted token ids[:, i+1]
        queries = h[:, 1:]                                      # row r is position t = r + 1
        scores = self.cache_theta * queries @ keys.transpose(1, 2)       # [batch, length-1, length-1]
        allowed = torch.ones(length - 1, length - 1, dtype=torch.bool, device=ids.device).tril()  # i <= r, i.e. i < t
        weights = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1)
        same = (values[:, :, None] == values[:, None, :]).to(weights.dtype)  # same[j, i]: key j and key i share a token
        cache_at_key = weights @ same                           # p_cache(values[i]) at each row
        index = values[:, None, :].expand(batch, length - 1, length - 1)
        mixed_at_key = torch.logaddexp(log_probs[:, 1:].gather(-1, index) + math.log(1 - self.cache_lambda),
                                       cache_at_key.log() + math.log(self.cache_lambda))
        shift = torch.full((1, length, 1), math.log(1 - self.cache_lambda), dtype=log_probs.dtype, device=ids.device)
        shift[:, 0] = 0.                                        # position 0 has no history: pure p_model
        out = log_probs + shift
        out[:, 1:].scatter_(-1, index, mixed_at_key)            # duplicate tokens write identical values
        return out


def build_model(config):
    return StudentGPT(config)
