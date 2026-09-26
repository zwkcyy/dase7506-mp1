"""Check that the sparse mix_cache equals the original dense implementation."""
import math
import torch
from torch.nn import functional as F
from student import build_model


def dense_reference(model, log_probs, hidden, ids):
    batch, length, vocab = log_probs.shape
    h = F.normalize(hidden, dim=-1)
    keys, values = h[:, :-1], ids[:, 1:]
    scores = model.cache_theta * h @ keys.transpose(1, 2)
    allowed = torch.ones(length, length - 1, dtype=torch.bool, device=ids.device).tril(-1)
    weights = torch.softmax(scores.masked_fill(~allowed, float('-inf')), dim=-1).nan_to_num(0.)
    cache = torch.zeros_like(log_probs).scatter_add_(-1, values[:, None, :].expand(batch, length, length - 1), weights)
    mixed = torch.logaddexp(log_probs + math.log(1 - model.cache_lambda), cache.log() + math.log(model.cache_lambda))
    return torch.where(allowed.any(-1)[None, :, None], mixed, log_probs)


torch.manual_seed(0)
config = dict(vocab=2048, width=64, heads=4, depth=2, context=256, norm='rms', mlp='swiglu', pos='rope',
              bias=False, cache_lambda=0.05, cache_theta=10.0)
model = build_model(config).eval()
worst = 0.
with torch.no_grad():
    for vocab_used in (8, 64, 2048):        # few distinct tokens -> many duplicates
        for lam, theta in ((0.05, 10.), (0.3, 0.), (0.02, 80.)):
            model.cache_lambda, model.cache_theta = lam, theta
            ids = torch.randint(0, vocab_used, (4, 256))
            hidden = model.features(ids)
            log_probs = F.log_softmax(model.head(hidden).float(), dim=-1)
            sparse = model.mix_cache(log_probs, hidden.float(), ids)
            dense = dense_reference(model, log_probs, hidden.float(), ids)
            assert torch.isfinite(sparse).all()
            worst = max(worst, (sparse - dense).abs().max().item(),
                        (sparse.logsumexp(-1)).abs().max().item())
print('max |sparse - dense| and |logsumexp|:', worst)
assert worst < 1e-4, 'sparse cache does not match the dense reference'
print('ok  sparse cache matches dense reference')
