"""Run the contract checks of tests/test_contract.py against any config file.

Usage: python check_config.py configs/modern_w256d6.json
tests/ stays unchanged; this only repeats its checks for a non-baseline config.
"""
import json
import sys
import torch
from student import build_model

torch.manual_seed(17)
config = json.loads(open(sys.argv[1]).read())
model = build_model(config).eval()
print('parameters:', sum(p.numel() for p in model.parameters()))
with torch.no_grad():
    x = torch.randint(0, 2048, (2, 256))
    changed = x.clone(); changed[:, 100:] = (changed[:, 100:] + 19) % 2048
    a, b = model.predict_log_probs(x), model.predict_log_probs(changed)
    torch.testing.assert_close(a[:, :100], b[:, :100], atol=1e-5, rtol=1e-5)
    print('ok  future inputs do not change earlier predictions')
    assert a.shape == (2, 256, 2048) and torch.isfinite(a).all()
    torch.testing.assert_close(a.logsumexp(-1), torch.zeros(2, 256), atol=1e-5, rtol=1e-5)
    print('ok  shape, finite, normalized')
    torch.testing.assert_close(model.predict_log_probs(x[:1]), a[:1], atol=1e-5, rtol=1e-5)
    print('ok  examples independent within a batch')
    model.predict_log_probs((x + 31) % 2048)
    torch.testing.assert_close(model.predict_log_probs(x), a, atol=1e-6, rtol=1e-6)
    print('ok  no state carried between calls (and eval mode is deterministic)')
model.train()
loss = torch.nn.functional.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
assert torch.isfinite(loss) and grads and all(torch.isfinite(g).all() for g in grads)
print('ok  training loss %.3f with finite gradients' % loss.item())
