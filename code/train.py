"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens.

Student additions (all defaults reproduce the original recipe exactly):
  --lr, --min-lr-ratio, --warmup   peak LR, final LR as a fraction of peak, warmup steps
  --weight-decay, --decay-2d-only  AdamW decay; optionally only on matrices (not norms/biases)
  --grad-clip                      gradient-norm clipping threshold
  --ema                            EMA of weights (e.g. 0.999); 0 disables. When enabled, the saved
                                   checkpoint holds the EMA weights and metrics.json also reports
                                   the raw (non-averaged) weights' validation score.
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--weight-decay', type=float, default=.1)
    p.add_argument('--decay-2d-only', action='store_true')
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--ema', type=float, default=0.)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.warmup < 1 or not 0 <= args.ema < 1:
        p.error('Warmup must be positive and --ema must be in [0, 1).')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.decay_2d_only:
        named = list(model.named_parameters())
        groups = [{'params': [q for _, q in named if q.dim() >= 2], 'weight_decay': args.weight_decay},
                  {'params': [q for _, q in named if q.dim() < 2], 'weight_decay': 0.}]
    else:
        groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if args.ema > 0 else None
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.lr * min(1.,(step+1)/args.warmup) * (
            args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
        optimizer.step()
        if ema is not None:
            decay = min(args.ema, (step+1)/(step+10))
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    if value.dtype.is_floating_point:
                        ema[key].lerp_(value, 1-decay)
                    else:
                        ema[key].copy_(value)
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            intermediate = score(model,*data['validation'],device,'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    validation_raw = None
    if ema is not None:
        validation_raw = score(model,*data['validation'],device,'fp32')
        validation_raw.pop('window_nll_nats')
        model.load_state_dict(ema)
    validation = score(model,*data['validation'],device,'fp32')
    validation.pop('window_nll_nats')
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':args.steps*args.batch_size*256},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':args.steps*args.batch_size*256,'preparation_seconds':preparation_seconds,
              'train_seconds':train_seconds,'validation':validation,'validation_raw_weights':validation_raw,
              'recipe':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},
              'history':history,
              'validation_history':validation_history,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
