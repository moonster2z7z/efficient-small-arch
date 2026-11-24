"""One budgeted run: train a model on one task for a fixed number of CPU-seconds, then evaluate.

The training budget is process CPU time (all threads) spent inside training steps, including batch
construction. Evaluation is not charged to the training budget but its cost is measured and reported.
The learning-rate schedule is a function of budget progress, so every model uses its whole budget.
"""
import json, math, os, resource, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import models as M

LN2 = math.log(2.0)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def n_params(model):
    seen, n = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p)); n += p.numel()
    return n


# ----------------------------------------------------------------------------- batching
class LMSampler:
    def __init__(self, arr, ctx, bs, rng):
        self.a = torch.from_numpy(arr.astype(np.int64)); self.ctx, self.bs, self.rng = ctx, bs, rng

    def next(self):
        ix = torch.from_numpy(self.rng.integers(0, len(self.a) - self.ctx - 1, size=self.bs))
        idx = ix[:, None] + torch.arange(self.ctx + 1)[None]
        chunk = self.a[idx]
        return chunk[:, :-1], chunk[:, 1:], None


class ClsSampler:
    """Epoch-wise shuffling; variable-length inputs are bucketed by length to limit padding."""

    def __init__(self, split, bs, rng, max_len, image, augment):
        self.x, self.y, self.len = split
        self.bs, self.rng, self.max_len, self.image, self.augment = bs, rng, max_len, image, augment
        self.batches, self.epochs = [], 0

    def _refill(self):
        n = len(self.y)
        perm = self.rng.permutation(n)
        if self.len is not None and int(self.len.min()) != int(self.len.max()):
            chunks = [perm[i:i + self.bs * 50] for i in range(0, n, self.bs * 50)]
            perm = np.concatenate([c[np.argsort(self.len[c], kind='stable')] for c in chunks])
        b = [perm[i:i + self.bs] for i in range(0, n, self.bs)]
        order = self.rng.permutation(len(b))
        self.batches = [b[i] for i in order]
        self.epochs += 1

    def next(self):
        if not self.batches:
            self._refill()
        idx = self.batches.pop()
        return make_batch(self.x, self.y, self.len, idx, self.max_len, self.image, self.augment, self.rng)


MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def make_batch(x, y, ln, idx, max_len, image, augment, rng):
    idx = np.sort(idx)
    yb = torch.from_numpy(y[idx])
    if image:
        xb = torch.from_numpy(x[idx]).permute(0, 3, 1, 2).float().div_(255.0)
        if augment:
            B = xb.shape[0]
            flip = torch.from_numpy(rng.random(B) < 0.5)
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            pad = F.pad(xb, (4, 4, 4, 4), mode='reflect')
            dx, dy = rng.integers(0, 9, size=2)
            xb = pad[:, :, dy:dy + 32, dx:dx + 32]
        return (xb - MEAN) / STD, yb, None
    L = min(int(ln[idx].max()), max_len)
    xb = torch.from_numpy(x[idx, :L].astype(np.int64))
    lb = torch.from_numpy(np.minimum(ln[idx], L).astype(np.int64))
    return xb, yb, lb


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def eval_lm(model, arr, ctx, bs=64):
    """Bits per byte. Windows of length ctx with stride ctx/2; every byte after the first half-window
    is predicted with at least ctx/2 bytes of context."""
    a = torch.from_numpy(arr.astype(np.int64))
    half = ctx // 2
    starts = list(range(0, len(a) - ctx - 1, half))
    tot, cnt = 0.0, 0
    for i in range(0, len(starts), bs):
        st = torch.tensor(starts[i:i + bs])
        idx = st[:, None] + torch.arange(ctx + 1)[None]
        ch = a[idx]
        logits = model(ch[:, :-1])
        lp = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ch[:, 1:].reshape(-1), reduction='none').view(len(st), ctx)
        if i == 0:
            tot += lp[0, :half].sum().item(); cnt += half
        tot += lp[:, half:].sum().item(); cnt += lp[:, half:].numel()
    return tot / cnt / LN2


@torch.no_grad()
def eval_cls(model, split, max_len, image, bs=200):
    x, y, ln = split
    n = len(y)
    order = np.arange(n) if ln is None else np.argsort(ln, kind='stable')
    correct = 0
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        xb, yb, lb = make_batch(x, y, ln, idx, max_len, image, False, None)
        correct += (model(xb, lb).argmax(-1) == yb).sum().item()
    return correct / n


def evaluate(model, task, split, cfg):
    model.eval()
    t0 = time.process_time()
    if task.kind == 'lm':
        v = eval_lm(model, getattr(task, split), cfg['ctx'])
    else:
        v = eval_cls(model, getattr(task, split), cfg.get('max_len', 4096), task.input == 'image')
    model.train()
    return v, time.process_time() - t0


# ----------------------------------------------------------------------------- one run
def run(job):
    """job: dict(task, model (name), cfg, seed, frac, budget). Returns a result dict."""
    torch.set_num_threads(int(job.get('threads', 1)))
    seed = int(job['seed'])
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    cfg = dict(job['cfg'])
    task = D.load(job['task'], job.get('frac', 1.0), seed, test_cap=int(job.get('test_cap', 5000)), val_cap=int(job.get('val_cap', 2000)))
    cfg.setdefault('bi', task.kind != 'lm')
    budget = float(job['budget'])
    rss0 = rss_mb()
    model = M.build(task, cfg)
    npar = n_params(model)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg.get('wd', 0.05)},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=cfg['lr'], betas=(0.9, cfg.get('beta2', 0.98)))
    bs = cfg['bs']
    if task.kind == 'lm':
        sampler = LMSampler(task.train, cfg['ctx'], bs, rng)
    else:
        sampler = ClsSampler(task.train, bs, rng, cfg.get('max_len', 4096), task.input == 'image', cfg.get('augment', True))
    warm, lr0, floor = cfg.get('warmup', 0.03), cfg['lr'], cfg.get('lr_floor', 0.05)
    ls = cfg.get('label_smoothing', 0.0)
    used, step, seen, curve = 0.0, 0, 0, []
    n_evals = int(job.get('n_evals', 4))
    next_eval = budget / n_evals
    ema_loss = None
    model.train()
    while used < budget:
        t0 = time.process_time()
        p = used / budget
        lr = lr0 * (p / warm if p < warm else floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (p - warm) / (1 - warm))))
        for g in opt.param_groups:
            g['lr'] = lr
        xb, yb, lb = sampler.next()
        if task.kind == 'lm':
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            seen += yb.numel()
        else:
            loss = F.cross_entropy(model(xb, lb), yb, label_smoothing=ls)
            seen += len(yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('clip', 1.0))
        opt.step()
        step += 1
        used += time.process_time() - t0
        li = loss.item()
        if not math.isfinite(li):
            return dict(job=job, status='diverged', step=step, used=used, params=npar)
        ema_loss = li if ema_loss is None else 0.98 * ema_loss + 0.02 * li
        if used >= next_eval and used < budget:
            v, _ = evaluate(model, task, 'val', cfg)
            curve.append([round(used, 1), round(v, 5), round(ema_loss, 4)])
            next_eval += budget / n_evals
    val, tv = evaluate(model, task, 'val', cfg)
    test, tt = evaluate(model, task, 'test', cfg)
    curve.append([round(used, 1), round(val, 5), round(ema_loss, 4)])
    n_test = len(task.test) if task.kind == 'lm' else len(task.test[1])
    epochs = seen / (len(task.train) if task.kind == 'lm' else len(task.train[1]))
    return dict(job=job, status='ok', params=npar, steps=step, seen=seen, epochs=round(epochs, 3),
                train_cpu_s=round(used, 1), val=round(val, 5), test=round(test, 5), curve=curve,
                train_loss=round(ema_loss, 4), eval_cpu_s_per_1k=round(tt / n_test * 1000, 4),
                peak_rss_mb=round(rss_mb(), 1), model_rss_mb=round(rss_mb() - rss0, 1))


if __name__ == '__main__':
    job = json.loads(sys.argv[1])
    t = time.time()
    res = run(job)
    res['wall_s'] = round(time.time() - t, 1)
    print('RESULT ' + json.dumps(res))
