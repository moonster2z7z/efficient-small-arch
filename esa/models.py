"""Sequence models built from one shared skeleton so that only the sequence mixer differs.

    input -> frontend (token embedding | image patches) -> N x [mixer + MLP, pre-norm residual] -> head

Mixers:
  attn   multi-head self-attention (Transformer baseline)
  gru    gated diagonal linear recurrence (minGRU-style), computed with a parallel scan
  ema    fixed multi-timescale leaky integrators (reservoir / "time cell" memory): no learned recurrence
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fastscan as FS


# ----------------------------------------------------------------------------- scans
# Pure-PyTorch reference implementations (log-depth). The models use the compiled single-pass
# versions in fastscan.py, which are about 35x faster on one CPU core and give identical results.
def scan_const_ref(u, lam):
    """h_t = lam * h_{t-1} + u_t for a per-channel constant lam in [0,1).  u: (B,T,D), lam: (D,).
    Uses 1/(1-lam z^-1) = prod_k (1 + lam^(2^k) z^-(2^k)): log2(T) shifted adds, exact and stable."""
    T = u.shape[1]
    k = 1
    p = lam
    while k < T:
        u = torch.cat([u[:, :k], u[:, k:] + p * u[:, :-k]], 1)
        p = p * p
        k *= 2
    return u


def scan_gated_ref(a, b):
    """h_t = a_t * h_{t-1} + b_t with h_0 = 0 (Hillis-Steele parallel prefix).  a, b: (B,T,D)."""
    T = a.shape[1]
    k = 1
    while k < T:
        b = torch.cat([b[:, :k], b[:, k:] + a[:, k:] * b[:, :-k]], 1)
        if k * 2 < T:
            a = torch.cat([a[:, :k], a[:, k:] * a[:, :-k]], 1)
        k *= 2
    return b


# ----------------------------------------------------------------------------- mixers
class Attn(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        self.h = cfg.get('heads', 4)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = cfg.get('pos', 'learned') == 'rope'
        self.dh = d // self.h

    def _rot(self, x):  # x: (B,h,T,dh)
        T = x.shape[2]
        half = self.dh // 2
        freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
        ang = torch.arange(T, dtype=torch.float32)[:, None] * freq[None]
        c, s = ang.cos(), ang.sin()
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)

    def forward(self, x, mask, causal):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = self._rot(q), self._rot(k)
        am = None
        if mask is not None:
            am = mask[:, None, None, :]  # True = keep
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am, is_causal=causal and am is None)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class GRUScan(nn.Module):
    """minGRU-style gated linear recurrence; bidirectional when the task is not causal."""

    def __init__(self, d, cfg):
        super().__init__()
        self.m = int(d * cfg.get('expand', 1.0))
        self.bi = cfg.get('bi', True)
        self.f = nn.Linear(d, 2 * self.m)
        self.b = nn.Linear(d, 2 * self.m) if self.bi else None
        self.o = nn.Linear(self.m * (2 if self.bi else 1), d, bias=False)
        nn.init.constant_(self.f.bias[:self.m], -1.0)  # start with slow forgetting

    def _run(self, lin, x, mask):
        z, c = lin(x).chunk(2, -1)
        z = torch.sigmoid(z)
        if mask is not None:
            z = z * mask[..., None]
        return FS.scan_gated(1 - z, z * c)

    def forward(self, x, mask, causal):
        h = self._run(self.f, x, mask)
        if self.bi and not causal:
            hb = self._run(self.b, x.flip(1), None if mask is None else mask.flip(1)).flip(1)
            h = torch.cat([h, hb], -1)
        elif self.bi:
            h = torch.cat([h, torch.zeros_like(h)], -1)
        return self.o(h)


class EMABank(nn.Module):
    """Fixed bank of leaky integrators with log-spaced time constants (no learned recurrent weights).
    Each input channel is projected to m channels; channel j is low-pass filtered with time constant tau_j."""

    def __init__(self, d, cfg):
        super().__init__()
        self.m = int(d * cfg.get('expand', 1.0))
        self.bi = cfg.get('bi', True)
        tmax = cfg.get('tau_max', 256.0)
        tau = torch.exp(torch.linspace(0, math.log(tmax), self.m))
        tau = tau[torch.randperm(self.m, generator=torch.Generator().manual_seed(0))]
        self.register_buffer('lam', 1.0 - 1.0 / tau)  # tau = 1 -> lam = 0 (identity)
        self.i = nn.Linear(d, self.m, bias=False)
        self.o = nn.Linear(self.m * (2 if self.bi else 1), d, bias=False)

    def _run(self, u, mask):
        if mask is not None:
            u = u * mask[..., None]
        return FS.scan_const(u * (1 - self.lam), self.lam)

    def forward(self, x, mask, causal):
        u = self.i(x)
        h = self._run(u, mask)
        if self.bi and not causal:
            hb = self._run(u.flip(1), None if mask is None else mask.flip(1)).flip(1)
            h = torch.cat([h, hb], -1)
        elif self.bi:
            h = torch.cat([h, torch.zeros_like(h)], -1)
        return self.o(h)


MIXERS = {'attn': Attn, 'gru': GRUScan, 'ema': EMABank}


class Block(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.mix = MIXERS[cfg['mixer']](d, cfg)
        r = cfg.get('mlp', 4)
        self.has_mlp = r > 0
        if self.has_mlp:
            self.n2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, int(r * d)), nn.GELU(), nn.Linear(int(r * d), d))
        self.drop = nn.Dropout(cfg.get('dropout', 0.0))

    def forward(self, x, mask, causal):
        x = x + self.drop(self.mix(self.n1(x), mask, causal))
        if self.has_mlp:
            x = x + self.drop(self.mlp(self.n2(x)))
        return x


class Net(nn.Module):
    def __init__(self, task, cfg):
        super().__init__()
        d = cfg['d']
        self.cfg, self.kind, self.input = cfg, task.kind, task.input
        self.max_len = cfg.get('max_len', cfg.get('ctx', 4096))
        if task.input == 'tokens':
            self.emb = nn.Embedding(task.vocab, d)
        else:
            self.p = cfg.get('patch', 4)
            self.emb = nn.Linear(3 * self.p * self.p, d)
            self.max_len = (32 // self.p) ** 2
        self.pos = None
        if cfg.get('pos', 'learned') == 'learned':
            self.pos = nn.Parameter(torch.zeros(self.max_len, d))
            nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, cfg) for _ in range(cfg['layers'])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, task.n_out)
        if task.kind == 'lm' and cfg.get('tie', True):
            self.head.weight = self.emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, lengths=None):
        if self.input == 'image':  # x: (B,3,32,32) float
            p = self.p
            x = F.unfold(x, p, stride=p).transpose(1, 2)
        h = self.emb(x)
        B, T, _ = h.shape
        if self.pos is not None:
            h = h + self.pos[:T]
        mask = None
        if lengths is not None and int(lengths.min()) < T:
            mask = torch.arange(T)[None, :] < lengths[:, None]
        causal = self.kind == 'lm'
        for b in self.blocks:
            h = b(h, mask, causal)
        h = self.norm(h)
        if self.kind == 'lm':
            return self.head(h)
        if mask is None:
            pooled = h.mean(1)
        else:
            m = mask[..., None].to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1)
        return self.head(pooled)


def build(task, cfg):
    return Net(task, cfg)
