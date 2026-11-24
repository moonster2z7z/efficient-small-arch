"""Linear recurrences h_t = a_t * h_{t-1} + b_t as single-pass compiled loops with exact gradients.

PyTorch has no native first-order recurrence op, and the log-depth parallel scan is memory-bound on a
CPU. One sequential pass (and one reverse pass for the gradient) is the true cost of the operation.
"""
import numpy as np
import numba
import torch


@numba.njit(cache=True, fastmath=True)
def _fwd(a, b, h):
    B, T, D = b.shape
    for i in range(B):
        for d in range(D):
            h[i, 0, d] = b[i, 0, d]
        for t in range(1, T):
            for d in range(D):
                h[i, t, d] = a[i, t, d] * h[i, t - 1, d] + b[i, t, d]


@numba.njit(cache=True, fastmath=True)
def _bwd(a, h, gh, ga, gb):
    B, T, D = h.shape
    acc = np.zeros(D, dtype=np.float32)
    for i in range(B):
        acc[:] = 0.0
        for t in range(T - 1, -1, -1):
            for d in range(D):
                g = gh[i, t, d] + acc[d]
                gb[i, t, d] = g
                ga[i, t, d] = g * h[i, t - 1, d] if t > 0 else 0.0
                acc[d] = g * a[i, t, d]


@numba.njit(cache=True, fastmath=True)
def _fwd_c(lam, b, h):
    B, T, D = b.shape
    for i in range(B):
        for d in range(D):
            h[i, 0, d] = b[i, 0, d]
        for t in range(1, T):
            for d in range(D):
                h[i, t, d] = lam[d] * h[i, t - 1, d] + b[i, t, d]


@numba.njit(cache=True, fastmath=True)
def _bwd_c(lam, gh, gb):
    B, T, D = gh.shape
    acc = np.zeros(D, dtype=np.float32)
    for i in range(B):
        acc[:] = 0.0
        for t in range(T - 1, -1, -1):
            for d in range(D):
                g = gh[i, t, d] + acc[d]
                gb[i, t, d] = g
                acc[d] = g * lam[d]


class _Gated(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        an, bn = np.ascontiguousarray(a.detach().numpy()), np.ascontiguousarray(b.detach().numpy())
        h = np.empty_like(bn)
        _fwd(an, bn, h)
        ht = torch.from_numpy(h)
        ctx.save_for_backward(a, ht)
        return ht

    @staticmethod
    def backward(ctx, gh):
        a, h = ctx.saved_tensors
        ghn = np.ascontiguousarray(gh.numpy())
        ga, gb = np.empty_like(ghn), np.empty_like(ghn)
        _bwd(np.ascontiguousarray(a.detach().numpy()), h.numpy(), ghn, ga, gb)
        return torch.from_numpy(ga), torch.from_numpy(gb)


class _Const(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lam, b):
        bn = np.ascontiguousarray(b.detach().numpy())
        h = np.empty_like(bn)
        _fwd_c(lam.numpy(), bn, h)
        ctx.lam = lam
        return torch.from_numpy(h)

    @staticmethod
    def backward(ctx, gh):
        ghn = np.ascontiguousarray(gh.numpy())
        gb = np.empty_like(ghn)
        _bwd_c(ctx.lam.numpy(), ghn, gb)
        return None, torch.from_numpy(gb)


def scan_gated(a, b):
    """h_t = a_t h_{t-1} + b_t, h_{-1} = 0.  a, b: (B,T,D) float32."""
    return _Gated.apply(a, b)


def scan_const(b, lam):
    """h_t = lam h_{t-1} + b_t with a fixed per-channel lam (D,). No gradient to lam."""
    return _Const.apply(lam, b)


def warmup():
    a = torch.rand(2, 4, 3, requires_grad=True); b = torch.rand(2, 4, 3, requires_grad=True)
    scan_gated(a, b).sum().backward()
    scan_const(b, torch.rand(3)).sum().backward()


warmup()
