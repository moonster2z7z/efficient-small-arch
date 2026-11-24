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


# ----------------------------------------------------------------------------- delta-rule fast weights
# State S (dv x dk) is an associative memory rewritten at every step by an error-correcting (Widrow-Hoff)
# update:  S_t = a_t * S_{t-1} + (v_t - b_t * S_{t-1} k_t) k_t^T ,   out_t = S_t q_t
# a_t in (0,1] is a decay, b_t in [0,1] the in-context learning rate, k_t a unit-norm key.
@numba.njit(cache=True, fastmath=True)
def _delta_fwd(k, q, v, al, be, out, hist):
    N, T, dk = k.shape
    dv = v.shape[2]
    S = np.zeros((dv, dk), dtype=np.float32)
    for n in range(N):
        S[:] = 0.0
        for t in range(T):
            a = al[n, t]; b = be[n, t]
            for i in range(dv):
                sk = 0.0
                for j in range(dk):
                    sk += S[i, j] * k[n, t, j]
                e = v[n, t, i] - b * sk
                o = 0.0
                for j in range(dk):
                    s = a * S[i, j] + e * k[n, t, j]
                    S[i, j] = s
                    hist[n, t, i, j] = s
                    o += s * q[n, t, j]
                out[n, t, i] = o


@numba.njit(cache=True, fastmath=True)
def _delta_bwd(k, q, v, al, be, hist, go, gk, gq, gv, gal, gbe):
    N, T, dk = k.shape
    dv = v.shape[2]
    G = np.zeros((dv, dk), dtype=np.float32)
    zero = np.zeros((dv, dk), dtype=np.float32)
    for n in range(N):
        G[:] = 0.0
        for t in range(T - 1, -1, -1):
            a = al[n, t]; b = be[n, t]
            St = hist[n, t]
            Sp = hist[n, t - 1] if t > 0 else zero
            for j in range(dk):
                gk[n, t, j] = 0.0
                acc = 0.0
                for i in range(dv):
                    acc += St[i, j] * go[n, t, i]
                gq[n, t, j] = acc
            ga = 0.0; gb = 0.0
            for i in range(dv):
                goi = go[n, t, i]
                sk = 0.0; ge = 0.0
                for j in range(dk):
                    G[i, j] += goi * q[n, t, j]
                    sk += Sp[i, j] * k[n, t, j]
                    ge += G[i, j] * k[n, t, j]
                e = v[n, t, i] - b * sk
                gv[n, t, i] = ge
                gb -= ge * sk
                gsk = -b * ge
                for j in range(dk):
                    gk[n, t, j] += G[i, j] * e + Sp[i, j] * gsk
                    ga += G[i, j] * Sp[i, j]
                    G[i, j] = a * G[i, j] + gsk * k[n, t, j]
            gal[n, t] = ga; gbe[n, t] = gb


class _Delta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, q, v, al, be):
        c = lambda x: np.ascontiguousarray(x.detach().numpy())
        kn, qn, vn, an, bn = c(k), c(q), c(v), c(al), c(be)
        out = np.empty_like(vn)
        hist = np.empty(vn.shape + (kn.shape[2],), dtype=np.float32)
        _delta_fwd(kn, qn, vn, an, bn, out, hist)
        ctx.arrs = (kn, qn, vn, an, bn, hist)
        return torch.from_numpy(out)

    @staticmethod
    def backward(ctx, go):
        kn, qn, vn, an, bn, hist = ctx.arrs
        gon = np.ascontiguousarray(go.numpy())
        gk, gq, gv = np.empty_like(kn), np.empty_like(qn), np.empty_like(vn)
        gal, gbe = np.empty_like(an), np.empty_like(bn)
        _delta_bwd(kn, qn, vn, an, bn, hist, gon, gk, gq, gv, gal, gbe)
        f = torch.from_numpy
        return f(gk), f(gq), f(gv), f(gal), f(gbe)


def delta_memory(k, q, v, al, be):
    """k, q: (N,T,dk); v: (N,T,dv); al, be: (N,T).  Returns out (N,T,dv)."""
    return _Delta.apply(k, q, v, al, be)


def _warm_delta():
    g = lambda *s: torch.rand(*s, requires_grad=True)
    delta_memory(g(2, 3, 4), g(2, 3, 4), g(2, 3, 5), g(2, 3), g(2, 3)).sum().backward()


_warm_delta()
