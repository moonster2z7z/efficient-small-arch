"""Count-based byte language model: hashed n-gram contexts mixed by an online logistic mixer.

No backpropagation and no gradient through time. Learning is one streaming pass: every table cell is
a probability nudged toward the observed bit, and the mixer is a single linear unit per bit trained by
a local delta rule (the PAQ / gated-linear-network family of compressors). Evaluation is frozen: no
table or mixer is updated on validation or test data.

Model size is counted as table bytes / 4 (float32-equivalent parameters).
"""
import math, time
import numpy as np
import numba

ORDERS_DEFAULT = (1, 2, 3, 4, 5, 6, 8)


@numba.njit(cache=True)
def _squash(x):
    if x > 20.0: x = 20.0
    if x < -20.0: x = -20.0
    return 1.0 / (1.0 + math.exp(-x))


@numba.njit(cache=True)
def _process(data, orders, tables, mask, w, lr_w, limit, learn):
    """Returns total code length in bits. tables: (n_orders, size) uint16, 12-bit prob | 4-bit count."""
    no = len(orders)
    st = np.zeros(no + 1, dtype=np.float64)
    idx = np.zeros(no, dtype=np.int64)
    hs = np.zeros(no, dtype=np.uint64)
    total = 0.0
    n = len(data)
    for pos in range(n):
        for i in range(no):
            h = np.uint64(orders[i] * 0x9E3779B1 + 12345)
            o = orders[i]
            for j in range(1, o + 1):
                c = np.uint64(data[pos - j]) if pos - j >= 0 else np.uint64(0)
                h = (h ^ c) * np.uint64(0x100000001B3)
                h ^= h >> np.uint64(29)
            hs[i] = h
        byte = data[pos]
        c0 = 1
        for b in range(8):
            bit = (byte >> (7 - b)) & 1
            dot = 0.0
            for i in range(no):
                ii = np.int64(((hs[i] + np.uint64(c0) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(20)) & np.uint64(mask))
                idx[i] = ii
                p = ((tables[i, ii] >> 4) + 0.5) / 4096.0
                s = math.log(p / (1.0 - p))
                st[i] = s
                dot += w[c0, i] * s
            st[no] = 0.3
            dot += w[c0, no] * 0.3
            p = _squash(dot)
            if p < 1e-6: p = 1e-6
            if p > 1 - 1e-6: p = 1 - 1e-6
            total -= math.log2(p) if bit == 1 else math.log2(1.0 - p)
            if learn:
                err = bit - p
                for i in range(no + 1):
                    w[c0, i] += lr_w * err * st[i]
                for i in range(no):
                    v = tables[i, idx[i]]
                    pr = np.int64(v >> 4)
                    cnt = np.int64(v & 15)
                    target = 4095 if bit == 1 else 0
                    pr = pr + np.int64((target - pr) / (cnt + 1.5))
                    if cnt < limit: cnt += 1
                    tables[i, idx[i]] = np.uint16((pr << 4) | cnt)
            c0 = (c0 << 1) | bit
    return total


class CM:
    def __init__(self, cfg):
        self.orders = np.array(cfg.get('orders', ORDERS_DEFAULT), dtype=np.int64)
        self.bits = int(cfg.get('table_bits', 18))
        self.tables = np.full((len(self.orders), 1 << self.bits), np.uint16(2048 << 4), dtype=np.uint16)
        self.w = np.full((256, len(self.orders) + 1), 0.3, dtype=np.float64)
        self.lr_w = float(cfg.get('lr_w', 0.002))
        self.limit = int(cfg.get('limit', 15))

    def params(self):  # float32-equivalent
        return int(self.tables.nbytes // 4 + self.w.size)

    def fit(self, data):
        return _process(data, self.orders, self.tables, (1 << self.bits) - 1, self.w, self.lr_w, self.limit, True) / len(data)

    def bpb(self, data):
        return _process(data, self.orders, self.tables, (1 << self.bits) - 1, self.w, self.lr_w, self.limit, False) / len(data)


def _warm():
    m = CM(dict(table_bits=8))
    d = np.frombuffer(b'hello hello hello', dtype=np.uint8)
    m.fit(d); m.bpb(d)


_warm()


def run(job, task, rss_mb):
    """Budgeted run for the text task. Streams over the training bytes in chunks until the data is
    exhausted (`passes` times) or the CPU budget is spent."""
    cfg = job['cfg']
    budget = float(job['budget'])
    m = CM(cfg)
    tr = task.train
    chunk = 250_000
    used, seen, curve = 0.0, 0, []
    passes = int(cfg.get('passes', 1))
    n_evals = int(job.get('n_evals', 4)); next_eval = budget / n_evals
    online = 0.0
    done = False
    for ps in range(passes):
        for i in range(0, len(tr), chunk):
            t0 = time.process_time()
            seg = tr[i:i + chunk]
            online = m.fit(seg)
            seen += len(seg)
            used += time.process_time() - t0
            if used >= next_eval and used < budget:
                curve.append([round(used, 1), round(m.bpb(task.val[:50_000]), 5), round(online, 4)])
                next_eval += budget / n_evals
            if used >= budget:
                done = True; break
        if done: break
    t0 = time.process_time(); val = m.bpb(task.val); test = m.bpb(task.test); tt = time.process_time() - t0
    curve.append([round(used, 1), round(val, 5), round(online, 4)])
    return dict(job=job, status='ok', params=m.params(), steps=seen // chunk, seen=seen, epochs=round(seen / len(tr), 3),
                train_cpu_s=round(used, 1), val=round(val, 5), test=round(test, 5), curve=curve, train_loss=round(online, 4),
                eval_cpu_s_per_1k=round(tt / (len(task.val) + len(task.test)) * 1000, 4), peak_rss_mb=round(rss_mb(), 1))
