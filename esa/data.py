"""Fixed benchmark suite: five tasks, each with train / val / test and a 1/10-data condition.

  text        TinyStories byte-level language modelling        metric: bits per byte (lower is better)
  cifar10     CIFAR-10 image classification                    metric: accuracy
  listops     LRA ListOps (10-way, length <= 890)              metric: accuracy
  pathfinder  LRA Pathfinder-32 (pixel sequence, length 1024)  metric: accuracy
  imdb        LRA Text = IMDB byte-level sentiment (<= 4096)   metric: accuracy

Test data is never used for training or model selection; `val` is carved from training data
(CIFAR-10) or is the official validation split.
"""
import os
import numpy as np

DATA = os.environ.get('ESA_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'work', 'data'))
TASKS = ['text', 'cifar10', 'listops', 'pathfinder', 'imdb']
HIGHER_BETTER = {'text': False, 'cifar10': True, 'listops': True, 'pathfinder': True, 'imdb': True}
METRIC = {'text': 'bits/byte', 'cifar10': 'acc', 'listops': 'acc', 'pathfinder': 'acc', 'imdb': 'acc'}
CHANCE = {'text': 8.0, 'cifar10': 0.1, 'listops': 0.1, 'pathfinder': 0.5, 'imdb': 0.5}


class Task:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _subset(n, frac, seed):
    if frac >= 1.0:
        return np.arange(n)
    rng = np.random.default_rng(1000 + seed)
    return np.sort(rng.choice(n, size=max(1, int(round(n * frac))), replace=False))


def load(name, frac=1.0, seed=0, test_cap=5000, val_cap=2000):
    """frac < 1 keeps a random `frac` of the training set (seed-dependent); val/test never change."""
    fixed = np.random.default_rng(1234)
    if name == 'text':
        z = np.load(os.path.join(DATA, 'tinystories', 'tinystories.npz'))
        tr = z['train']
        if frac < 1.0:  # one contiguous slice so that document structure is preserved
            n = int(len(tr) * frac)
            off = int(np.random.default_rng(1000 + seed).integers(0, len(tr) - n))
            tr = tr[off:off + n]
        return Task(name=name, kind='lm', input='tokens', vocab=256, n_out=256,
                    train=tr, val=z['val'][:200_000], test=z['test'])
    if name == 'cifar10':
        z = np.load(os.path.join(DATA, 'cifar10', 'cifar10.npz'))
        x, y = z['train_x'], z['train_y'].astype(np.int64)
        trx, try_, vax, vay = x[:45000], y[:45000], x[45000:], y[45000:]
        idx = _subset(len(trx), frac, seed)
        ti = fixed.permutation(10000)[:test_cap]
        return Task(name=name, kind='cls', input='image', vocab=None, n_out=10,
                    train=(trx[idx], try_[idx], None), val=(vax[:val_cap], vay[:val_cap], None),
                    test=(z['test_x'][ti], z['test_y'].astype(np.int64)[ti], None))
    if name in ('listops', 'pathfinder', 'imdb'):
        z = np.load(os.path.join(DATA, 'lra', name + '.npz'))
        idx = _subset(len(z['train_y']), frac, seed)
        vi = fixed.permutation(len(z['val_y']))[:val_cap]
        ti = fixed.permutation(len(z['test_y']))[:test_cap]
        vocab = int(z['vocab'])
        if name == 'pathfinder':
            vocab = 257
        return Task(name=name, kind='cls', input='tokens', vocab=vocab, n_out=int(z['train_y'].max()) + 1,
                    train=(z['train_x'][idx], z['train_y'][idx], z['train_len'][idx]),
                    val=(z['val_x'][vi], z['val_y'][vi], z['val_len'][vi]),
                    test=(z['test_x'][ti], z['test_y'][ti], z['test_len'][ti]))
    raise KeyError(name)
