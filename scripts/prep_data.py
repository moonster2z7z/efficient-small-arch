"""Download and convert every benchmark dataset into compact arrays under $ESA_DATA (default ../work/data).

Sources (all public):
  TinyStories V2 (GPT-4) text  : huggingface.co/datasets/roneneldan/TinyStories
  CIFAR-10 (PNG mirror)        : s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz
  LRA ListOps / IMDB / Pathfinder (already tokenised, original LRA splits):
                                 huggingface.co/datasets/monteirot/lra-benchmarks
The LRA .pt files are pickles; they are loaded with torch's weights_only loader and three inert
placeholder classes, so no code from the files is executed.
"""
import os, sys, subprocess, tarfile, numpy as np

DATA = os.environ.get('ESA_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'work', 'data'))
HF = 'https://huggingface.co/datasets'

def fetch(url, path, byte_range=None):
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ['curl', '-sS', '-L', '--fail', '-o', path + '.part']
    if byte_range: cmd += ['-r', byte_range]
    subprocess.run(cmd + [url], check=True)
    os.replace(path + '.part', path)

def tinystories():
    d = os.path.join(DATA, 'tinystories')
    fetch(f'{HF}/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt', f'{d}/valid.txt')
    fetch(f'{HF}/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt', f'{d}/train_200MB.txt', '0-209715199')
    out = f'{d}/tinystories.npz'
    if os.path.exists(out): return
    tr = np.fromfile(f'{d}/train_200MB.txt', dtype=np.uint8)
    va = np.fromfile(f'{d}/valid.txt', dtype=np.uint8)
    # train pool: first 20 MB of the train file. val/test: disjoint slices of the official validation file.
    np.savez(out, train=tr[:20_000_000], val=va[:500_000], test=va[2_000_000:3_000_000])

def cifar10():
    d = os.path.join(DATA, 'cifar10'); out = f'{d}/cifar10.npz'
    if os.path.exists(out): return
    from PIL import Image
    fetch('https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz', f'{d}/cifar10.tgz')
    with tarfile.open(f'{d}/cifar10.tgz') as t: t.extractall(d, filter='data')
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    res = {}
    for split in ['train', 'test']:
        X, Y = [], []
        for ci, c in enumerate(classes):
            dd = os.path.join(d, 'cifar10', split, c)
            for f in sorted(os.listdir(dd)):
                X.append(np.asarray(Image.open(os.path.join(dd, f)).convert('RGB'), dtype=np.uint8)); Y.append(ci)
        X = np.stack(X); Y = np.array(Y, dtype=np.uint8)
        p = np.random.default_rng(0).permutation(len(X))
        res[split + '_x'] = X[p]; res[split + '_y'] = Y[p]
    np.savez(out, **res)

class ListOpsDataset: pass
class IMDbDataset: pass
class PathfinderDataset: pass

def lra():
    import torch
    d = os.path.join(DATA, 'lra_dl'); o = os.path.join(DATA, 'lra'); os.makedirs(o, exist_ok=True)
    torch.serialization.add_safe_globals([ListOpsDataset, IMDbDataset, PathfinderDataset])
    for name, f in [('listops', 'listops.pt'), ('imdb', 'imdb_lra.pt'), ('pathfinder', 'pathfinder_lra.pt')]:
        out = f'{o}/{name}.npz'
        if os.path.exists(out): continue
        fetch(f'{HF}/monteirot/lra-benchmarks/resolve/main/{f}', f'{d}/{f}')
        obj = torch.load(f'{d}/{f}', weights_only=True)
        res = {}
        for split in ['train', 'val', 'test']:
            s = obj[split].__dict__
            if 'tokens' in s:
                x = s['tokens'].numpy(); y = s['labels'].numpy(); ln = np.full(len(x), x.shape[1], dtype=np.int32)
                assert x.min() >= 0 and x.max() <= 256
                x = x.astype(np.uint16)
            else:
                seqs = [np.asarray(t[0][:4096], dtype=np.uint16) for t in s['data']]; y = np.array([int(t[1]) for t in s['data']])
                ln = np.array([len(q) for q in seqs], dtype=np.int32)
                x = np.zeros((len(seqs), int(ln.max())), dtype=np.uint16)
                for i, q in enumerate(seqs): x[i, :len(q)] = q
            res[split + '_x'] = x; res[split + '_y'] = y.astype(np.int64); res[split + '_len'] = ln
            print(name, split, x.shape, 'len', ln.min(), ln.mean().round(1), ln.max(), 'tok', x.min(), x.max(), 'labels', np.bincount(y))
        res['vocab'] = np.array(obj['vocab_size'])
        np.savez(out, **res)

if __name__ == '__main__':
    for fn in (tinystories, cifar10, lra):
        fn(); print('ok', fn.__name__)
