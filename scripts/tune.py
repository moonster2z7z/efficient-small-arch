"""Equal-budget hyperparameter search. Every model gets the same trial list: trial i uses the same
(d, layers, lr, batch size, ...) draw for every mixer, plus that mixer's own options.

  python scripts/tune.py --models attn,gru --tasks text,cifar10 --budget 300 --trials 8
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from esa.sched import run_jobs, job_key
from esa import data as D

PARAM_CAP = 1_000_000


def shared_space(task, i):
    r = np.random.default_rng(TASK_ID[task] * 1000 + i)
    c = dict(d=int(r.choice([64, 96, 128, 192])), layers=int(r.choice([2, 3, 4, 6])),
             lr=float(np.round(np.exp(r.uniform(np.log(5e-4), np.log(4e-3))), 5)), mlp=int(r.choice([2, 4])))
    if task == 'text':
        c.update(ctx=int(r.choice([128, 256])), bs=int(r.choice([16, 32, 64])))
    elif task == 'cifar10':
        c.update(patch=int(r.choice([4, 8])), bs=int(r.choice([64, 128])))
    elif task == 'imdb':
        c.update(max_len=int(r.choice([512, 1024, 2048])), bs=int(r.choice([16, 32])))
    elif task == 'listops':
        c.update(max_len=900, bs=int(r.choice([16, 32])))
    else:
        c.update(max_len=1024, bs=int(r.choice([16, 32])))
    return c, r


TASK_ID = {t: k for k, t in enumerate(D.TASKS)}

MODEL_SPACE = {
    'attn': lambda r: dict(mixer='attn', pos=str(r.choice(['learned', 'rope'])), heads=4),
    'gru': lambda r: dict(mixer='gru', pos=str(r.choice(['learned', 'none'])), expand=float(r.choice([1.0, 2.0]))),
    'ema': lambda r: dict(mixer='ema', pos=str(r.choice(['learned', 'none'])), expand=float(r.choice([1.0, 2.0, 4.0])),
                          tau_max=float(r.choice([64.0, 256.0, 1024.0]))),
}


def make_cfg(task, model, i):
    c, r = shared_space(task, i)
    rm = np.random.default_rng(TASK_ID[task] * 1000 + i + 7919 * (sorted(MODEL_SPACE).index(model) + 1))
    c.update(MODEL_SPACE[model](rm))
    while count_params(task, c) > PARAM_CAP:  # enforce the shared parameter cap
        if c['layers'] > 2: c['layers'] -= 1
        else: c['d'] = int(c['d'] * 0.75)
    return c


def count_params(task, cfg):
    from esa import models as M
    from esa.train import n_params
    stub = dict(text=D.Task(kind='lm', input='tokens', vocab=256, n_out=256),
                cifar10=D.Task(kind='cls', input='image', vocab=None, n_out=10),
                listops=D.Task(kind='cls', input='tokens', vocab=17, n_out=10),
                pathfinder=D.Task(kind='cls', input='tokens', vocab=257, n_out=2),
                imdb=D.Task(kind='cls', input='tokens', vocab=257, n_out=2))[task]
    c = dict(cfg); c.setdefault('bi', stub.kind != 'lm')
    return n_params(M.build(stub, c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='attn')
    ap.add_argument('--tasks', default=','.join(D.TASKS))
    ap.add_argument('--budget', type=float, default=300)
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--first', type=int, default=0)
    ap.add_argument('--log', default=os.path.join(ROOT, 'logs', 'tune.jsonl'))
    a = ap.parse_args()
    jobs = []
    for i in range(a.first, a.first + a.trials):
        for task in a.tasks.split(','):
            for model in a.models.split(','):
                jobs.append(dict(task=task, model=model, cfg=make_cfg(task, model, i), seed=0, frac=1.0,
                                 budget=a.budget, n_evals=2, val_cap=1000, test_cap=1000, trial=i))
    res = run_jobs(jobs, a.log)
    summarize(a.log, a.budget)


def summarize(log, budget=None):
    rows = [json.loads(l) for l in open(log)]
    best = {}
    for r in rows:
        j = r['job']
        if r.get('status') != 'ok' or (budget and j['budget'] != budget):
            continue
        k = (j['task'], j['model'], j['budget'])
        hb = D.HIGHER_BETTER[j['task']]
        if k not in best or (r['val'] > best[k]['val']) == hb:
            best[k] = r
    for k in sorted(best):
        r = best[k]
        print(f"BEST {k[0]:10s} {k[1]:6s} B={k[2]:.0f} val={r['val']:.4f} test={r['test']:.4f} params={r['params']} ep={r['epochs']} cfg={json.dumps(r['job']['cfg'])}")


if __name__ == '__main__':
    main()
