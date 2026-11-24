"""One-shot verdict: candidate vs. Transformer baseline on the fixed five-task suite.

    python bench.py                 # full verdict (3 seeds, full data and 1/10 data, half-budget check)
    python bench.py --budget 60     # quick smoke run

Configurations are frozen in configs/final.json (chosen on validation data only, see EXPERIMENTS.md).
Every run is logged to logs/bench_B<budget>.jsonl; finished runs are reused, so the command can be resumed.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from esa.sched import run_jobs, job_key
from esa import data as D


def stats(rs, key='test'):
    v = np.array([r[key] for r in rs if r.get('status') == 'ok'], dtype=float)
    if len(v) == 0:
        return float('nan'), float('nan'), 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)


def compare(task, cand, base):
    """+1 candidate wins, -1 loses, 0 tie (difference within 2 combined standard errors)."""
    mc, sc, _ = cand; mb, sb, _ = base
    diff = (mc - mb) if D.HIGHER_BETTER[task] else (mb - mc)
    tol = 2.0 * np.sqrt(sc ** 2 + sb ** 2)
    return (1 if diff > tol else -1 if diff < -tol else 0), diff, tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(ROOT, 'configs', 'final.json'))
    ap.add_argument('--budget', type=float, default=None)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--tasks', default=','.join(D.TASKS))
    ap.add_argument('--no-half', action='store_true')
    a = ap.parse_args()
    conf = json.load(open(a.config))
    budget = a.budget or conf['budget']
    seeds = [int(s) for s in a.seeds.split(',')]
    tasks = a.tasks.split(',')
    log = os.path.join(ROOT, 'logs', f'bench_B{int(budget)}.jsonl')

    def job(task, who, seed, frac, b):
        m = conf['tasks'][task][who]
        return dict(task=task, model=m['model'], cfg=m['cfg'], seed=seed, frac=frac, budget=b, n_evals=4)

    jobs, index = [], {}
    for seed in seeds:
        for task in tasks:
            for frac in (1.0, 0.1):
                for who in ('baseline', 'candidate'):
                    j = job(task, who, seed, frac, budget); jobs.append(j); index.setdefault((task, who, frac, 'full'), []).append(j)
            if not a.no_half:
                j = job(task, 'candidate', seed, 1.0, budget / 2); jobs.append(j); index.setdefault((task, 'candidate', 1.0, 'half'), []).append(j)
    res = run_jobs(jobs, log)
    get = lambda *k: [res[job_key(j)] for j in index[k]]

    print(f"\n=== bench: budget {budget:.0f} CPU-s per run, seeds {seeds}, param cap {conf.get('param_cap')} ===")
    hdr = f"{'task':10s} {'metric':9s} | {'baseline':>17s} | {'candidate':>17s} | {'verdict':7s} | {'base 1/10':>9s} {'cand 1/10':>9s} | {'drop b':>7s} {'drop c':>7s} | {'cand@half':>9s} {'B':>4s} | params b/c"
    print(hdr); print('-' * len(hdr))
    winsA = tiesA = okB = winsC = 0
    rows = []
    for task in tasks:
        b_full, c_full = stats(get(task, 'baseline', 1.0, 'full')), stats(get(task, 'candidate', 1.0, 'full'))
        b_10, c_10 = stats(get(task, 'baseline', 0.1, 'full')), stats(get(task, 'candidate', 0.1, 'full'))
        v, diff, tol = compare(task, c_full, b_full)
        sign = 1 if D.HIGHER_BETTER[task] else -1
        drop_b, drop_c = sign * (b_full[0] - b_10[0]), sign * (c_full[0] - c_10[0])
        winsA += v == 1; tiesA += v == 0
        cwin = drop_c < drop_b
        winsC += cwin
        pb = get(task, 'baseline', 1.0, 'full')[0].get('params'); pc = get(task, 'candidate', 1.0, 'full')[0].get('params')
        half_s, bflag = '', ''
        if not a.no_half:
            c_half = stats(get(task, 'candidate', 1.0, 'half'))
            vh, _, _ = compare(task, c_half, b_full)
            mem_ok = pc is not None and pb is not None and pc <= pb / 2
            okb = vh >= 0 and mem_ok
            okB += okb
            half_s, bflag = f"{c_half[0]:9.4f}", ('yes' if okb else 'no' if vh < 0 else 'mem')
        print(f"{task:10s} {D.METRIC[task]:9s} | {b_full[0]:8.4f} ±{b_full[1]:.4f} | {c_full[0]:8.4f} ±{c_full[1]:.4f} | "
              f"{'WIN' if v == 1 else 'LOSS' if v == -1 else 'tie':7s} | {b_10[0]:9.4f} {c_10[0]:9.4f} | {drop_b:7.4f} {drop_c:7.4f} | {half_s:>9s} {bflag:>4s} | {pb}/{pc}")
        rows.append(dict(task=task, baseline=b_full, candidate=c_full, verdict=int(v), baseline_tenth=b_10, candidate_tenth=c_10,
                         drop_baseline=drop_b, drop_candidate=drop_c, params=(pb, pc)))
    n = len(tasks)
    A = winsA > n / 2
    B = (not a.no_half) and okB == n
    C = winsC > n / 2
    print(f"\nA  wins on a majority of tasks at equal budget : {winsA}/{n} wins, {tiesA} ties -> {'PASS' if A else 'FAIL'}")
    if not a.no_half:
        print(f"B  matches baseline with half compute and half memory on every task : {okB}/{n} -> {'PASS' if B else 'FAIL'}")
    print(f"C  smaller drop than baseline at 1/10 data on a majority of tasks : {winsC}/{n} -> {'PASS' if C else 'FAIL'}")
    ok = (A or B) and C
    print(f"\nVERDICT: {'SUCCESS' if ok else 'NOT YET'}")
    out = os.path.join(ROOT, 'results', f'bench_B{int(budget)}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(budget=budget, seeds=seeds, rows=rows, A=bool(A), B=bool(B), C=bool(C), success=bool(ok)), open(out, 'w'), indent=1)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
