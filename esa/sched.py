"""Run a list of jobs, one single-threaded process per CPU core, and append results to a JSONL log."""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job_key(job):
    d = {k: job[k] for k in ('task', 'model', 'cfg', 'seed', 'frac')}
    d['budget'] = float(job['budget'])  # 1200 and 1200.0 are the same job
    return json.dumps(d, sort_keys=True)


def load_log(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                out[job_key(r['job'])] = r
            except Exception:
                pass
    return out


def run_jobs(jobs, log_path, workers=None, verbose=True):
    """Runs jobs not already in the log. Returns {key: result} for all jobs."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    done = load_log(log_path)
    todo = [j for j in jobs if job_key(j) not in done]
    workers = workers or os.cpu_count()
    cores = list(range(workers))
    running = {}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=ROOT)
    t0 = time.time()
    while todo or running:
        while todo and cores:
            j = todo.pop(0); c = cores.pop(0)
            p = subprocess.Popen(['taskset', '-c', str(c), sys.executable, '-m', 'esa.train', json.dumps(j)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT)
            running[p] = (j, c)
        time.sleep(0.5)
        for p in list(running):
            if p.poll() is None:
                continue
            j, c = running.pop(p); cores.append(c)
            out, err = p.communicate()
            res = None
            for line in out.splitlines():
                if line.startswith('RESULT '):
                    res = json.loads(line[7:])
            if res is None:
                res = dict(job=j, status='crashed', error=err[-1500:])
            with open(log_path, 'a') as f:
                f.write(json.dumps(res) + '\n')
            done[job_key(j)] = res
            if verbose:
                print(f"[{time.time() - t0:6.0f}s] {j['task']:10s} {j['model']:12s} seed={j['seed']} frac={j['frac']} "
                      f"budget={j['budget']} -> {res.get('status')} val={res.get('val')} test={res.get('test')} "
                      f"params={res.get('params')} steps={res.get('steps')} ep={res.get('epochs')} {json.dumps(j['cfg'])}", flush=True)
                if res.get('status') == 'crashed':
                    print(res['error'], flush=True)
    return {job_key(j): done[job_key(j)] for j in jobs}
