"""Ad-hoc exploration runs: python scripts/explore.py <jobs.json> [log]. jobs.json is a list of job dicts
(task, model, cfg, optional seed/frac/budget/tag). Results append to logs/explore.jsonl."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from esa.sched import run_jobs, job_key

jobs = json.load(open(sys.argv[1]))
log = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'logs', 'explore.jsonl')
for j in jobs:
    j.setdefault('seed', 0); j.setdefault('frac', 1.0); j.setdefault('budget', 300)
    j.setdefault('n_evals', 2); j.setdefault('val_cap', 1000); j.setdefault('test_cap', 1000)
res = run_jobs(jobs, log)
for j in jobs:
    r = res[job_key(j)]
    print(j.get('tag', ''), j['task'], j['model'], r.get('status'), 'val', r.get('val'), 'test', r.get('test'), 'steps', r.get('steps'),
          'ep', r.get('epochs'), 'P', r.get('params'), 'loss', r.get('train_loss'))
