"""Hyperparameter + map-size sweep for PPO and Double DQN on the rover task (run overnight, compare afterwards).

Every sampled configuration is trained once per (map size, slip probability, seed) as a separate subprocess
(scripts/train_ppo.py or scripts/train_dqn.py). Everything is written under --out so a sweep is reproducible and resumable:

    <out>/sweep_info.json       environment settings, budgets, argv, versions
    <out>/configs.json          every sampled configuration (the plan)
    <out>/runs/<cfg>/<scenario>/seed<N>/   per-run log.json, config.json (exact settings), cmd.txt, stdout.txt, result.json
    <out>/summary.csv           one row per (configuration, scenario): all parameters + metrics averaged over seeds
    <out>/boundary.csv          per scenario: random-policy baseline, default vs best config for each algorithm
    <out>/marginals.csv         mean success per parameter value, per scenario (which settings matter)
    <out>/summary.json          summary.csv plus the per-seed numbers

A "scenario" is a map size + slip probability, e.g. s6_p0.3.

Examples
    py scripts/sweep.py --algo both --n-configs 20 --sizes 4 5 6 8 10 --seeds 0 1 --out sweeps/sizes
    py scripts/sweep.py --sizes 6 8 10 --budget-mult 8:2 10:4 --p-slips 0 0.3 --out sweeps/hard
    py scripts/sweep.py --algo all --n-configs 20 --sizes 4 6 8 --out sweeps/all   # adds fixed-map Q-learning/SARSA
    py scripts/sweep.py --algo ppo --mode grid --vary lr,gamma,ent_coef --sizes 6 --seeds 0 1 2 --out sweeps/ppo_grid
    py scripts/sweep.py --out sweeps/sizes --summarize-only
Re-running the same command resumes: finished runs are skipped.
Pause: create an empty file named STOP inside the --out folder (no new runs start, running ones finish, then it exits);
delete the file and re-run the same command to continue. Ctrl+C also stops safely (running jobs restart next time).
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STOP_EVENT = threading.Event()   # set by a STOP file or Ctrl+C: launch nothing new
CHILDREN, CHILDREN_LOCK = set(), threading.Lock()

# ("choice", [values]) or ("loguniform", lo, hi)
PPO_SPACE = {
    "lr": ("loguniform", 1e-4, 1e-3),
    "gamma": ("choice", [0.99, 0.995, 0.999]),
    "gae_lambda": ("choice", [0.9, 0.95, 0.98]),
    "clip": ("choice", [0.1, 0.2, 0.3]),
    "ent_coef": ("choice", [0.0, 0.01, 0.03, 0.05, 0.1]),
    "ent_decay": ("choice", [False, True]),
    "n_envs": ("choice", [8, 16, 32]),
    "n_steps": ("choice", [64, 128, 256]),
    "batch_size": ("choice", [128, 256, 512]),
    "epochs": ("choice", [3, 4, 8]),
    "target_kl": ("choice", [0.0, 0.02, 0.05]),
    "reward_scale": ("choice", [0.01, 0.05, 0.1]),
    "hidden": ("choice", [64, 128, 256]),
}
PPO_DEFAULT = dict(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, ent_decay=False, n_envs=8,
                   n_steps=128, batch_size=256, epochs=4, target_kl=0.02, reward_scale=0.05, hidden=256)

DQN_SPACE = {
    "lr": ("loguniform", 1e-4, 3e-3),
    "gamma": ("choice", [0.95, 0.99, 0.999]),
    "batch_size": ("choice", [32, 64, 128]),
    "buffer_size": ("choice", [50_000, 100_000]),
    "warmup": ("choice", [500, 1000, 5000]),
    "target_sync": ("choice", [200, 500, 1000, 2000]),
    "eps_end": ("choice", [0.01, 0.05, 0.1]),
    "eps_decay_frac": ("choice", [0.1, 0.25, 0.5]),
    "grad_clip": ("choice", [1.0, 10.0]),
    "hidden": ("choice", [64, 128, 256]),
    "double": ("choice", [True, False]),
}
DQN_DEFAULT = dict(lr=1e-3, gamma=0.99, batch_size=64, buffer_size=100_000, warmup=1000, target_sync=500,
                   eps_end=0.05, eps_decay_frac=0.25, grad_clip=10.0, hidden=256, double=True)

# Tabular Q-learning / SARSA on ONE fixed map (not comparable to PPO/DQN, which face unseen random maps)
QL_SPACE = {
    "alpha": ("loguniform", 0.02, 0.5),
    "gamma": ("choice", [0.9, 0.95, 0.99, 0.999]),
    "eps_end": ("choice", [0.01, 0.05, 0.1]),
    "eps_decay_frac": ("choice", [0.1, 0.25, 0.5, 0.8]),
    "q_init": ("choice", [0.0, 20.0, 100.0]),
    "sarsa": ("choice", [False, True]),
}
QL_DEFAULT = dict(alpha=0.1, gamma=0.99, eps_end=0.05, eps_decay_frac=0.5, q_init=0.0, sarsa=False)

SPACES = {"ppo": (PPO_SPACE, PPO_DEFAULT), "dqn": (DQN_SPACE, DQN_DEFAULT), "qlearn": (QL_SPACE, QL_DEFAULT)}
ALGOS = tuple(SPACES)
METRICS = ["success", "crater", "battery", "timeout", "mean_return", "mean_steps"]


def scenario_id(size, p_slip):
    return f"s{size}_p{p_slip:g}"


# ----------------------------------------------------------------------------- configuration generation
def sample_value(spec, rng):
    if spec[0] == "choice":
        return rng.choice(spec[1])
    lo, hi = spec[1], spec[2]
    return float(f"{math.exp(rng.uniform(math.log(lo), math.log(hi))):.2g}")


def grid_values(spec):
    if spec[0] == "choice":
        return list(spec[1])
    lo, hi = spec[1], spec[2]
    return [float(f"{v:.2g}") for v in (lo, math.sqrt(lo * hi), hi)]


def make_configs(algo, mode, n_configs, vary, rng):
    space, default = SPACES[algo]
    configs = [dict(default)]  # index 0 is always the untouched default
    if mode == "random":
        while len(configs) < n_configs:
            c = {k: sample_value(spec, rng) for k, spec in space.items()}
            if c not in configs:
                configs.append(c)
    else:  # grid over the --vary parameters, everything else at default
        names = [v for v in vary if v in space]
        if not names:
            sys.exit(f"--vary needs parameter names from: {', '.join(space)}")
        combos = [dict()]
        for n in names:
            combos = [dict(c, **{n: v}) for c in combos for v in grid_values(space[n])]
        for combo in combos:
            c = dict(default, **combo)
            if c not in configs:
                configs.append(c)
    return [{"cfg_id": f"{algo}_{i:03d}", "algo": algo, "params": c} for i, c in enumerate(configs)]


# ----------------------------------------------------------------------------- running one training run
def flagify(name):
    return "--" + name.replace("_", "-")


def build_cmd(task, run_dir, a):
    algo, params = task["algo"], task["params"]
    script = ROOT / "scripts" / {"ppo": "train_ppo.py", "dqn": "train_dqn.py", "qlearn": "train_qlearn.py"}[algo]
    cmd = [sys.executable, str(script), "--seed", str(task["seed"]), "--out", str(run_dir), "--size", str(task["size"]),
           "--p-slip", str(task["p_slip"]), "--b0", str(a.b0)]
    if algo == "qlearn":
        cmd += ["--map-seed", str(a.map_seed), "--eval-episodes", str(a.q_eval_episodes)]
    else:
        cmd += ["--eval-maps", str(a.eval_maps)]
    for k, v in params.items():
        if k == "double":
            if not v:
                cmd.append("--vanilla")
        elif isinstance(v, bool):
            if v:
                cmd.append(flagify(k))
        else:
            cmd += [flagify(k), str(v)]
    mult = task["mult"]
    if algo == "ppo":
        iters = max(1, int(a.ppo_steps * mult) // (params["n_envs"] * params["n_steps"]))
        cmd += ["--iterations", str(iters), "--eval-every", str(max(1, iters // a.evals))]
    else:
        eps = max(1, int((a.q_episodes if algo == "qlearn" else a.dqn_episodes) * mult))
        cmd += ["--episodes", str(eps), "--eval-every", str(max(1, eps // a.evals))]
    return cmd


def run_dir_for(task, a):
    return Path(a.out) / "runs" / task["cfg_id"] / scenario_id(task["size"], task["p_slip"]) / f"seed{task['seed']}"


def summarize_log(log):
    """Final metrics = mean of the last 3 evaluations (a single evaluation is noisy)."""
    tail = log[-3:]
    out = {m: statistics.fmean(e[m] for e in tail) for m in METRICS}
    out["final_success"] = log[-1]["success"]
    out["best_success"] = max(e["success"] for e in log)
    out["env_steps"] = log[-1].get("steps")
    out["n_evals"] = len(log)
    return out


def run_one(task, a, deadline):
    run_dir = run_dir_for(task, a)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if deadline and time.time() > deadline:
        return None  # out of time budget: leave for the next (resumed) invocation
    if not STOP_EVENT.is_set() and (Path(a.out) / "STOP").exists():
        STOP_EVENT.set()
        print("STOP file found: no new runs will start; waiting for the running ones to finish", flush=True)
    if STOP_EVENT.is_set():
        return None  # paused: leave for the next (resumed) invocation
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(task, run_dir, a)
    (run_dir / "cmd.txt").write_text(" ".join(cmd))
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    t0, status = time.time(), "ok"
    with open(run_dir / "stdout.txt", "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        with CHILDREN_LOCK:
            CHILDREN.add(proc)
        try:
            rc = proc.wait(timeout=a.timeout_min * 60 if a.timeout_min else None)
            if rc != 0:
                status = f"failed(rc={rc})"
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
            status = "timeout"
        finally:
            with CHILDREN_LOCK:
                CHILDREN.discard(proc)
    result = dict(cfg_id=task["cfg_id"], algo=task["algo"], seed=task["seed"], size=task["size"],
                  p_slip=task["p_slip"], params=task["params"], status=status,
                  runtime_s=round(time.time() - t0, 1))
    logs = list(run_dir.glob("*/log.json"))
    if logs:
        try:
            result.update(summarize_log(json.loads(logs[0].read_text())))
        except (ValueError, IndexError, KeyError):
            result["status"] += "+no_metrics"
    else:
        result["status"] += "+no_log"
    # a failed/interrupted run is recorded in failed.json (NOT result.json), so the next invocation retries it
    failed_path = run_dir / "failed.json"
    if status.startswith("failed"):
        failed_path.write_text(json.dumps(result, indent=1))
    else:
        result_path.write_text(json.dumps(result, indent=1))
        failed_path.unlink(missing_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] {task['cfg_id']} {scenario_id(task['size'], task['p_slip'])} seed{task['seed']} "
          f"{result['status']} success={result.get('success', float('nan')):.2f} ({result['runtime_s']:.0f}s)", flush=True)
    return result


# ----------------------------------------------------------------------------- random-policy baseline
def random_baseline(out, scenarios, b0, n=300):
    """Success rate of a uniformly random policy per scenario: shows where the task becomes an exploration problem."""
    from rover.env import RoverEnv
    path = Path(out) / "random_baseline.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for size, p_slip in scenarios:
        sid = scenario_id(size, p_slip)
        if sid in cache:
            continue
        rng, wins = random.Random(0), 0
        for i in range(n):
            e = RoverEnv(size=size, p_slip=p_slip, b0=b0)
            e.reset(seed=50_000 + i)
            done, info = False, {}
            while not done:
                _, _, term, trunc, info = e.step(rng.randrange(4))
                done = term or trunc
            wins += info["outcome"] == "success"
        cache[sid] = wins / n
    path.write_text(json.dumps(cache, indent=1))
    return cache


# ----------------------------------------------------------------------------- aggregation
def mean_std(xs):
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)


def bin_label(algo, param, value):
    spec = SPACES[algo][0][param]
    if spec[0] == "choice":
        return str(value)
    lo, hi = spec[1], spec[2]
    i = min(2, max(0, int(3 * (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)))))
    edges = [lo * (hi / lo) ** (k / 3) for k in range(4)]
    return f"{edges[i]:.2g}-{edges[i + 1]:.2g}"


def summarize(out, b0):
    out = Path(out)
    results = [json.loads(p.read_text()) for p in sorted((out / "runs").glob("*/*/seed*/result.json"))]
    by_cell = {}
    for r in results:
        by_cell.setdefault((r["cfg_id"], r["size"], r["p_slip"]), []).append(r)
    rows = []
    for (cfg_id, size, p_slip), rs in sorted(by_cell.items()):
        good = [r for r in rs if "success" in r]
        row = {"cfg_id": cfg_id, "algo": rs[0]["algo"], "size": size, "p_slip": p_slip, "n_seeds": len(rs),
               "n_ok": len(good), **rs[0]["params"]}
        for m in METRICS + ["best_success", "final_success"]:
            mu, sd = mean_std([r[m] for r in good])
            row[f"mean_{m}"] = round(mu, 4)
            if m in ("success", "mean_return"):
                row[f"std_{m}"] = round(sd, 4)
        row["env_steps"] = good[0].get("env_steps") if good else None
        row["runtime_min"] = round(statistics.fmean(r["runtime_s"] for r in rs) / 60, 2)
        row["statuses"] = ";".join(sorted({r["status"] for r in rs}))
        row["per_seed_success"] = ";".join(f"{r.get('success', float('nan')):.2f}" for r in sorted(rs, key=lambda r: r["seed"]))
        rows.append(row)
    rows.sort(key=lambda r: (r["algo"], r["size"], r["p_slip"], -r["mean_success"], -r["mean_mean_return"]))
    if not rows:
        print("no finished runs yet"); return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"configs": rows, "runs": results}, indent=1))

    marg = {}  # (algo, size, p_slip, param, label) -> list of per-config mean_success
    for r in rows:
        if r["n_ok"] == 0:
            continue
        for p in SPACES[r["algo"]][0]:
            marg.setdefault((r["algo"], r["size"], r["p_slip"], p, bin_label(r["algo"], p, r[p])), []).append(r["mean_success"])
    with open(out / "marginals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["algo", "size", "p_slip", "param", "value", "n_configs", "mean_success"])
        for (algo, size, p_slip, p, lab), xs in sorted(marg.items()):
            w.writerow([algo, size, p_slip, p, lab, len(xs), round(statistics.fmean(xs), 4)])

    # ---- boundary table: where does each algorithm stop working?
    scenarios = sorted({(r["size"], r["p_slip"]) for r in rows})
    rnd = random_baseline(out, scenarios, b0)
    table = []
    for size, p_slip in scenarios:
        cell = {"scenario": scenario_id(size, p_slip), "size": size, "p_slip": p_slip,
                "random_success": rnd[scenario_id(size, p_slip)]}
        for algo in ALGOS:
            rs = [r for r in rows if r["algo"] == algo and r["size"] == size and r["p_slip"] == p_slip and r["n_ok"]]
            if not rs:
                continue
            default = next((r for r in rs if r["cfg_id"].endswith("_000")), None)
            best = max(rs, key=lambda r: (r["mean_success"], r["mean_mean_return"]))
            cell[f"{algo}_default"] = default["mean_success"] if default else None
            cell[f"{algo}_best"] = best["mean_success"]
            cell[f"{algo}_best_cfg"] = best["cfg_id"]
            cell[f"{algo}_n_configs"] = len(rs)
            cell[f"{algo}_best_changed"] = json.dumps({k: best[k] for k in SPACES[algo][1] if best[k] != SPACES[algo][1][k]})
        table.append(cell)
    bcols = []
    for c in table:
        bcols += [k for k in c if k not in bcols]
    with open(out / "boundary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bcols, restval="")
        w.writeheader(); w.writerows(table)

    def fmt(x):
        return "   -  " if x is None else f"{x:6.2f}"
    print("\nSuccess rate by scenario (mean over seeds; last-3-eval average):")
    print(f"{'scenario':<10}{'random':>8}{'PPO dflt':>10}{'PPO best':>10}{'DQN dflt':>10}{'DQN best':>10}"
          f"{'Q dflt*':>10}{'Q best*':>10}")
    for c in table:
        print(f"{c['scenario']:<10}{c['random_success']:8.2f}{fmt(c.get('ppo_default')):>10}{fmt(c.get('ppo_best')):>10}"
              f"{fmt(c.get('dqn_default')):>10}{fmt(c.get('dqn_best')):>10}"
              f"{fmt(c.get('qlearn_default')):>10}{fmt(c.get('qlearn_best')):>10}")
    print("* Q-learning is trained and tested on ONE fixed map, so it is an easier problem than PPO/DQN (unseen maps).")
    print(f"\nWrote {out / 'summary.csv'}, {out / 'boundary.csv'}, {out / 'marginals.csv'}, {out / 'summary.json'}")


# ----------------------------------------------------------------------------- main
def parse_mults(items):
    mults = {}
    for it in items:
        s, m = it.split(":")
        mults[int(s)] = float(m)
    return mults


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["ppo", "dqn", "qlearn", "both", "all"], default="both",
                   help="both = ppo+dqn, all = ppo+dqn+qlearn")
    p.add_argument("--mode", choices=["random", "grid"], default="random")
    p.add_argument("--n-configs", type=int, default=20, help="random mode: configs per algorithm (incl. the default)")
    p.add_argument("--vary", default="", help="grid mode: comma-separated parameters to sweep (others at default)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--sweep-seed", type=int, default=0, help="seed for sampling configurations")
    p.add_argument("--sizes", type=int, nargs="+", default=[6], help="map sizes to test, e.g. 4 5 6 8 10")
    p.add_argument("--p-slips", type=float, nargs="+", default=[0.30], help="slip probabilities to test")
    p.add_argument("--b0", type=int, default=15)
    p.add_argument("--ppo-steps", type=int, default=2_000_000, help="PPO environment-step budget per run")
    p.add_argument("--dqn-episodes", type=int, default=10_000, help="DQN episode budget per run")
    p.add_argument("--q-episodes", type=int, default=20_000, help="Q-learning episode budget per run")
    p.add_argument("--q-eval-episodes", type=int, default=200, help="Q-learning evaluation episodes on its fixed map")
    p.add_argument("--map-seed", type=int, default=10_000, help="seed of the fixed map used by Q-learning")
    p.add_argument("--budget-mult", nargs="*", default=[], metavar="SIZE:MULT",
                   help="scale both budgets for bigger maps, e.g. 8:2 10:4 (default 1 for unlisted sizes)")
    p.add_argument("--evals", type=int, default=20, help="evaluations per run")
    p.add_argument("--eval-maps", type=int, default=10, help="unseen maps per evaluation (x20 episodes each)")
    p.add_argument("--jobs", type=int, default=2, help="runs in parallel (each uses 1 CPU thread)")
    p.add_argument("--timeout-min", type=float, default=0, help="kill a single run after this many minutes (0 = no limit)")
    p.add_argument("--max-hours", type=float, default=0, help="stop launching new runs after this long (0 = no limit)")
    p.add_argument("--out", default="sweeps/sweep")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    if a.summarize_only:
        summarize(out, a.b0); return

    cfg_path = out / "configs.json"
    if cfg_path.exists():
        configs = json.loads(cfg_path.read_text())
        print(f"resuming: loaded {len(configs)} configurations from {cfg_path}")
    else:
        rng = random.Random(a.sweep_seed)
        algos = {"both": ["ppo", "dqn"], "all": list(ALGOS)}.get(a.algo, [a.algo])
        configs = [c for algo in algos for c in make_configs(algo, a.mode, a.n_configs, a.vary.split(","), rng)]
    mults = parse_mults(a.budget_mult)
    # order: seed -> scenario (small maps first) -> config index -> algo, so a partial overnight run covers everything evenly
    tasks = sorted(({"cfg_id": c["cfg_id"], "algo": c["algo"], "params": c["params"], "seed": s, "size": size,
                     "p_slip": ps, "mult": mults.get(size, 1.0)}
                    for c in configs for s in a.seeds for size in a.sizes for ps in a.p_slips),
                   key=lambda t: (t["seed"], t["size"], t["p_slip"], int(t["cfg_id"].split("_")[1]), t["algo"]))
    print(f"{len(configs)} configurations x {len(a.sizes)} sizes x {len(a.p_slips)} slip values x {len(a.seeds)} seeds "
          f"= {len(tasks)} runs (b0={a.b0}; PPO {a.ppo_steps:,} steps, DQN {a.dqn_episodes:,} episodes, x budget-mult)")
    if a.dry_run:
        for t in tasks[:6]:
            print(" ".join(build_cmd(t, run_dir_for(t, a), a)))
        print("... (dry run, nothing executed)")
        return

    if (out / "STOP").exists():
        sys.exit(f"{out / 'STOP'} exists (left from a pause). Delete that file, then run the command again.")
    out.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(configs, indent=1))
    info_path = out / "sweep_info.json"
    if not info_path.exists():
        try:
            import torch
            tv = torch.__version__
        except ImportError:
            tv = None
        info_path.write_text(json.dumps(dict(argv=sys.argv, args=vars(a), started=time.ctime(), python=sys.version,
                                             torch=tv, platform=platform.platform(),
                                             search_spaces={k: {n: list(s) for n, s in v[0].items()} for k, v in SPACES.items()},
                                             defaults={k: v[1] for k, v in SPACES.items()}), indent=1, default=str))
    deadline = time.time() + a.max_hours * 3600 if a.max_hours else None
    ex = ThreadPoolExecutor(max_workers=max(1, a.jobs))
    try:
        list(ex.map(lambda t: run_one(t, a, deadline), tasks))
    except KeyboardInterrupt:
        STOP_EVENT.set()
        print("\ninterrupted: stopping running jobs (they restart from scratch next time); "
              "finished runs are saved. Re-run the same command to resume.", flush=True)
        with CHILDREN_LOCK:
            for proc in list(CHILDREN):
                proc.terminate()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    n_failed = len(list((out / "runs").glob("*/*/seed*/failed.json")))
    if n_failed:
        print(f"{n_failed} run(s) failed or were interrupted; they will be retried when you run the command again.")
    if STOP_EVENT.is_set():
        print("paused. To continue: delete the STOP file (if you used one) and run the same command again.")
    summarize(out, a.b0)


if __name__ == "__main__":
    main()
