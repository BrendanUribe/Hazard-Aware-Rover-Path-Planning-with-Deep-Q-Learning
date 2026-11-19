"""Report figures, trajectory maps and GIF/MP4 rollouts from FINISHED runs -- no retraining needed.

From a sweep (scripts/sweep.py):
    py scripts/report.py --sweep sweeps/sizes
    py scripts/report.py --sweep sweeps/sizes --scenarios s6_p0.3 s8_p0.3 --trials 6 --mp4
From individual training runs (any folder with log.json + model.pt / qtable.npy):
    py scripts/report.py --run runs_s6b/ppo_seed0 --algo ppo --size 6 --p-slip 0.3 --b0 15 --label "PPO 6x6"

Everything is written to <sweep>/report/ (or --out) as PNG + PDF (figures) and GIF (rollouts):
    boundary                 success vs map size: random policy, default and best configuration per algorithm
    outcomes                 stacked success / crater / battery / timeout bars for the best configuration
    marginals_<algo>         mean success per parameter value (which settings matter)
    curves_<scenario>        learning curves (success, return vs environment steps), mean +/- std over seeds
    maps_<algo>_<scenario>   greedy-policy paths on unseen maps (Q-learning: its fixed map)
    rollout_<algo>_<scenario>.gif   step-by-step animation over several trials
    results_table.md         the boundary table as markdown, ready to paste into a report
"""
import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rover.env import RoverEnv  # noqa: E402
from rover.viz import draw_map  # noqa: E402
from sweep import ALGOS, SPACES, scenario_id  # noqa: E402

COLOR = {"ppo": "#0072B2", "dqn": "#D55E00", "qlearn": "#009E73", "random": "#7f7f7f"}
NAME = {"ppo": "PPO", "dqn": "Double DQN", "qlearn": "Q-learning (fixed map)"}
OUTCOME_COLOR = {"success": "#009E73", "crater": "#D55E00", "battery": "#E69F00", "timeout": "#999999"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})


# ----------------------------------------------------------------------------- small helpers
def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return x


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def title_of(size, p_slip):
    return f"{int(size)}x{int(size)} map, slip {p_slip:g}"


def find_run(seed_dir):
    seed_dir = Path(seed_dir)
    if (seed_dir / "log.json").exists():
        return seed_dir
    for d in sorted(seed_dir.iterdir()) if seed_dir.exists() else []:
        if d.is_dir() and (d / "log.json").exists():
            return d
    return None


def load_log(run):
    return json.loads((Path(run) / "log.json").read_text())


def xkey(log):
    return next(k for k in ("steps", "episode", "iteration") if k in log[0])


def best_row(rows):
    return max(rows, key=lambda r: (r["mean_success"], r["mean_mean_return"]))


def curve(seed_dirs, key):
    """Mean +/- std of a log metric across seeds on a common x grid."""
    series = []
    for sd in seed_dirs:
        run = find_run(sd)
        if run is None:
            continue
        log = load_log(run)
        xk = xkey(log)
        series.append((np.array([e[xk] for e in log], float), np.array([e[key] for e in log], float)))
    if not series:
        return None
    lo, hi = max(s[0][0] for s in series), min(s[0][-1] for s in series)
    grid = np.linspace(lo, max(lo, hi), min(len(s[0]) for s in series))
    Y = np.stack([np.interp(grid, x, y) for x, y in series])
    return grid, Y.mean(0), Y.std(0)


# ----------------------------------------------------------------------------- sweep-level figures
def fig_boundary(rows, out):
    p_slips = sorted({r["p_slip"] for r in rows})
    fig, axes = plt.subplots(1, len(p_slips), figsize=(5.8 * len(p_slips), 4.6), squeeze=False, sharey=True)
    for ax, ps in zip(axes[0], p_slips):
        rs = sorted((r for r in rows if r["p_slip"] == ps), key=lambda r: r["size"])
        sizes = [r["size"] for r in rs]
        ax.plot(sizes, [r["random_success"] for r in rs], ":", color=COLOR["random"], marker="s", label="random policy")
        for algo in ALGOS:
            for kind, ls in (("best", "-"), ("default", "--")):
                ys = [r.get(f"{algo}_{kind}") for r in rs]
                if all(y is None for y in ys):
                    continue
                ax.plot(sizes, [np.nan if y is None else y for y in ys], ls, color=COLOR[algo], marker="o",
                        label=f"{NAME[algo]} - {kind}")
        ax.set_xticks(sizes); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("map size (n x n)"); ax.set_title(f"slip probability {ps:g}")
    axes[0][0].set_ylabel("success rate (last-3-eval mean, avg over seeds)")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Where does each method stop working?")
    save(fig, out, "boundary")


def fig_outcomes(rows, out):
    rows = [r for r in rows if r["n_ok"]]
    groups = sorted({(r["size"], r["p_slip"]) for r in rows})
    algos = [a for a in ALGOS if any(r["algo"] == a for r in rows)]
    width = 0.8 / len(algos)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(groups) * len(algos) / 1.5), 4.8))
    for gi, (size, ps) in enumerate(groups):
        for ai, algo in enumerate(algos):
            cand = [r for r in rows if r["algo"] == algo and (r["size"], r["p_slip"]) == (size, ps)]
            if not cand:
                continue
            best, bottom = best_row(cand), 0.0
            x = gi + (ai - (len(algos) - 1) / 2) * width
            for k, c in OUTCOME_COLOR.items():
                v = best.get(f"mean_{k}") or 0.0
                ax.bar(x, v, width * 0.92, bottom=bottom, color=c)
                bottom += v
            ax.text(x, -0.03, {"ppo": "P", "dqn": "D", "qlearn": "Q"}[algo], ha="center", va="top", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([scenario_id(int(s), p) for s, p in groups])
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 1.02); ax.set_ylabel("share of evaluation episodes")
    ax.set_title("Outcome breakdown of the best configuration  (P = PPO, D = DQN, Q = Q-table on a fixed map)")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in OUTCOME_COLOR.items()], loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    save(fig, out, "outcomes")


def sort_key(s):
    try:
        return (0, float(s.split("-")[0]))
    except ValueError:
        return (1, s)


def fmt_val(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def fig_marginals(mrows, algo, out):
    rows = [r for r in mrows if r["algo"] == algo]
    if not rows:
        return
    params = list(SPACES[algo][0])
    scens = sorted({(r["size"], r["p_slip"]) for r in rows})
    ncol = 4
    nrow = math.ceil(len(params) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False, sharey=True)
    for ax, p in zip(axes.flat, params):
        vals = sorted({fmt_val(r["value"]) for r in rows if r["param"] == p}, key=sort_key)
        for sc in scens:
            ys = []
            for v in vals:
                m = [r["mean_success"] for r in rows if r["param"] == p and (r["size"], r["p_slip"]) == sc
                     and fmt_val(r["value"]) == v]
                ys.append(m[0] if m else np.nan)
            ax.plot(range(len(vals)), ys, marker="o", label=scenario_id(int(sc[0]), sc[1]))
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals, rotation=30, fontsize=7)
        ax.set_title(p, fontsize=10); ax.set_ylim(-0.02, 1.02)
    for ax in list(axes.flat)[len(params):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(6, len(l)), fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{NAME[algo]}: mean success by parameter value (each point averages the configurations using it)", y=1.07)
    fig.tight_layout()
    save(fig, out, f"marginals_{algo}")


def fig_learning_curves(sweep, rows, scen, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for algo in ALGOS:
        cand = [r for r in rows if r["algo"] == algo and scenario_id(int(r["size"]), r["p_slip"]) == scen and r["n_ok"]]
        if not cand:
            continue
        best = best_row(cand)
        dflt = next((r for r in cand if r["cfg_id"].endswith("_000")), None)
        items = [("best" if not dflt or dflt["cfg_id"] != best["cfg_id"] else "default = best", best, "-")]
        if dflt and dflt["cfg_id"] != best["cfg_id"]:
            items.append(("default", dflt, "--"))
        for tag, r, ls in items:
            dirs = sorted((sweep / "runs" / r["cfg_id"] / scen).glob("seed*"))
            for ax, key in zip(axes, ("success", "mean_return")):
                c = curve(dirs, key)
                if c is None:
                    continue
                x, m, s = c
                ax.plot(x, m, ls, color=COLOR[algo], label=f"{NAME[algo]} ({tag})", linewidth=2)
                ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.13)
    axes[0].set_ylabel("greedy success rate"); axes[1].set_ylabel("greedy mean return")
    for ax in axes:
        ax.set_xlabel("environment steps")
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    r0 = rows[0]
    size, ps = next((r["size"], r["p_slip"]) for r in rows if scenario_id(int(r["size"]), r["p_slip"]) == scen)
    fig.suptitle(f"Learning curves - {title_of(size, ps)} (mean +/- std over seeds)")
    save(fig, out, f"curves_{scen}")


def write_table(bound, out):
    cols = [("random_success", "random")] + [(f"{a}_{k}", f"{NAME[a].split(' (')[0]} {k}") for a in ALGOS
                                             for k in ("default", "best")]
    cols = [(c, n) for c, n in cols if any(r.get(c) is not None for r in bound)]
    lines = ["| scenario | " + " | ".join(n for _, n in cols) + " |", "|---|" + "---|" * len(cols)]
    for r in sorted(bound, key=lambda r: (r["p_slip"], r["size"])):
        lines.append(f"| {r['scenario']} | " + " | ".join("-" if r.get(c) is None else f"{r[c]:.2f}" for c, _ in cols) + " |")
    lines += ["", "Success rate = mean over seeds of the last-3-evaluation average. Q-learning is trained and tested on one "
              "fixed map (an easier problem than PPO/DQN, which are tested on unseen random maps).", "",
              "Best configuration per cell (parameters changed from default):", ""]
    for r in sorted(bound, key=lambda r: (r["p_slip"], r["size"])):
        for a in ALGOS:
            if r.get(f"{a}_best_cfg"):
                lines.append(f"- {r['scenario']} {NAME[a]}: `{r[f'{a}_best_cfg']}` {r.get(f'{a}_best_changed') or '{}'}")
    (out / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out / "results_table.md")


# ----------------------------------------------------------------------------- policies, rollouts, maps, animation
def load_policy(run, algo, env_kw, hidden=None):
    """Return (policy(env, obs) -> action, fixed_grid or None) for a saved run."""
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    hidden = hidden or cfg.get("hidden", 256)
    if algo == "qlearn":
        Q = np.load(run / "qtable.npy")
        return (lambda env, obs: int(np.argmax(Q[env.pos[0], env.pos[1], int(env.battery)]))), np.load(run / "fixed_map.npy")
    import torch
    obs_dim = RoverEnv(**env_kw).obs_dim
    if algo == "ppo":
        from rover.ppo import PPOAgent, PPOConfig
        ag = PPOAgent(obs_dim, PPOConfig(hidden=hidden))
        ag.ac.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
        return (lambda env, obs: ag.act(obs)), None
    from rover.dqn import DQNAgent, DQNConfig
    ag = DQNAgent(obs_dim, DQNConfig(hidden=hidden))
    ag.q.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    return (lambda env, obs: ag.act(obs, 0.0)), None


def make_rollouts(policy, grid, env_kw, n, seed0):
    out = []
    for i in range(n):
        env = RoverEnv(fixed_map=grid, **env_kw) if grid is not None else RoverEnv(**env_kw)
        obs, _ = env.reset(seed=seed0 + i)
        path, bat, done, info = [env.pos], [int(env.battery)], False, {}
        while not done:
            obs, _, term, trunc, info = env.step(policy(env, obs))
            path.append(env.pos); bat.append(int(env.battery)); done = term or trunc
        out.append({"grid": env.grid.copy(), "path": path, "battery": bat, "outcome": info["outcome"]})
    return out


OUTCOME_TITLE = {"success": "#009E73", "crater": "#D55E00", "battery": "#B07A00", "timeout": "#666666"}


def fig_trajectories(rollouts, title, out, name, n=4):
    rollouts = rollouts[:n]
    fig, axes = plt.subplots(1, len(rollouts), figsize=(4.6 * len(rollouts), 4.8), squeeze=False)
    for ax, ro in zip(axes[0], rollouts):
        draw_map(ro["grid"], ro["path"], ax, f"{ro['outcome']} ({len(ro['path']) - 1} steps, battery {ro['battery'][-1]})")
        ax.get_legend().remove()
        ax.title.set_color(OUTCOME_TITLE.get(ro["outcome"], "black"))
        ax.grid(False)
    from rover.viz import COLORS, LABELS
    fig.legend(handles=[Patch(color=COLORS[i], label=LABELS[i]) for i in COLORS], loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, out, name)


def make_animation(rollouts, label, path, fps=4, mp4=False):
    frames = []
    for i, ro in enumerate(rollouts):
        L = len(ro["path"])
        frames += [(i, t) for t in range(L)] + [(i, L - 1)] * 4   # hold the final frame of each trial
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.05)

    def draw(k):
        i, t = frames[k]
        ro = rollouts[i]
        ax.clear()
        draw_map(ro["grid"], ro["path"][:t + 1], ax)
        ax.get_legend().remove(); ax.grid(False)
        x, y = ro["path"][t]
        ax.plot(y, x, "o", mfc="red", mec="white", ms=13, mew=1.5, zorder=5)
        last = t == len(ro["path"]) - 1
        ax.set_title(f"{label}\ntrial {i + 1}/{len(rollouts)}   step {t}   battery {ro['battery'][t]}"
                     + (f"   ->  {ro['outcome'].upper()}" if last else ""), fontsize=10,
                     color=OUTCOME_TITLE.get(ro["outcome"], "black") if last else "black")

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    print("wrote", path)
    if mp4:
        if shutil.which("ffmpeg"):
            anim.save(path.with_suffix(".mp4"), writer=FFMpegWriter(fps=fps), dpi=100)
            print("wrote", path.with_suffix(".mp4"))
        else:
            print("ffmpeg not found: skipped MP4 (the GIF can be used in slides)")
    plt.close(fig)


def visuals_for_run(run, algo, env_kw, label, out, name, a, hidden=None):
    policy, grid = load_policy(run, algo, env_kw, hidden)
    ros = make_rollouts(policy, grid, env_kw, max(a.trials, a.n_maps), a.rollout_seed)
    where = "its fixed map" if grid is not None else "unseen maps"
    fig_trajectories(ros, f"{label} - greedy path on {where}", out, f"maps_{name}", a.n_maps)
    if not a.no_gif:
        make_animation(ros[:a.trials], label, out / f"rollout_{name}.gif", a.fps, a.mp4)


# ----------------------------------------------------------------------------- entry points
def report_sweep(a):
    sw = Path(a.sweep)
    out = Path(a.out) if a.out else sw / "report"
    for need in ("summary.csv", "boundary.csv"):
        if not (sw / need).exists():
            sys.exit(f"{sw / need} missing - run: py scripts/sweep.py --out {sw} --summarize-only")
    info = json.loads((sw / "sweep_info.json").read_text())
    b0 = info["args"]["b0"]
    summ, bound = read_csv(sw / "summary.csv"), read_csv(sw / "boundary.csv")
    marg = read_csv(sw / "marginals.csv") if (sw / "marginals.csv").exists() else []
    algos = [x for x in ALGOS if (not a.algos or x in a.algos) and any(r["algo"] == x for r in summ)]

    fig_boundary(bound, out)
    fig_outcomes(summ, out)
    write_table(bound, out)
    for algo in algos:
        fig_marginals(marg, algo, out)

    scens = sorted({scenario_id(int(r["size"]), r["p_slip"]) for r in summ}, key=lambda s: (float(s.split("_p")[1]), int(s[1:].split("_")[0])))
    for scen in scens:
        if a.scenarios and scen not in a.scenarios:
            continue
        fig_learning_curves(sw, summ, scen, out)
        for algo in algos:
            cand = [r for r in summ if r["algo"] == algo and scenario_id(int(r["size"]), r["p_slip"]) == scen and r["n_ok"]]
            if not cand:
                continue
            best = best_row(cand)
            run = find_run(sw / "runs" / best["cfg_id"] / scen / f"seed{a.seed}")
            if run is None:
                print(f"skip {algo} {scen}: no run for seed {a.seed}")
                continue
            env_kw = dict(size=int(best["size"]), p_slip=best["p_slip"], b0=int(b0))
            label = f"{NAME[algo]} (best cfg {best['cfg_id']}) - {title_of(best['size'], best['p_slip'])}"
            visuals_for_run(run, algo, env_kw, label, out, f"{algo}_{scen}", a)
    print(f"\nAll report files are in {out}")


def report_runs(a):
    out = Path(a.out or "report")
    for rd in a.run:
        run = find_run(rd)
        if run is None:
            print(f"skip {rd}: no log.json found"); continue
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
        env_kw = dict(size=cfg.get("size", a.size), p_slip=cfg.get("p_slip", a.p_slip), b0=cfg.get("b0", a.b0))
        label = a.label or run.parent.name + "/" + run.name
        name = slug(label)
        log = load_log(run)
        xk = xkey(log)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (key, yl) in zip(axes, (("success", "greedy success rate"), ("mean_return", "greedy mean return"),
                                        ("train_return", "training return (exploring policy)"))):
            ax.plot([e[xk] for e in log], [e[key] for e in log], color=COLOR[a.algo], linewidth=2)
            ax.set_xlabel(xk); ax.set_ylabel(yl)
        axes[0].set_ylim(-0.02, 1.02)
        fig.suptitle(f"{label} - learning curves ({title_of(env_kw['size'], env_kw['p_slip'])})")
        save(fig, out, f"curves_{name}")
        visuals_for_run(run, a.algo, env_kw, label, out, name, a, a.hidden)
    print(f"\nAll report files are in {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", help="sweep output folder (made by scripts/sweep.py)")
    p.add_argument("--run", nargs="+", help="individual run folder(s) with log.json + model.pt / qtable.npy")
    p.add_argument("--algo", choices=["ppo", "dqn", "qlearn"], help="algorithm of the --run folder(s)")
    p.add_argument("--size", type=int, default=10, help="--run mode: map size (ignored if the run has config.json)")
    p.add_argument("--p-slip", type=float, default=0.30)
    p.add_argument("--b0", type=int, default=15)
    p.add_argument("--hidden", type=int, default=None, help="--run mode: network width if the run has no config.json")
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--scenarios", nargs="*", help="sweep mode: only these scenarios, e.g. s6_p0.3")
    p.add_argument("--algos", nargs="*", choices=list(ALGOS), help="sweep mode: only these algorithms")
    p.add_argument("--seed", type=int, default=0, help="sweep mode: which training seed's model to visualise")
    p.add_argument("--n-maps", type=int, default=4, help="paths drawn in each trajectory figure")
    p.add_argument("--trials", type=int, default=5, help="episodes shown in each GIF")
    p.add_argument("--rollout-seed", type=int, default=10_000, help="map/slip seed of the first rollout")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--mp4", action="store_true", help="also write .mp4 (needs ffmpeg on PATH)")
    p.add_argument("--fps", type=int, default=4)
    a = p.parse_args()
    if a.sweep:
        report_sweep(a)
    elif a.run:
        if not a.algo:
            p.error("--run needs --algo")
        report_runs(a)
    else:
        p.error("give --sweep or --run")


if __name__ == "__main__":
    main()
