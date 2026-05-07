import os
import gzip
import pickle
import time
import json
import csv as csv_mod
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from layout import RoundaboutSegmentedEnv
from seed import make_seeds, set_global_seeds

# ====== TensorBoard (optional) ======
BASE_LOG_DIR = "logs/roundabout_q_factorized"
SummaryWriter = None
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None

# ====== Hyperparameters ======
RUNS = 75                   # run_00 .. run_74
SEED0            = 42
DISCOUNT        = 0.97
EPISODES        = 6000
SHOW_EVERY      = 500
SHOW = True
REWARD_CLIP     = 1.0

EPSILON0        = 0.90
EPSILON_DECAY   = 0.9994
EPSILON_MIN     = 0.02

ALPHA0_SHARED   = 0.10
ALPHA_FLOOR     = 0.02  # floor for late learning

# ====== Early stopping (tunable) ======
EARLY_STOP_WINDOW         = 500     # compare last 500 vs previous 500
EARLY_STOP_MIN_IMPROVEMENT = 0.02   # absolute delta on rolling mean reward
EARLY_CHECK_EVERY         = 500      # check every N episodes once enough history exists

ANGLE_DEG = [65, 90, 120]
DIR_DEG = [0, 45, 90, 135, 180, 225, 270, 315]


BASE_SAVE_DIR   = "logs/roundabout_q_factorized"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
os.makedirs(BASE_LOG_DIR, exist_ok=True)

# Featurization bins (for the base 's' that is shared across positions)
COM_BINS        = 4      # 4x4 center-of-mass grid
DENSITY_BINS    = 4      # 0..3 (peak occupancy of any 20m x 20m bin)
# NOTE: previous version used `total // DENSITY_BIN_SZ` for the density bin,
# but `total` equals N (=20) every step (all cars are always binned), so the
# bin was constant at 2 and the [0..3] state range was effectively dead.
# We now bin on `grid.max()` (peak crowding in any cell) so all four bins
# are reachable and the feature actually carries information.

# ====== Env shape discovery ======
def make_env(render: bool, seed: int | None = None):
    return RoundaboutSegmentedEnv(render_mode="human" if render else None, seed=seed)

_env0 = make_env(render=False)
assert isinstance(_env0.action_space, gym.spaces.MultiDiscrete), "Expected MultiDiscrete action space"
N0, N1 = map(int, _env0.action_space.nvec)   # (8,3)
A = N0 * N1                                  # 24
_env0.close()

# ====== Helpers ======
def flatten_action(a0: int, a1: int) -> int:
    return a0 * N1 + a1

def unflatten_action(a_flat: int) -> tuple[int, int]:
    return int(a_flat // N1), int(a_flat % N1)

def _com_bins_from_grid(grid: np.ndarray) -> tuple[int, int, int]:
    gx, gy = grid.shape
    total = float(grid.sum())
    if total <= 0.0:
        cx = (gx - 1) / 2.0
        cy = (gy - 1) / 2.0
    else:
        xs = np.arange(gx, dtype=np.float32)[:, None]
        ys = np.arange(gy, dtype=np.float32)[None, :]
        cx = float((grid * xs).sum() / total)
        cy = float((grid * ys).sum() / total)
    # map COM to [0..COM_BINS-1]
    x_bin = int(np.clip(int(cx / max(1, gx) * COM_BINS), 0, COM_BINS - 1))
    y_bin = int(np.clip(int(cy / max(1, gy) * COM_BINS), 0, COM_BINS - 1))
    # density = peak car count in any 20m x 20m cell, clipped to [0..DENSITY_BINS-1]
    peak = int(grid.max()) if grid.size else 0
    dens = int(min(DENSITY_BINS - 1, peak))
    return x_bin, y_bin, dens

def base_state_key(obs: dict) -> int:
    """
    Compact, repeatable base state *without* bpos:
      - COM bins (x,y) in 4x4,
      - density bin (0..3),
      - current beam direction (0..7) and angle (0..2)
    This keeps the state space small and learnable; we add bpos via factorized Q.
    """
    grid = obs["grid"].astype(np.float32, copy=False)
    x_bin, y_bin, dens = _com_bins_from_grid(grid)
    bdir, bang = map(int, obs["beam"])
    # pack to a single int
    return int((((x_bin * COM_BINS + y_bin) * DENSITY_BINS + dens) * (N0 * N1)) + (bdir * N1 + bang))

def epsilon_greedy(q_vec: np.ndarray, epsilon: float) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, A))
    m = np.max(q_vec)
    idxs = np.flatnonzero(q_vec == m)
    return int(np.random.choice(idxs))

def save_q(out_dir: str, run_id: int, step: int,
           Q_shared: dict, seed_used: int):
    """Save factorized Q-tables as gzipped pickle."""
    path = os.path.join(out_dir, f"Q_factorized_run{run_id:02d}_step{step}.pkl.gz")
    blob = {
        "Q_shared": dict(Q_shared),
        "meta": {
            "run_id": run_id,
            "seed": seed_used,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "episodes": EPISODES,
            "COM_BINS": COM_BINS,
            "DENSITY_BINS": DENSITY_BINS,
            "N0": N0, "N1": N1
        },
    }
    with gzip.open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save] {path}")

def load_q(path: str) -> dict:
    """Load factorized Q-tables from gzipped pickle."""
    with gzip.open(path, "rb") as f:
        return pickle.load(f)["Q_shared"]


def run_episode_greedy(q_shared, pairIndex: int, render: bool = False, max_steps: int = 10_000) -> float:
    """Evaluates current Q policy greedily."""
    def q_total(s_base: int) -> np.ndarray:
        return q_shared[s_base]

    e = make_env(render=render)
    obs, _ = e.reset(pairIndex=pairIndex)
    s_base = base_state_key(obs)
    total_r = 0.0
    steps = 0
    done = False
    trunc = False
    while not (done or trunc):
        qv = q_total(s_base)
        a = int(np.argmax(qv))
        a0, a1 = unflatten_action(a)
        obs, r, done, trunc, _ = e.step((a0, a1))
        r = float(np.clip(r, -REWARD_CLIP, REWARD_CLIP))
        total_r += r
        s_base = base_state_key(obs)
        steps += 1
        if steps >= max_steps:
            break
    e.close()
    return float(total_r)


def q_total(Q_shared, s_base: int) -> np.ndarray:
    return Q_shared[s_base]

def train_one_run(run_id: int) -> dict:
    """Runs a full training and returns a summary dict (with early stopping)."""
    # --- Repro per run
    sb = make_seeds(SEED0)
    set_global_seeds(sb)

    # --- Per-run dirs
    run_save_dir = os.path.join(BASE_SAVE_DIR, f"run_{run_id:02d}")
    run_log_dir  = os.path.join(BASE_LOG_DIR,  f"run_{run_id:02d}")
    os.makedirs(run_save_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir) if SummaryWriter else None

    # --- TensorBoard writer
    if writer:
        writer.add_text("meta/run_info", json.dumps({
            "run_id": run_id,
            "seed": sb.__dict__,
            "episodes_planned": EPISODES,
            "discount": DISCOUNT,
            "epsilon0": EPSILON0,
            "epsilon_decay": EPSILON_DECAY,
            "epsilon_min": EPSILON_MIN,
            "alpha0_shared": ALPHA0_SHARED,
            "alpha_floor": ALPHA_FLOOR,
            "early_stop_window": EARLY_STOP_WINDOW,
            "early_stop_min_improvement": EARLY_STOP_MIN_IMPROVEMENT,
            "early_check_every": EARLY_CHECK_EVERY,
        }, indent=2), global_step=0)

    # --- Initialize tables
    Q_shared = defaultdict(lambda: np.zeros(A, dtype=np.float32))
    def _make_state_table():
        return defaultdict(lambda: np.zeros(A, dtype=np.float32))

    ep_rewards = []
    aggr_ep_rewards = {"ep": [], "avg": [], "min": [], "max": []}

    epsilon = EPSILON0
    alpha_shared = ALPHA0_SHARED

    env = make_env(render=False, seed=sb.env_seed)
    t_train_start = time.perf_counter()

    position = env.allPairs[run_id]

    start_time = datetime.now().isoformat() + "Z"

    early_stopped = False
    stop_episode = None

    episode_rewards = []

    for episode in range(EPISODES):
        render = (episode % SHOW_EVERY == 0 and SHOW)
        if render:
            env.close()
            env = make_env(render=True, seed=sb.env_seed)
        elif episode % SHOW_EVERY == 1 and not SHOW:
            env.close()
            env = make_env(render=False, seed=sb.env_seed)

        obs, info = env.reset(pairIndex=run_id, seed=sb.env_seed)
        s_base = base_state_key(obs)

        # schedules (gentle decay + floor)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        scale = 1.0 / np.sqrt(1.0 + episode / 1500.0)
        alpha_shared = max(ALPHA_FLOOR, ALPHA0_SHARED * scale)

        t0 = time.time()
        steps = 0
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            steps += 1
            qv = q_total(Q_shared, s_base)
            a = epsilon_greedy(qv, epsilon)
            a0, a1 = unflatten_action(a)

            next_obs, reward, terminated, truncated, info = env.step((a0, a1))
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

            s_next = base_state_key(next_obs)

            # target
            if terminated or truncated:
                best_next = 0.0
            else:
                best_next = float(np.max(q_total(Q_shared, s_next)))

            td_target = reward + DISCOUNT * best_next
            td_error  = td_target - float(qv[a])

            # updates split across shared + per-position
            Q_shared[s_base][a]     += alpha_shared * td_error

            episode_reward += reward
            s_base = s_next


        dt = time.time() - t0
        ep_rewards.append(episode_reward)
        print(f"[run {run_id:02d}] Episode {episode+1:5d} | steps {steps:5d} | time {dt:7.3f} | "
              f"reward {episode_reward:8.3f} | eps {epsilon:6.3f} | "
              f"alpha_shared {alpha_shared:5.3f}")

        if writer:
            episode_rewards.append((episode, episode_reward))
            writer.add_scalar("Train/episode_reward", episode_reward, episode)
            writer.add_scalar("Train/epsilon", epsilon, episode)
            writer.add_scalar("Train/alpha_shared", alpha_shared, episode)

        # aggregation + eval + checkpoint
        if (episode + 1) % SHOW_EVERY == 0:
            window = ep_rewards[-SHOW_EVERY:]
            avg_r = float(np.mean(window))
            min_r = float(np.min(window))
            max_r = float(np.max(window))
            aggr_ep_rewards["ep"].append(episode + 1)
            aggr_ep_rewards["avg"].append(avg_r)
            aggr_ep_rewards["min"].append(min_r)
            aggr_ep_rewards["max"].append(max_r)


            print(f"[run {run_id:02d}][stats] Episode {episode+1:5d} | avg {avg_r:8.3f} | "
                  f"min {min_r:8.3f} | max {max_r:8.3f}")

            save_q(run_save_dir, run_id, step=(episode + 1),
                   Q_shared=Q_shared, seed_used=sb.env_seed)

        # ---------- Early stopping check ----------
        # Compare mean over last W episodes vs previous W episodes
        if (episode + 1) % EARLY_CHECK_EVERY == 0 and len(ep_rewards) >= 2 * EARLY_STOP_WINDOW:
            prev_window_mean = float(np.mean(ep_rewards[-2*EARLY_STOP_WINDOW:-EARLY_STOP_WINDOW]))
            curr_window_mean = float(np.mean(ep_rewards[-EARLY_STOP_WINDOW:]))
            delta = abs(curr_window_mean - prev_window_mean)

            if writer:
                writer.add_scalar("EarlyStop/prev_window_mean", prev_window_mean, episode + 1)
                writer.add_scalar("EarlyStop/curr_window_mean", curr_window_mean, episode + 1)
                writer.add_scalar("EarlyStop/delta", delta, episode + 1)

            if delta < EARLY_STOP_MIN_IMPROVEMENT:
                early_stopped = True
                stop_episode = episode + 1
                print(f"[run {run_id:02d}] EARLY STOP at episode {stop_episode}: "
                      f"Δmean{EARLY_STOP_WINDOW}={delta:.4f} < {EARLY_STOP_MIN_IMPROVEMENT}")

                final_eval = run_episode_greedy(Q_shared, pairIndex=run_id, render=False)
                # Multi-seed evaluation
                eval_seeds = [make_seeds(SEED0 + i) for i in range(100)]
                counts = np.zeros(env.action_space.nvec[0] * env.action_space.nvec[1], dtype=np.int64)
                per_seed_returns = []
                for ep_i, sb_eval in enumerate(eval_seeds):
                    obs, info = env.reset(pairIndex=run_id, seed=sb_eval.eval_seed)
                    s_base = base_state_key(obs)

                    r = 0
                    terminated = False
                    truncated = False

                    while not (terminated or truncated):
                        qv = q_total(Q_shared, s_base)
                        a = epsilon_greedy(qv, 0.0)
                        a0, a1 = unflatten_action(a)

                        next_obs, reward, terminated, truncated, info = env.step((a0, a1))
                        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

                        counts[int(a)] += 1
                        r += reward
                        s_base = base_state_key(next_obs)

                    per_seed_returns.append(float(r))
                    print(f"Test episode {ep_i}")
                eval_mean = float(np.mean(per_seed_returns))
                eval_std = float(np.std(per_seed_returns))
                total_train_time_sec = time.perf_counter() - t_train_start
                episodes_trained = stop_episode

                order = np.argsort(counts)[::-1]
                total = int(counts.sum())
                top_k = 3
                top = []
                for rank, idx in enumerate(order[:top_k], start=1):
                    cnt = int(counts[idx]); freq = float(cnt / max(1, total))
                    dir_idx = int(idx // N1); ang_idx = int(idx % N1)
                    top.append({
                        "rank": rank,
                        "dir_idx": dir_idx, "dir_deg": DIR_DEG[dir_idx], "ang_idx": ang_idx, "ang_deg": ANGLE_DEG[ang_idx],
                        "count": cnt, "freq": freq
                    })
                # Per-run summary JSON
                run_summary = {
                    "run_id": run_id,
                    "seed": sb.__dict__,
                    "n_eval_seeds": len(eval_seeds),
                    "eval_mean_reward": eval_mean,
                    "eval_std_reward": eval_std,
                    "eval_returns_per_seed": per_seed_returns,
                    "episodes_planned": EPISODES,
                    "episodes_trained": episodes_trained,
                    "early_stopped": early_stopped,
                    "stop_episode": stop_episode,
                    "total_train_time_sec": total_train_time_sec,
                    "final_eval_reward": float(final_eval),
                    "last_window_avg_reward": float(aggr_ep_rewards["avg"][-1]) if aggr_ep_rewards["avg"] else None,
                    "checkpoints_every": SHOW_EVERY,
                    "save_dir": run_save_dir,
                    "position": [int(i) for i in position],
                    "best_actions": top,
                    "log_dir": run_log_dir,
                    "start_time": start_time,
                    "end_time": datetime.now().isoformat() + "Z",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                summary_path = os.path.join(run_save_dir, "run_summary.json")
                with open(summary_path, "w") as fjs:
                    json.dump(run_summary, fjs, indent=2)
                print(f"[run {run_id:02d}] wrote {summary_path}")
                return run_summary, episode_rewards
                break
        # -----------------------------------------

    total_train_time_sec = time.perf_counter() - t_train_start
    env.close()

    episodes_trained = stop_episode if early_stopped else EPISODES

    # final eval + save (save with actual trained length)
    final_eval = run_episode_greedy(Q_shared, pairIndex=run_id,  render=False)
    save_q(run_save_dir, run_id, step=episodes_trained,
           Q_shared=Q_shared, seed_used=sb.env_seed)

    # Save aggregated rewards CSV
    aggr_csv_path = os.path.join(run_save_dir, "aggr_ep_rewards.csv")
    with open(aggr_csv_path, "w", newline="") as fcsv:
        w = csv_mod.writer(fcsv)
        w.writerow(["episode", "avg", "min", "max"])
        for ep, avg, mn, mx in zip(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"],
                                   aggr_ep_rewards["min"], aggr_ep_rewards["max"]):
            w.writerow([ep, avg, mn, mx])
    print(f"[run {run_id:02d}] wrote {aggr_csv_path}")

    # Save training curve PNG
    plt.figure()
    plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
    plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
    plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")
    plt.legend(loc="upper left")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title(f"Run {run_id:02d} - Factorized Q-learning (early stop={early_stopped})")
    png_path = os.path.join(run_save_dir, "training_curve.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[run {run_id:02d}] wrote {png_path}")

    # Multi-seed evaluation: deterministic but distinct seed per test episode
    eval_seeds = [make_seeds(SEED0 + i) for i in range(100)]
    counts = np.zeros(env.action_space.nvec[0] * env.action_space.nvec[1], dtype=np.int64)
    per_seed_returns = []
    for ep_i, sb_eval in enumerate(eval_seeds):
        obs, info = env.reset(pairIndex=run_id, seed=sb_eval.eval_seed)
        s_base = base_state_key(obs)

        r = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            qv = q_total(Q_shared, s_base)
            a = epsilon_greedy(qv, 0.0)
            a0, a1 = unflatten_action(a)

            next_obs, reward, terminated, truncated, info = env.step((a0, a1))
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

            counts[int(a)] += 1
            r += reward
            s_base = base_state_key(next_obs)

        per_seed_returns.append(float(r))
        print(f"Test episode {ep_i}")
    eval_mean = float(np.mean(per_seed_returns))
    eval_std = float(np.std(per_seed_returns))
    print(f"[run {run_id:02d}] eval over {len(eval_seeds)} seeds: "
          f"mean={eval_mean:.3f} std={eval_std:.3f}")

    order = np.argsort(counts)[::-1]
    total = int(counts.sum())
    top_k = 3
    top = []
    for rank, idx in enumerate(order[:top_k], start=1):
        cnt = int(counts[idx]); freq = float(cnt / max(1, total))
        dir_idx = int(idx // N1); ang_idx = int(idx % N1)
        top.append({
            "rank": rank,
            "dir_idx": dir_idx, "dir_deg": DIR_DEG[dir_idx], "ang_idx": ang_idx, "ang_deg": ANGLE_DEG[ang_idx],
            "count": cnt, "freq": freq
        })

    # Per-run summary JSON
    run_summary = {
        "run_id": run_id,
        "seed": sb.__dict__,
        "episodes_planned": EPISODES,
        "episodes_trained": episodes_trained,
        "early_stopped": early_stopped,
        "stop_episode": stop_episode,
        "total_train_time_sec": total_train_time_sec,
        "final_eval_reward": float(final_eval),
        "last_window_avg_reward": float(aggr_ep_rewards["avg"][-1]) if aggr_ep_rewards["avg"] else 0,
        "checkpoints_every": SHOW_EVERY,
        "save_dir": run_save_dir,
        "best_actions": top,
        "log_dir": run_log_dir,
        "position": [int(i) for i in position],
        "start_time": start_time,
        "end_time": datetime.now().isoformat() + "Z",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    summary_path = os.path.join(run_save_dir, "run_summary.json")
    with open(summary_path, "w") as fjs:
        json.dump(run_summary, fjs, indent=2)
    print(f"[run {run_id:02d}] wrote {summary_path}")

    if writer:
        writer.add_scalar("Run/total_train_time_sec", total_train_time_sec, global_step=episodes_trained)
        writer.add_scalar("Run/final_eval_reward", final_eval, global_step=episodes_trained)
        writer.add_scalar("Run/episodes_trained", episodes_trained, global_step=episodes_trained)
        writer.flush()
        writer.close()

    return run_summary, episode_rewards

# ====== Run 75 trainings (IDs 00..75) and aggregate ======
if __name__ == "__main__":
    all_summaries = []
    all_episode_rewards = []
    t_all_start = time.perf_counter()
    all_start_times = datetime.now().isoformat() + "Z"

    last_run_id = 0
    for root, dirs, files in os.walk(BASE_SAVE_DIR):
        for dir in dirs:
            if dir.startswith("run_"):
                run_id = int(dir.split("_")[-1])
                if not os.path.exists(os.path.join(root+"/"+dir, "run_summary.json")):
                    last_run_id = run_id
                else:
                    last_run_id = run_id+1
                    
                    
    print("Last run id: ", last_run_id)

    for run_id in range(last_run_id, RUNS):  # 0..75 exlusive
        print(f"\n======================")
        print(f"Starting training run {run_id:02d}")
        print(f"======================\n")
        summary, episode_rewards = train_one_run(run_id)
        all_summaries.append(summary)
        all_episode_rewards.append( episode_rewards)
        run_save_dir = os.path.join(BASE_SAVE_DIR, f"run_{run_id:02d}")
        episode_rewards_path = os.path.join(run_save_dir, "runs.json")
        with open(episode_rewards_path, "w") as fjs:
            json.dump(episode_rewards, fjs, indent=2)
    
    total_wall_time_sec = time.perf_counter() - t_all_start
    print(f"\nAll {RUNS} runs completed in {total_wall_time_sec:.2f} seconds.")


    # Save global summaries
    global_csv = os.path.join(BASE_SAVE_DIR, "all_runs_summary.csv")
    with open(global_csv, "w", newline="") as fcsv:
        w = csv_mod.DictWriter(
            fcsv,
            fieldnames=[
                "run_id","seed",
                "episodes_planned","episodes_trained",
                "early_stopped","stop_episode",
                "total_train_time_sec",
                "final_eval_reward","last_window_avg_reward",
                "checkpoints_every","save_dir","log_dir","timestamp","best_actions","position","start_time","end_time",
            ]
        )
        w.writeheader()
        for s in all_summaries:
            w.writerow(s)
    print(f"[global] wrote {global_csv}")

    global_json = os.path.join(BASE_SAVE_DIR, "all_runs_summary.json")
    with open(global_json, "w") as fjs:
        json.dump({
            "total_runs": len(all_summaries),
            "total_wall_time_sec": total_wall_time_sec,
            "start_time": all_start_times,
            "end_time": datetime.now().isoformat() + "Z",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "runs": all_summaries
        }, fjs, indent=2)
    print(f"[global] wrote {global_json}")

    # ====== Global TensorBoard summary (Top-K + Best) ======
    if SummaryWriter:
        GLOBAL_LOG_DIR = os.path.join(BASE_LOG_DIR, "_global")  # separate TB run for aggregates
        
        # 1) Split all runs into 3 segments and find the best in each segment
        total_runs = len(all_summaries)
        segment_size = total_runs // 3
        remainder = total_runs % 3
        
        # Calculate segment boundaries
        segments = []
        start_idx = 0
        for i in range(3):
            # Distribute remainder across first segments
            current_size = segment_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_size
            segments.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Find best run in each segment
        for seg_idx, (start, end) in enumerate(segments):
            segment_runs = all_summaries[start:end]
            if segment_runs:
                best_in_segment = max(segment_runs, key=lambda s: s["last_window_avg_reward"])
                # Log the best run from this segment
                best_run_id = best_in_segment["run_id"]
                best_episode_rewards = all_episode_rewards[best_run_id]
                global_writer = SummaryWriter(GLOBAL_LOG_DIR+"/Best_Segment_"+str(seg_idx)+"_Run_"+str(best_run_id))
                for episode, reward in best_episode_rewards:
                    global_writer.add_scalar(
                        f"Global/Segments/Best/episode_reward",
                        reward,
                        global_step=episode,
                    )
        global_writer.flush()
        global_writer.close()


    ##tensorboard --logdir=runs --port 6006