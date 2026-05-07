import os
import time
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

from layout import RoundaboutSegmentedEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
# Optional global TensorBoard writer (requires PyTorch)
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None

from seed import make_seeds, set_global_seeds


# =================== Config ===================
RUNS = 75                       # run_00 .. run_74
SEED0 = 42
TOTAL_TIMESTEPS = 300_000            # per run

LOG_ROOT = "logs/reinforce/roundabout_dqn"    # per-run logs, TB, CSVs, evals, summaries
os.makedirs(LOG_ROOT, exist_ok=True)

# Early stopping on episodic reward
EARLY_WINDOW = 400                   # last 400 vs previous 400 episodes
EARLY_MIN_IMPROVE = 0.01             # absolute delta threshold
EARLY_CHECK_EVERY_EP = 50            # check every N episodes (once enough history)

ANGLE_DEG = [65, 90, 120]
DIR_DEG = [0, 45, 90, 135, 180, 225, 270, 315]


# DQN hyperparams
DQN_KW = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=1_000,
    batch_size=256,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    exploration_fraction=0.2,
    verbose=1,
)

# =================== Wrappers ===================
class PairIndexResetWrapper(gym.Wrapper):
    """
    Bridges a custom env.reset(pairIndex) to Gymnasium's reset(seed=None, options=None).
    Chooses pairIndex from:
      1) options['pairIndex'] if provided,
      2) self.default_pair if set,
      3) random in [0, n_positions-1].
    Place this wrapper *directly* above the base env so it can call base.reset(pairIndex).
    """
    def __init__(self, env, n_positions: int = 75, default_pair: int | None = None):
        super().__init__(env)
        self.n_positions = int(n_positions)
        self.default_pair = default_pair

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # optional: propagate seed to underlying env if it supports seeding
        try:
            if seed is not None and hasattr(self.env, "seed"):
                self.env.seed(seed)  # legacy API; ignore if unavailable
        except Exception:
            pass

        if options is None:
            options = {}
        pair_idx = options.get("pairIndex", self.default_pair)
        if pair_idx is None:
            pair_idx = int(np.random.randint(self.n_positions))

        # Call the base env's custom signature (positional argument)
        # IMPORTANT: This wrapper must sit directly on top of the base env.
        obs, info = self.env.reset(pair_idx)
        return obs, info

class DictObsToVector(gym.ObservationWrapper):
    """Dict(grid=int32, beam=MultiDiscrete[8,3]) → Box(vector)."""
    def __init__(self, env, max_count: int = 20):
        super().__init__(env)
        self.max_count = max_count
        gx, gy = env.observation_space["grid"].shape
        self.grid_dim = gx * gy
        self.vec_dim  = self.grid_dim + 8 + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.vec_dim,), dtype=np.float32
        )

    @staticmethod
    def _one_hot(n, idx):
        v = np.zeros((n,), dtype=np.float32)
        v[int(idx)] = 1.0
        return v

    def observation(self, obs: dict):
        grid = obs["grid"].astype(np.float32) / float(self.max_count)
        grid_flat = grid.reshape(-1)
        direc, ang = map(int, obs["beam"])
        vec = np.concatenate([
            grid_flat,
            self._one_hot(8,  direc),
            self._one_hot(3,  ang),
        ], axis=0).astype(np.float32)
        return vec

class FlattenMultiDiscreteAction(gym.ActionWrapper):
    """Wrap MultiDiscrete([8,3]) → Discrete(24) for DQN."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self.n0, self.n1 = map(int, env.action_space.nvec)  # 8, 3
        self.action_space = spaces.Discrete(self.n0 * self.n1)

    def action(self, act_flat: int):
        a0 = int(act_flat // self.n1)
        a1 = int(act_flat %  self.n1)
        return (a0, a1)

# =================== Early Stop ===================
class EarlyStopCallback(BaseCallback):
    """
    Stop training if the rolling mean of the last W episodes fails to improve
    over the previous W by at least min_delta. Checked every `check_every` episodes.
    Requires Monitor wrapper so infos contain 'episode'.
    """
    def __init__(self, window=400, min_delta=0.01, check_every=50, verbose=0, episode_rewards=[]):
        super().__init__(verbose)
        self.window = int(window)
        self.min_delta = float(min_delta)
        self.check_every = int(check_every)
        self.ep_rewards = []
        self.episodes_seen = 0
        self.last_check_ep = 0
        self.triggered = False
        self.stop_step = None
        self.prev_mean = None
        self.curr_mean = None
        self.delta = None
        self.episode_rewards = episode_rewards

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode" in info:
                r = float(info["episode"]["r"])
                self.episode_rewards.append((self.episodes_seen, r))
                self.ep_rewards.append(r)
                self.episodes_seen += 1

                if self.episodes_seen >= 2 * self.window:
                    if (self.episodes_seen - self.last_check_ep) >= self.check_every:
                        prev = float(np.mean(self.ep_rewards[-2*self.window:-self.window]))
                        curr = float(np.mean(self.ep_rewards[-self.window:]))
                        delta = curr - prev
                        self.prev_mean = prev
                        self.curr_mean = curr
                        self.delta = delta
                        self.last_check_ep = self.episodes_seen

                        self.logger.record("early_stop/prev_window_mean", prev)
                        self.logger.record("early_stop/curr_window_mean", curr)
                        self.logger.record("early_stop/delta", delta)
                        self.logger.dump(self.model.num_timesteps)

                        if delta < self.min_delta:
                            self.triggered = True
                            self.stop_step = int(self.model.num_timesteps)
                            if self.verbose:
                                print(f"[EarlyStop] Stop at step={self.stop_step} "
                                      f"(episodes={self.episodes_seen}): "
                                      f"Δ={delta:.4f} < {self.min_delta}")
                            return False  # stop training
        return True


from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import json
import os




# =================== Utils ===================
def make_wrapped_env(render_mode=None, n_positions: int = 75, default_pair: int | None = None, seed: int | None = None):
    """
    ORDER matters:
      RoundaboutSegmentedEnv
        -> PairIndexResetWrapper  (consumes Gym reset, calls base.reset(pairIndex))
        -> DictObsToVector
        -> FlattenMultiDiscreteAction
        -> Monitor (outermost; logs episode stats)
    """
    base = RoundaboutSegmentedEnv(render_mode=render_mode, seed=seed)
    env = PairIndexResetWrapper(base, n_positions=n_positions, default_pair=default_pair)
    env = DictObsToVector(env, max_count=20)
    env = FlattenMultiDiscreteAction(env)
    env = Monitor(env)  # must be outermost for episodic stats
    return env, base.allPairs[default_pair]

def save_plots(run_dir: str):
    prog_csv = os.path.join(run_dir, "progress.csv")
    if os.path.exists(prog_csv):
        df = pd.read_csv(prog_csv)
        x = df.get("time/total_timesteps", pd.Series(range(len(df))))
        y = df.get("rollout/ep_rew_mean", None)
        if y is not None:
            y_smooth = y.rolling(10, min_periods=1).mean()
            plt.figure()
            plt.plot(x, y_smooth, label="Mean episode reward (rolling=10)")
            plt.xlabel("Timesteps"); plt.ylabel("Reward")
            plt.title("DQN Training — Episode Reward")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "reward_curve.png"), dpi=200)
            plt.close()
        if "train/loss" in df.columns:
            plt.figure()
            plt.plot(x, df["train/loss"].rolling(50, min_periods=1).mean())
            plt.xlabel("Timesteps"); plt.ylabel("Loss (rolling=50)")
            plt.title("DQN Training — Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=200)
            plt.close()

    eval_npz = os.path.join(run_dir, "evaluations.npz")
    if os.path.exists(eval_npz):
        data = np.load(eval_npz, allow_pickle=True)
        t = data["timesteps"]
        results = data["results"]  # (n_evals, n_eval_episodes)
        mean_rewards = results.mean(axis=1)
        plt.figure()
        plt.plot(t, mean_rewards, marker="o")
        plt.xlabel("Timesteps"); plt.ylabel("Eval mean reward")
        plt.title("Evaluation Performance")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "eval_curve.png"), dpi=200)
        plt.close()

def maybe_save_weights_pth(model, path_pth: str):
    try:
        import torch
        state = {
            "q_net": model.policy.q_net.state_dict(),
            "q_net_target": model.policy.q_net_target.state_dict(),
            "policy_kwargs": model.policy_kwargs if hasattr(model, "policy_kwargs") else {},
            "date": datetime.utcnow().isoformat() + "Z",
        }
        torch.save(state, path_pth)
        return True
    except Exception as e:
        print(f"[warn] Could not export .pth weights: {e}")
        return False

# =================== One run ===================
def run_one(run_id: int) -> dict:
    run_dir = os.path.join(LOG_ROOT, f"run_{run_id:02d}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    episode_rewards = []

    sb = make_seeds(SEED0)
    set_global_seeds(sb)

    # Build envs (PairIndexResetWrapper fixes the reset signature)
    train_env, position = make_wrapped_env(render_mode=None, default_pair=run_id, seed=sb.env_seed)
    eval_env, _  = make_wrapped_env(render_mode=None, default_pair=run_id, seed=sb.eval_seed)


    # Seed via Gymnasium API if supported by your env chain (optional)
    try:
        train_env.reset(seed=sb.env_seed)
        eval_env.reset(seed=sb.eval_seed)
    except TypeError:
        pass

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=TOTAL_TIMESTEPS,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=0,
    )

    early_cb = EarlyStopCallback(
        window=EARLY_WINDOW,
        min_delta=EARLY_MIN_IMPROVE,
        check_every=EARLY_CHECK_EVERY_EP,
        verbose=1,
        episode_rewards=episode_rewards,
    )

    model = DQN(env=train_env,seed=sb.torch_seed, tensorboard_log=run_dir, **DQN_KW)
    model.set_logger(configure(run_dir, ["stdout", "csv", "tensorboard"]))

    t0 = time.perf_counter()
    start_time = datetime.now().isoformat() + "Z"
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([eval_callback, early_cb]))
    wall_sec = time.perf_counter() - t0

    model.logger.record("time/run_wall_time_sec", wall_sec)
    model.logger.dump(model.num_timesteps)

    final_zip = os.path.join(run_dir, f"dqn_roundabout_mlp_run{run_id:02d}")
    model.save(final_zip)
    saved_pth = maybe_save_weights_pth(model, os.path.join(run_dir, "policy_weights.pth"))

    print(f"Testing model {run_id:02d}")
    # Multi-seed evaluation: each test episode uses a deterministically-derived
    # different seed so that action statistics are aggregated over distinct
    # stochastic environment realisations (vehicle spawns, shadow fading, etc.).
    eval_seeds = [make_seeds(SEED0 + i) for i in range(100)]
    counts = np.zeros(train_env.action_space.n, dtype=np.int64)
    per_seed_returns = []
    for sb_eval in eval_seeds:
        obs, _ = train_env.reset(seed=sb_eval.eval_seed)
        done = False; trunc = False
        r = 0
        while not (done or trunc):
            a, _ = model.predict(obs, deterministic=True)  # greedy action
            counts[int(a)] += 1
            obs, rew, done, trunc, info = train_env.step(int(a))
            r += rew
        per_seed_returns.append(float(r))
    eval_mean = float(np.mean(per_seed_returns))
    eval_std = float(np.std(per_seed_returns))
    print(f"Tested model {run_id:02d} | eval over {len(eval_seeds)} seeds: "
          f"mean={eval_mean:.3f} std={eval_std:.3f}")

    order = np.argsort(counts)[::-1]
    total = int(counts.sum())
    top_k = 3
    top = []
    for rank, idx in enumerate(order[:top_k], start=1):
        cnt = int(counts[idx]); freq = float(cnt / max(1, total))
        dir_idx = int(idx // 3); ang_idx = int(idx % 3)
        top.append({
            "rank": rank, 
            "dir_idx": dir_idx, "dir_deg": DIR_DEG[dir_idx], "ang_idx": ang_idx, "ang_deg": ANGLE_DEG[ang_idx],
            "count": cnt, "freq": freq
        })

    train_env.close(); eval_env.close()

    save_plots(run_dir)

    # Eval summary (last eval mean if available)
    eval_npz = os.path.join(run_dir, "evaluations.npz")
    final_eval_mean = None
    if os.path.exists(eval_npz):
        data = np.load(eval_npz, allow_pickle=True)
        results = data["results"]
        if results.size > 0:
            final_eval_mean = float(results.mean(axis=1)[-1])

    summary = {
        "run_id": run_id,
        "seed": sb.__dict__,
        "n_eval_seeds": len(eval_seeds),
        "eval_mean_reward": eval_mean,
        "eval_std_reward": eval_std,
        "eval_returns_per_seed": per_seed_returns,
        "episodes_window": EARLY_WINDOW,
        "early_stop_min_improvement": EARLY_MIN_IMPROVE,
        "early_stop_check_every_ep": EARLY_CHECK_EVERY_EP,
        "episodes_trained": int(early_cb.episodes_seen),
        "early_stopped": bool(early_cb.triggered),
        "early_stop_step": int(early_cb.stop_step) if early_cb.stop_step is not None else None,
        "early_prev_mean": float(early_cb.prev_mean) if early_cb.prev_mean is not None else None,
        "early_curr_mean": float(early_cb.curr_mean) if early_cb.curr_mean is not None else None,
        "early_delta": float(early_cb.delta) if early_cb.delta is not None else None,
        "timesteps_trained": int(model.num_timesteps),
        "train_wall_time_sec": float(wall_sec),
        "final_eval_mean_reward": final_eval_mean,
        "run_dir": run_dir,
        "best_actions": top,
        "final_model_path": final_zip + ".zip",
        "weights_pth_saved": saved_pth,
        "start_time": t0,
        "position": [int(i) for i in position],
        "start_time": start_time,
        "end_time": datetime.now().isoformat() + "Z",
        "timestamp": datetime.now().isoformat() + "Z",
    }

    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    summary_cols = list(summary.keys())
    import csv
    with open(os.path.join(run_dir, "run_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader(); w.writerow(summary)

    print(f"[run {run_id:02d}] Done. Wall time: {wall_sec:.2f}s, "
          f"timesteps: {model.num_timesteps}, episodes: {early_cb.episodes_seen}, "
          f"early_stop: {early_cb.triggered}")

    return summary, episode_rewards

# =================== Main: 75 runs + global summary ===================
if __name__ == "__main__":
    all_summaries = []
    all_episode_rewards = []
    t_all0 = time.perf_counter()

    last_run_id = 0

    for root, dirs, files in os.walk(LOG_ROOT):
        for dir in sorted(dirs):
            if dir.startswith("run_"):
                run_id = int(dir.split("_")[-1])
                if not os.path.exists(os.path.join(root+"/"+dir, "run_summary.json")):
                    last_run_id = run_id
                else:
                    last_run_id = run_id+1
                    
                    
    print("Last run id: ", last_run_id)
    for rid in range(last_run_id, RUNS):
        print("\n======================")
        print(f"Starting training run {rid:02d}")
        print("======================\n")
        s, episode_rewards = run_one(rid)
        all_episode_rewards.append(episode_rewards)
        run_save_dir = os.path.join(LOG_ROOT, f"run_{rid:02d}")
        episode_rewards_path = os.path.join(run_save_dir, "runs.json")
        with open(episode_rewards_path, "w") as fjs:
            json.dump(episode_rewards, fjs, indent=2)
        all_summaries.append(s)

    total_wall = time.perf_counter() - t_all0
    print(f"\nAll {RUNS} runs completed in {total_wall:.2f} seconds.\n")

    df = pd.DataFrame(all_summaries)
    global_csv = os.path.join(LOG_ROOT, "all_runs_summary.csv")
    df.to_csv(global_csv, index=False)

    global_json = os.path.join(LOG_ROOT, "all_runs_summary.json")
    with open(global_json, "w") as f:
        json.dump({
            "total_runs": RUNS,
            "total_wall_time_sec": total_wall,
            "runs": all_summaries
        }, f, indent=2)

    print(f"[global] wrote {global_csv}")
    print(f"[global] wrote {global_json}")

        # ====== Global TensorBoard summary: best & top-K runs ======
    if SummaryWriter:
        GLOBAL_LOG_DIR = os.path.join(LOG_ROOT, "_global")  # separate aggregate run

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
                best_in_segment = max(segment_runs, key=lambda s: s["final_eval_mean_reward"])
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


        global_writer.flush()
        global_writer.close()
    else:
        print("[global] torch.utils.tensorboard.SummaryWriter not available; skipping global TB logging.")

