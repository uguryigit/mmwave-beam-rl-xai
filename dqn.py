import os
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from layout import RoundaboutSegmentedEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

# ---------------- Wrappers you already have ----------------
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

# ---------------- Training with logging & plotting ----------------
if __name__ == "__main__":
    LOG_DIR = "logs/dqn_roundabout"
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1) Build base envs
    base_train = RoundaboutSegmentedEnv(render_mode=None)
    base_eval  = RoundaboutSegmentedEnv(render_mode=None)

    # 2) Monitor to capture episode returns/lengths
    train_env = Monitor(base_train, filename=os.path.join(LOG_DIR, "monitor_train.csv"))
    eval_env  = Monitor(base_eval,  filename=os.path.join(LOG_DIR, "monitor_eval.csv"))

    # 3) Your wrappers (same order on both train & eval envs)
    train_env = DictObsToVector(train_env, max_count=20)
    train_env = FlattenMultiDiscreteAction(train_env)

    eval_env = DictObsToVector(eval_env, max_count=20)
    eval_env = FlattenMultiDiscreteAction(eval_env)

    # 4) Evaluation callback (also writes evaluations.npz)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # 5) Model + CSV & TensorBoard logger
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
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
        tensorboard_log=LOG_DIR,  # TB logs
    )
    # Ensure a CSV "progress.csv" is written into LOG_DIR
    model.set_logger(configure(LOG_DIR, ["stdout", "csv", "tensorboard"]))

    # 6) Train
    model.learn(total_timesteps=300_000, callback=eval_callback)
    model.save(os.path.join(LOG_DIR, "dqn_roundabout_mlp"))
    train_env.close()
    eval_env.close()

    # 7) Plot & save PNGs
    #    - progress.csv: SB3 training stats (includes rollout/ep_rew_mean if Monitor present)
    #    - evaluations.npz: results from EvalCallback
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    prog_csv = os.path.join(LOG_DIR, "progress.csv")
    if os.path.exists(prog_csv):
        df = pd.read_csv(prog_csv)
        # Safely handle missing columns
        x = df.get("time/total_timesteps", pd.Series(range(len(df))))
        y = df.get("rollout/ep_rew_mean", None)
        if y is not None:
            y_smooth = y.rolling(10, min_periods=1).mean()
            plt.figure()
            plt.plot(x, y_smooth, label="Mean episode reward (rolling=10)")
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.title("DQN Training — Episode Reward")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, "reward_curve.png"), dpi=200)
            plt.close()

        # Optionally: plot TD error / loss if present
        if "train/loss" in df.columns:
            plt.figure()
            plt.plot(x, df["train/loss"].rolling(50, min_periods=1).mean())
            plt.xlabel("Timesteps")
            plt.ylabel("Loss (rolling=50)")
            plt.title("DQN Training — Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, "loss_curve.png"), dpi=200)
            plt.close()

    # Plot evaluation scores (from EvalCallback)
    eval_npz = os.path.join(LOG_DIR, "evaluations.npz")
    if os.path.exists(eval_npz):
        data = np.load(eval_npz, allow_pickle=True)
        t = data["timesteps"]
        results = data["results"]  # shape: (n_evals, n_eval_episodes)
        mean_rewards = results.mean(axis=1)
        plt.figure()
        plt.plot(t, mean_rewards, marker="o")
        plt.xlabel("Timesteps")
        plt.ylabel("Eval mean reward")
        plt.title("Evaluation Performance")
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, "eval_curve.png"), dpi=200)
        plt.close()

    print(f"Saved graphs to: {LOG_DIR}")
