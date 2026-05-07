import os
import gzip
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from layout import RoundaboutSegmentedEnv

# ====== TensorBoard (optional) ======
LOG_DIR = "runs/roundabout_q_factorized_10s"
SummaryWriter = None
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None
writer = SummaryWriter(LOG_DIR) if SummaryWriter else None

# ====== Hyperparameters ======
SEED            = 42
DISCOUNT        = 0.97
EPISODES        = 12_000
SHOW_EVERY      = 500
REWARD_CLIP     = 1.0

EPSILON0        = 0.90
EPSILON_DECAY   = 0.9994
EPSILON_MIN     = 0.02

ALPHA0_SHARED   = 0.10
ALPHA0_POS      = 0.20
ALPHA_FLOOR     = 0.02  # floor for late learning

SAVE_DIR        = "episodes_q_factorized_10s"
os.makedirs(SAVE_DIR, exist_ok=True)

# Featurization bins (for the base 's' that is shared across positions)
COM_BINS        = 4      # 4x4 center-of-mass grid
DENSITY_BINS    = 4      # 0..3
DENSITY_BIN_SZ  = 8      # cars per bin (tune to your env)

# ====== Repro ======
np.random.seed(SEED)

# ====== Env ======
def make_env(render: bool):
    return RoundaboutSegmentedEnv(render_mode="human" if render else None)

env = make_env(render=False)
assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
N0, N1 = map(int, env.action_space.nvec)   # (8,3)
A = N0 * N1                                 # 24

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
    dens = int(min(DENSITY_BINS - 1, int(total // DENSITY_BIN_SZ)))
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

# ====== Factorized Q ======
# Q_shared: base-state → action-values
Q_shared = defaultdict(lambda: np.zeros(A, dtype=np.float32))

# Q_pos: per-bpos → (base-state → action-values)
def _make_state_table():
    return defaultdict(lambda: np.zeros(A, dtype=np.float32))
Q_pos: dict[int, dict[int, np.ndarray]] = defaultdict(_make_state_table)

def q_total(bpos: int, s_base: int) -> np.ndarray:
    return Q_shared[s_base] + Q_pos[bpos][s_base]

def save_q(step: int):
    path = os.path.join(SAVE_DIR, f"Q_factorized_{step}.pkl.gz")
    blob = {
        "Q_shared": dict(Q_shared),
        "Q_pos": {bp: dict(tbl) for bp, tbl in Q_pos.items()},
        "meta": {
            "COM_BINS": COM_BINS,
            "DENSITY_BINS": DENSITY_BINS,
            "N0": N0, "N1": N1
        },
    }
    with gzip.open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save] {path}")

def load_q(step: int):
    global Q_shared, Q_pos
    path = os.path.join(SAVE_DIR, f"Q_factorized_{step}.pkl.gz")
    with gzip.open(path, "rb") as f:
        blob = pickle.load(f)
    Q_shared = defaultdict(lambda: np.zeros(A, dtype=np.float32),
                           blob["Q_shared"])
    # rebuild nested defaultdict
    Q_pos = defaultdict(_make_state_table)
    for bp, tbl in blob["Q_pos"].items():
        inner = defaultdict(lambda: np.zeros(A, dtype=np.float32), tbl)
        Q_pos[int(bp)] = inner
    print(f"[load] {path}")

def run_episode_greedy(render: bool = False, max_steps: int = 10_000) -> float:
    e = make_env(render=render)
    obs, _ = e.reset()
    s_base = base_state_key(obs)
    bpos = int(obs["beam"][0])
    total_r = 0.0
    steps = 0
    done = False
    trunc = False
    while not (done or trunc):
        qv = q_total(bpos, s_base)
        a = int(np.argmax(qv))
        a0, a1 = unflatten_action(a)
        obs, r, done, trunc, _ = e.step((a0, a1))
        r = float(np.clip(r, -REWARD_CLIP, REWARD_CLIP))
        total_r += r
        s_base = base_state_key(obs)
        bpos = int(obs["beam"][0])
        steps += 1
        if steps >= max_steps:
            break
    e.close()
    return float(total_r)

# ====== Training ======
ep_rewards = []
aggr_ep_rewards = {"ep": [], "avg": [], "min": [], "max": []}

epsilon = EPSILON0
alpha_shared = ALPHA0_SHARED
alpha_pos = ALPHA0_POS

for episode in range(EPISODES):
    render = (episode % SHOW_EVERY == 0)
    if render:
        env.close()
        env = make_env(render=True)
    elif episode % SHOW_EVERY == 1:
        env.close()
        env = make_env(render=False)

    obs, info = env.reset()
    s_base = base_state_key(obs)
    bpos = int(obs["beam"][0])

    # schedules (gentle decay + floor)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    scale = 1.0 / np.sqrt(1.0 + episode / 1500.0)
    alpha_shared = max(ALPHA_FLOOR, ALPHA0_SHARED * scale)
    alpha_pos    = max(ALPHA_FLOOR, ALPHA0_POS    * scale)

    t0 = time.time()
    steps = 0
    episode_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        steps += 1
        qv = q_total(bpos, s_base)
        a = epsilon_greedy(qv, epsilon)
        a0, a1 = unflatten_action(a)

        if steps >= 100:
            next_obs, reward, terminated, truncated, info = env.step((a0, a1))
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

            s_next = base_state_key(next_obs)
            bpos_next = int(next_obs["beam"][0])

            # target
            if terminated or truncated:
                best_next = 0.0
            else:
                best_next = float(np.max(q_total(bpos_next, s_next)))

            td_target = reward + DISCOUNT * best_next
            td_error  = td_target - float(qv[a])

            # updates split across shared + per-position
            Q_shared[s_base][a]     += alpha_shared * td_error
            Q_pos[bpos][s_base][a]  += alpha_pos    * td_error

            episode_reward += reward
            s_base, bpos = s_next, bpos_next

            if writer:
                writer.add_scalar("Train/td_error", td_error, episode)

    dt = time.time() - t0
    ep_rewards.append(episode_reward)
    print(f"Episode {episode+1:5d} | steps {steps:5d} | time {dt:7.3f} | "
          f"reward {episode_reward:8.3f} | eps {epsilon:6.3f} | "
          f"alpha_shared {alpha_shared:5.3f} | alpha_pos {alpha_pos:5.3f}")

    if writer:
        writer.add_scalar("Train/episode_reward", episode_reward, episode)
        writer.add_scalar("Train/epsilon", epsilon, episode)
        writer.add_scalar("Train/alpha_shared", alpha_shared, episode)
        writer.add_scalar("Train/alpha_pos", alpha_pos, episode)

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

        eval_r = run_episode_greedy(render=False)
        if writer:
            writer.add_scalar("Eval/greedy_reward", eval_r, episode + 1)

        print(f"[stats] Episode {episode+1:5d} | avg {avg_r:8.3f} | "
              f"min {min_r:8.3f} | max {max_r:8.3f} | eval {eval_r:8.3f}")
        save_q(episode + 1)

if writer:
    writer.close()

# ====== Plot ======
plt.figure()
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")
plt.legend(loc="upper left")
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("Factorized tabular Q-learning with bpos specialization")
plt.show()
