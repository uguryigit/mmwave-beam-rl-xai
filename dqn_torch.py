# dqn_torch_roundabout_tb.py
import os, math, random, time, datetime, json
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from torch.utils.tensorboard import SummaryWriter

# ---- bring your env & wrappers ----
from layout import RoundaboutSegmentedEnv

class DictObsToVector(gym.ObservationWrapper):
    """Dict(grid=int32, beam=MultiDiscrete[75,8,3]) → Box(vector)."""
    def __init__(self, env, max_count: int = 20):
        super().__init__(env)
        self.max_count = max_count
        gx, gy = env.observation_space["grid"].shape
        self.grid_dim = gx * gy
        self.vec_dim  = self.grid_dim + 75 + 8 + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.vec_dim,), dtype=np.float32
        )

    @staticmethod
    def _one_hot(n, idx):
        v = np.zeros((n,), dtype=np.float32)
        v[int(idx)] = 1.0
        return v

    def observation(self, obs: dict):
        grid = obs["grid"].astype(np.float32) / float(self.max_count)  # [0,1]
        grid_flat = grid.reshape(-1)
        pos, direc, ang = map(int, obs["beam"])
        vec = np.concatenate([
            grid_flat,
            self._one_hot(75, pos),
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

# -------------------- Torch DQN --------------------

def fanin_init(layer: nn.Linear):
    bound = 1.0 / math.sqrt(layer.weight.size(0))
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)

class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(512, 512)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], act_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fanin_init(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int, device: torch.device, dtype=torch.float32):
        self.device = device
        self.capacity = capacity
        self.ptr = 0
        self.full = False
        self.s  = torch.zeros((capacity, obs_dim), dtype=dtype, device=device)
        self.a  = torch.zeros((capacity, 1), dtype=torch.long, device=device)
        self.r  = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.ns = torch.zeros((capacity, obs_dim), dtype=dtype, device=device)
        self.d  = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, s, a, r, ns, done):
        i = self.ptr
        self.s[i].copy_(s)
        self.a[i, 0] = int(a)
        self.r[i, 0] = float(r)
        self.ns[i].copy_(ns)
        self.d[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(self, batch: int):
        size = self.capacity if self.full else self.ptr
        idx = torch.randint(0, size, (batch,), device=self.device)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]

@dataclass
class DQNConfig:
    total_steps: int = 300_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005                # Polyak soft update
    train_every: int = 4
    target_sync_every: int = 0        # 0 = use Polyak; >0 = hard update cadence
    buffer_size: int = 500_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.25        # fraction of steps to anneal epsilon
    grad_clip: float = 10.0
    seed: int = 42
    eval_every: int = 10_000
    log_every: int = 1_000
    reward_clip: float = 0.0          # 0 disables; else clip to [-R, R]
    run_name: str = "roundabout_dqn"

def linear_schedule(step, start, end, duration):
    t = min(1.0, step / max(1, duration))
    return start + t * (end - start)

def to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, device=device, dtype=torch.float32)

def make_env():
    base = RoundaboutSegmentedEnv(render_mode=None)  # keep headless for train
    env = DictObsToVector(base, max_count=20)
    env = FlattenMultiDiscreteAction(env)
    return env

@torch.no_grad()
def evaluate(env, q, device, episodes=3):
    q.eval()
    total = 0.0
    for _ in range(episodes):
        o, _ = env.reset()
        done = False
        ep = 0.0
        while not done:
            s = to_tensor(o, device).float().unsqueeze(0)
            a = torch.argmax(q(s), dim=1).item()
            o, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            ep += float(r)
        total += ep
    q.train()
    return total / max(1, episodes)

def global_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)

def main():
    cfg = DQNConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = QNet(obs_dim, act_dim).to(device)
    tgt = QNet(obs_dim, act_dim).to(device)
    tgt.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    loss_fn = nn.SmoothL1Loss()  # Huber
    rb = ReplayBuffer(obs_dim, cfg.buffer_size, device)

    # ----- TensorBoard -----
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("runs", f"{cfg.run_name}_{ts}")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    writer.add_text("config/json", json.dumps(cfg.__dict__, indent=2))
    # Add graph once
    dummy = torch.zeros(1, obs_dim, device=device)
    try:
        writer.add_graph(q, dummy)
    except Exception:
        pass  # some envs/models may not export cleanly; safe to ignore

    # ----- Training loop -----
    o, _ = env.reset()
    s = to_tensor(o, device).float()

    ep_return = 0.0
    ep_len = 0
    steps_start_time = time.time()
    eps_dur = int(cfg.total_steps * cfg.eps_fraction)
    best_eval = -1e9
    episodes_done = 0
    train_steps = 0

    for step in range(1, cfg.total_steps + 1):
        # epsilon-greedy
        eps = linear_schedule(step, cfg.eps_start, cfg.eps_end, eps_dur)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(torch.argmax(q(s.unsqueeze(0)), dim=1).item())

        o2, r, term, trunc, info = env.step(a)
        done = bool(term or trunc)

        if cfg.reward_clip > 0:
            r = float(np.clip(r, -cfg.reward_clip, cfg.reward_clip))

        s2 = to_tensor(o2, device).float()
        rb.add(s, a, r, s2, done)

        s = s2
        ep_return += float(r)
        ep_len += 1

        # episode end
        if done:
            writer.add_scalar("charts/episode_return", ep_return, global_step=step)
            writer.add_scalar("charts/episode_length", ep_len, global_step=step)
            episodes_done += 1
            o, _ = env.reset()
            s = to_tensor(o, device).float()
            ep_return = 0.0
            ep_len = 0

        # Learn
        if step >= cfg.learning_starts and step % cfg.train_every == 0:
            S, A, R, NS, D = rb.sample(cfg.batch_size)

            # current Q(s,a)
            q_sa = q(S).gather(1, A)  # [B,1]

            # Double DQN target
            with torch.no_grad():
                next_a = torch.argmax(q(NS), dim=1, keepdim=True)      # argmax_a' Q(s',a')
                next_q = tgt(NS).gather(1, next_a)                     # Q_tgt(s', argmax)
                y = R + (1.0 - D) * cfg.gamma * next_q

            loss = loss_fn(q_sa, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            gnorm = global_grad_norm(q)
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip)
            opt.step()
            train_steps += 1

            # Q stats
            with torch.no_grad():
                q_batch = q(S)
                q_mean   = q_batch.mean().item()
                q_max    = q_batch.max(dim=1).values.mean().item()
                target_mean = y.mean().item()

            # TB: training metrics
            writer.add_scalar("loss/td_huber", loss.item(), step)
            writer.add_scalar("q/mean", q_mean, step)
            writer.add_scalar("q/mean_max", q_max, step)
            writer.add_scalar("targets/mean", target_mean, step)
            writer.add_scalar("optim/grad_norm", gnorm, step)
            writer.add_scalar("optim/lr", opt.param_groups[0]["lr"], step)

            # target updates
            if cfg.target_sync_every and (step % cfg.target_sync_every == 0):
                tgt.load_state_dict(q.state_dict())
            else:
                with torch.no_grad():
                    for p, pt in zip(q.parameters(), tgt.parameters()):
                        pt.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)

            # occasional histograms (not every step to keep logs light)
            if train_steps % 1000 == 0:
                for name, module in q.named_modules():
                    if isinstance(module, nn.Linear):
                        writer.add_histogram(f"weights/{name}.weight", module.weight.data.cpu(), step)
                        if module.bias is not None:
                            writer.add_histogram(f"weights/{name}.bias", module.bias.data.cpu(), step)

        # Charts every log_every
        if step % cfg.log_every == 0:
            elapsed = time.time() - steps_start_time
            fps = int(cfg.log_every / max(1e-6, elapsed))
            steps_start_time = time.time()
            replay_size = rb.capacity if rb.full else rb.ptr
            writer.add_scalar("charts/fps", fps, step)
            writer.add_scalar("charts/replay_size", replay_size, step)
            writer.add_scalar("charts/epsilon", eps, step)
            print(f"[{step:>7}/{cfg.total_steps}] eps={eps:.3f}  buf={replay_size}  fps~{fps}")

        # Quick eval
        if cfg.eval_every and (step % cfg.eval_every == 0):
            avg = evaluate(make_env(), q, device, episodes=3)  # fresh env avoids any training state
            writer.add_scalar("eval/avg_return", avg, step)
            print(f"  eval_avg_return={avg:.3f}")
            if avg > best_eval:
                best_eval = avg
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(q.state_dict(), "checkpoints/dqn_roundabout_best.pt")
                writer.add_scalar("eval/best_avg_return", best_eval, step)

    # Save final
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(q.state_dict(), "checkpoints/dqn_roundabout_final.pt")
    writer.add_hparams(cfg.__dict__, {"eval/best_avg_return": best_eval})
    writer.close()
    env.close()
    print("Training complete.")

if __name__ == "__main__":
    main()
