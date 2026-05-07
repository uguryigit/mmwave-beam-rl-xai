from dataclasses import dataclass
import os, random, numpy as np
try:
    import torch
except Exception:
    torch = None

@dataclass
class SeedBundle:
    master: int
    np_seed: int
    py_seed: int
    torch_seed: int
    env_seed: int
    action_seed: int
    replay_seed: int
    eval_seed: int

def make_seeds(master_seed: int) -> SeedBundle:
    # Derive child seeds deterministically (stable, portable)
    ss = np.random.SeedSequence(master_seed)
    np_seed, py_seed, torch_seed, env_seed, action_seed, replay_seed, eval_seed = ss.spawn(7)
    to_int = lambda s: int(s.generate_state(1, dtype=np.uint32)[0])
    return SeedBundle(
        master=master_seed,
        np_seed=to_int(np_seed),
        py_seed=to_int(py_seed),
        torch_seed=to_int(torch_seed),
        env_seed=to_int(env_seed),
        action_seed=to_int(action_seed),
        replay_seed=to_int(replay_seed),
        eval_seed=to_int(eval_seed),
    )

def set_global_seeds(sb: SeedBundle):
    # Python & NumPy
    random.seed(sb.py_seed)
    np.random.seed(sb.np_seed)

    # Optional: prefer Generator API over legacy global RNG where possible:
    # rng = np.random.default_rng(sb.np_seed)  # pass rng around instead of using np.random.*

    # Torch (if available)
    if torch is not None:
        torch.manual_seed(sb.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sb.torch_seed)
        # Determinism knobs
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False