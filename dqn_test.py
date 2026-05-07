import time

from dqn_all import make_wrapped_env
from stable_baselines3 import DQN
from seed import set_global_seeds, make_seeds



sb = make_seeds(42)
set_global_seeds(sb)
print(sb.__dict__)

pair_idx = 26

# 1) Re-create the env in human mode
watch_env,_ = make_wrapped_env(render_mode="none", n_positions=1, default_pair=pair_idx, seed=sb.env_seed)

# 3) Load (or reuse) your model and run a few episodes
model = DQN.load(f"logs/roundabout_dqn/run_{pair_idx:02d}/dqn_roundabout_mlp_run{pair_idx:02d}.zip", seed=sb.torch_seed)

ang_deg = [65, 90, 120]
dir_deg = [0, 45, 90, 135, 180, 225, 270, 315]

actions = []
for ep in range(100):
    sb = make_seeds(42+ep)
    set_global_seeds(sb)
    obs, info = watch_env.reset(seed=sb.eval_seed    )
    done = False; truncated = False; ret = 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)  # no exploration
        actions.append((dir_deg[action//3], ang_deg[action%3]))
        # print(dir_deg[action//3], ang_deg[action%3])
        obs, reward, done, truncated, info = watch_env.step(action)
        ret += reward
        # Slow it down so you can see what's happening (optional)
    print(f"[watch] episode {ep+1} return={ret:.2f}")



# Count the actions and find most selected
from collections import Counter
action_counts = Counter(actions)
sorted_actions = action_counts.most_common()

print("\nAction frequency (most selected first):")
for i, ((dir_deg, ang_deg), count) in enumerate(sorted_actions):
    percentage = (count / len(actions)) * 100
    print(f"{i+1}. Dir: {dir_deg}°, Angle: {ang_deg}° - Count: {count} ({percentage:.1f}%)")

watch_env.close()
