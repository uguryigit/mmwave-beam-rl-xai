from q_l_all import load_q, make_env, base_state_key,q_total,epsilon_greedy,unflatten_action


run_id = 0
q_table = load_q(f"test/roundabout_q_factorized/run_{run_id:02d}/Q_factorized_run{run_id:02d}_step6000.pkl.gz")
# 1) Re-create the env in human mode

env = make_env(render=True)


for ep in range(5):
    terminated = False
    truncated = False
    obs, info = env.reset(run_id)
    s_base = base_state_key(obs)

    while not (terminated or truncated):
        qv = q_total(q_table,s_base)
        a = epsilon_greedy(qv, 0)
        a0, a1 = unflatten_action(a)

        next_obs, reward, terminated, truncated, info = env.step((a0, a1))

        s_next = base_state_key(next_obs)
        s_base = s_next




env.close()