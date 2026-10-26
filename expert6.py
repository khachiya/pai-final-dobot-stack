import numpy as np
from sim_env import DobotStackEnv, run_expert

for s in [21, 22, 23, 24, 25]:
    for color in ["blue", "red"]:
        env = DobotStackEnv(seed=s)
        env.reset(color=color)
        traj = run_expert(env)
        print(f"MARK seed {s} {color}: success={env.success()} frames={len(traj)}", flush=True)
