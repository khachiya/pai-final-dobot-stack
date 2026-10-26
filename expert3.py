import numpy as np
from sim_env import DobotStackEnv, run_expert

for s in [11, 12, 13]:
    env = DobotStackEnv(seed=s)
    env.reset(color="red")
    traj = run_expert(env)
    print(f"MARK seed {s}: success={env.success()} frames={len(traj)}", flush=True)
