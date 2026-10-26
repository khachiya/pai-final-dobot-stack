import numpy as np, cv2
from sim_env import DobotStackEnv, run_expert

env = DobotStackEnv(seed=7)
ok_n = 0
for i in range(6):
    color = "red" if i % 2 == 0 else "blue"
    env.reset(color=color)
    traj = run_expert(env)
    ok = env.success()
    ok_n += int(ok)
    print(f"ep{i} {color}: success={ok} steps={len(traj)}", flush=True)
    if i == 0:
        cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/v2_cam.png",
                    cv2.cvtColor(traj[len(traj) // 2][2], cv2.COLOR_RGB2BGR))
print(f"expert success {ok_n}/6")
env.close()
