"""Generate expert demonstrations in PyBullet -> LeRobot dataset.

Usage (inside .venv on MSI/WSL):
  python gen_data.py --episodes 300 --repo-id local/dobot_stack_sim \
      --root /home/khach/pai-final/data/dobot_stack_sim
"""
import argparse
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sim_env import DobotStackEnv, run_expert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--repo-id", type=str, default="local/dobot_stack_sim")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    features = {
        "observation.state": {"dtype": "float32", "shape": (5,),
                              "names": ["x", "y", "z", "r", "suction"]},
        "action": {"dtype": "float32", "shape": (5,),
                   "names": ["dx", "dy", "dz", "dr", "suction"]},
        "observation.images.top": {"dtype": "image", "shape": (256, 256, 3),
                                    "names": ["height", "width", "channels"]},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=10, features=features,
                               root=args.root, robot_type="dobot_magician",
                               use_videos=False)  # PNG-in-parquet: no torchcodec/FFmpeg needed

    env = DobotStackEnv(seed=args.seed)
    n_ok, i = 0, 0
    while n_ok < args.episodes and i < args.episodes * 2:
        color = "red" if (i % 2 == 0) else "blue"
        env.reset(color=color)
        traj = run_expert(env)
        i += 1
        if not env.success():
            continue
        n_ok += 1
        for obs, action, frame in traj:
            ds.add_frame({
                "observation.state": obs.astype(np.float32),
                "action": action.astype(np.float32),
                "observation.images.top": frame,   # uint8 (H,W,3)
                "task": env.task,
            })
        ds.save_episode()
        if n_ok % 10 == 0:
            print(f"{n_ok}/{args.episodes} episodes saved", flush=True)

    ds.finalize()
    print(f"DONE {n_ok} episodes -> {args.root}")
    env.close()


if __name__ == "__main__":
    main()
