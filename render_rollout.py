"""Render an MP4 of a trained policy rollout in the PyBullet sim (for the course video).

Usage:
  python render_rollout.py --policy-type act --ckpt outputs/train/act_dobot/checkpoints/last/pretrained_model \
      --episodes 2 --out rollout_act.mp4
"""
import argparse
import cv2
import numpy as np
import torch

from sim_env import DobotStackEnv


def load_policy(policy_type: str, ckpt: str):
    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy.from_pretrained(ckpt)
    if policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAPolicy.from_pretrained(ckpt)
    raise ValueError(policy_type)


def overlay(img, lines):
    out = img.copy()
    for i, (txt, color) in enumerate(lines):
        cv2.putText(out, txt, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-type", type=str, default="act")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", type=str, default="rollout.mp4")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    policy = load_policy(args.policy_type, args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.ckpt)

    env = DobotStackEnv(seed=args.seed)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (512, 256))

    for ep in range(args.episodes):
        color = "red" if ep % 2 == 0 else "blue"
        task = env.reset(color=color)
        policy.reset()
        step = 0
        while not env.success() and step < 300:
            raw = {
                "observation.state": torch.tensor(env.get_obs(), dtype=torch.float32)[None],
                "observation.images.top": torch.from_numpy(
                    np.ascontiguousarray(env.render().transpose(2, 0, 1))).float().div_(255.0)[None],
            }
            if args.policy_type == "smolvla":
                raw["task"] = task
            obs_t = preprocessor(raw)
            with torch.no_grad():
                action = policy.select_action(obs_t)
            a = postprocessor(action).squeeze(0).float().cpu().numpy()
            env.step(a)
            step += 1

            frame = env.render()                       # sim view (policy input)
            side = "LEFT" if env.side == "left" else "RIGHT"
            left = overlay(frame, [(f"{args.policy_type.upper()} policy", (255, 255, 255)),
                                   (task, (0, 255, 180)),
                                   (f"step {step}", (255, 255, 0))])
            # second view: slightly different camera for context
            env.cam_yaw = 30 + 90 * ((step // 30) % 2)
            right = overlay(env.render(), [(f"target: {side}", (80, 180, 255)),
                                           (f"suction={int(env.suction)}", (200, 200, 200))])
            env.cam_yaw = 30
            combo = np.concatenate([left, right], axis=1)
            vw.write(cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))

        ok = env.success()
        end = overlay(env.render(), [(f"SUCCESS: {ok}", (0, 255, 0) if ok else (0, 0, 255))])
        for _ in range(15):
            vw.write(cv2.cvtColor(np.concatenate([end, end], axis=1), cv2.COLOR_RGB2BGR))
        print(f"ep{ep} {color}: success={ok}", flush=True)

    vw.release()
    env.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
