"""Roll out a trained policy (ACT or SmolVLA) in the PyBullet env; print success rate.

Uses lerobot's saved pre/post-processor pipelines so normalization and
un-normalization match training exactly.

Usage:
  python eval_rollout.py --policy-type act --ckpt outputs/train/act_dobot/checkpoints/last/pretrained_model
"""
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-type", type=str, default="act")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    policy = load_policy(args.policy_type, args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()

    from lerobot.policies.factory import make_pre_post_processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.ckpt)

    env = DobotStackEnv(seed=args.seed)
    results = {"red": [0, 0], "blue": [0, 0]}

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
        ok = env.success()
        results[color][0] += int(ok)
        results[color][1] += 1
        print(f"ep{ep} {color}: success={ok} steps={step}", flush=True)

    total_ok = sum(v[0] for v in results.values())
    total_n = sum(v[1] for v in results.values())
    print(f"\nRESULT red {results['red'][0]}/{results['red'][1]}, "
          f"blue {results['blue'][0]}/{results['blue'][1]}, "
          f"overall {total_ok}/{total_n} ({100*total_ok/max(1,total_n):.0f}%)")
    env.close()


if __name__ == "__main__":
    main()
