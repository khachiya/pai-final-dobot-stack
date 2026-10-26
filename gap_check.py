import numpy as np, pybullet as p, torch
from sim_env import DobotStackEnv, CUP_DROP
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = "outputs/train/act_dobot/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(CKPT).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)
env = DobotStackEnv(seed=2026)
env.reset(color="red"); policy.reset()
gaps = []
for step in range(60):
    raw = {"observation.state": torch.tensor(env.get_obs(), dtype=torch.float32)[None],
           "observation.images.top": torch.from_numpy(
               np.ascontiguousarray(env.render().transpose(2, 0, 1))).float().div_(255.0)[None]}
    a = post(policy.select_action(pre(raw))).squeeze(0).float().cpu().numpy()
    env.step(a)
    if env.carrying:
        tip = env._tip_pose()[:3]
        bz = p.getBasePositionAndOrientation(env.box)[0][2]
        gaps.append(((tip[2] + CUP_DROP) - (bz + env.box_size / 2)) * 1000.0)
    if env.success():
        break
print(f"MARK n_carry={len(gaps)} gap_mm min={min(gaps):.2f} max={max(gaps):.2f} mean={np.mean(gaps):.2f}")
