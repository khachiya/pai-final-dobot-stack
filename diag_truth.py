import numpy as np, pybullet as p, torch
from sim_env import DobotStackEnv, CUP_DROP
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = "outputs/train/act_dobot/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(CKPT).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)
env = DobotStackEnv(seed=2026)
env.reset(color="red"); policy.reset()
for step in range(12):
    raw = {"observation.state": torch.tensor(env.get_obs(), dtype=torch.float32)[None],
           "observation.images.top": torch.from_numpy(
               np.ascontiguousarray(env.render().transpose(2, 0, 1))).float().div_(255.0)[None]}
    a = post(policy.select_action(pre(raw))).squeeze(0).float().cpu().numpy()
    env.step(a)
    tip = env._tip_pose()[:3]
    ls = p.getLinkState(env.arm, 5)
    l5 = np.array(ls[0])
    bz = np.array(p.getBasePositionAndOrientation(env.box)[0])
    print(f"MARK s{step:02d} cmd={a[4]:+.2f} suc={env.suction} carry={int(env.carrying)} "
          f"tipz={tip[2]:.4f} link5z={l5[2]:.4f} boxz={bz[2]:.4f} boxy={bz[1]:+.3f} tipx={tip[0]:.3f} boxx={bz[0]:.3f}")
