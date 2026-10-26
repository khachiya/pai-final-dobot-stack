import numpy as np, torch
from sim_env import DobotStackEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = "outputs/train/act_dobot/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(CKPT)
policy.to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)

env = DobotStackEnv(seed=999)
for ep in [3]:
    color = "red" if ep % 2 == 0 else "blue"
    env.reset(color=color)
    policy.reset()
    print(f"=== ep{ep} {color} box={np.round(env.rng and [0],2)} size={env.box_size:.3f}")
    for step in range(80):
        raw = {
            "observation.state": torch.tensor(env.get_obs(), dtype=torch.float32)[None],
            "observation.images.top": torch.from_numpy(
                np.ascontiguousarray(env.render().transpose(2, 0, 1))).float().div_(255.0)[None],
        }
        a = post(policy.select_action(pre(raw))).squeeze(0).float().cpu().numpy()
        import pybullet as p
        tip = env._tip_pose()[:3]
        bpos = np.array(p.getBasePositionAndOrientation(env.box)[0]) if env.box else None
        gap = tip[2] - (bpos[2] + env.box_size / 2) if bpos is not None else float("nan")
        dxy = max(abs(tip[0]-bpos[0]), abs(tip[1]-bpos[1])) if bpos is not None else float("nan")
        print(f"s{step:03d} cmd={a[4]:+.2f} suc={env.suction} carry={int(env.carrying)} "
              f"tipz={tip[2]:.3f} gap={gap:+.3f} dxy={dxy:.3f} armed={int(getattr(env,'_armed',False))} asst={int(getattr(env,'_assist',False))}", flush=True)
        env.step(a)
        if env.success():
            print(f"s{step:03d} SUCCESS"); break
