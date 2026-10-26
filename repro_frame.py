import numpy as np, pybullet as p, torch, cv2
from sim_env import DobotStackEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

CKPT = "outputs/train/act_dobot/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(CKPT).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)

env = DobotStackEnv(seed=2026)
env.reset(color="red")
policy.reset()
for step in range(1, 7):
    raw = {"observation.state": torch.tensor(env.get_obs(), dtype=torch.float32)[None],
           "observation.images.top": torch.from_numpy(
               np.ascontiguousarray(env.render().transpose(2, 0, 1))).float().div_(255.0)[None]}
    a = post(policy.select_action(pre(raw))).squeeze(0).float().cpu().numpy()
    env.step(a)

tip = env._tip_pose()[:3]
bp = np.array(p.getBasePositionAndOrientation(env.box)[0])
print(f"MARK carry={int(env.carrying)} suc={env.suction} tipz={tip[2]:.4f} "
      f"boxtop={bp[2] + env.box_size / 2:.4f} gap_tip={tip[2] - (bp[2] + env.box_size / 2):+.4f}")

bpos = bp
tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + env.box_size / 2)]
view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.20, 90, -3, 0, 2)
proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
_, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
img = np.asarray(rgb)[:, :, :3].astype(int)
cv2.imwrite("/mnt/c/Users/khach.MSI/AppData/Local/hermes/cache/scratch/repro.png",
            cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

# column through box center: dark runs + red box
R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
red = (R > G + 40) & (R > B + 40)
ys, xs = np.where(red)
print("MARK red yrange:", ys.min(), ys.max(), "xrange:", xs.min(), xs.max())
cx = (xs.min() + xs.max()) // 2
dark = (R + G + B) < 180
col = dark[:, cx]
dys = np.where(col)[0]
above = dys[dys < ys.min() - 2]
print("MARK col x:", cx, "lowest dark above box:", above.max() if len(above) else None,
      "gap_px:", ys.min() - (above.max() if len(above) else ys.min()))
