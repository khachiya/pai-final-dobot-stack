"""Side-view verification video: true horizontal camera tracking the box top.
Usage: python side_view_video.py --episodes 3 --out video/side_view.mp4
"""
import argparse, os
import numpy as np
import torch

from sim_env import DobotStackEnv

OW, OH, FPS = 512, 512, 10


def overlay(img, lines):
    import cv2
    out = img.copy()
    for i, (txt, color) in enumerate(lines):
        cv2.putText(out, txt, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
    return out


def up(frame):
    import cv2
    big = cv2.resize(frame, (OW, OH), cv2.INTER_CUBIC)
    return np.ascontiguousarray(big)


def load_act(ckpt):
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt)
    return policy, pre, post, device


def side_shot(env):
    import pybullet as p
    bpos = np.array(p.getBasePositionAndOrientation(env.box)[0]) if env.box is not None else np.array([0.18, 0.0, 0.02])
    tgt = [float(bpos[0]), float(bpos[1]), float(bpos[2] + env.box_size / 2)]
    view = p.computeViewMatrixFromYawPitchRoll(tgt, 0.22, 0, -2, 0, 2)
    proj = p.computeProjectionMatrixFOV(fov=45, aspect=1.0, nearVal=0.01, farVal=3.0)
    try:
        _, _, rgb, _, _ = p.getCameraImage(512, 512, view, proj,
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        img = rgb[:, :, :3].astype(np.uint8)
        import cv2
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    except Exception:
        _, _, rgb, _, _ = p.getCameraImage(256, 256, view, proj, renderer=p.ER_TINY_RENDERER)
        img = rgb[:, :, :3].astype(np.uint8)
    return img


def rollout_once(env, policy, pre, post, device, color):
    task = env.reset(color=color)
    policy.reset()
    frames, steps, attach_idx = [], 0, None
    while not env.success() and steps < 300:
        raw = {
            "observation.state": torch.tensor(env.get_obs(), dtype=torch.float32)[None],
            "observation.images.top": torch.from_numpy(
                np.ascontiguousarray(env.render().transpose(2, 0, 1))).float().div_(255.0)[None],
        }
        obs_t = pre(raw)
        with torch.no_grad():
            action = policy.select_action(obs_t)
        a = post(action).squeeze(0).float().cpu().numpy()
        env.step(a)
        steps += 1
        img = side_shot(env)
        grab = bool(env.carrying)
        lines = [(task, (0, 255, 180)), (f"step {steps}", (255, 255, 0))]
        if bool(env.carrying):
            lines.append(("GRABBED (contact)", (0, 255, 0)))
        elif int(env.suction):
            lines.append(("suction ON: approaching", (0, 255, 255)))
        else:
            lines.append(("suction OFF", (120, 120, 120)))
        frames.append(overlay(img, lines))
        if attach_idx is None and getattr(env, "_since_attach", 10 ** 9) == 0:
            attach_idx = len(frames) - 1
    return frames, env.success(), attach_idx


def make_slide(spec, bg=(15, 20, 30)):
    import cv2
    img = np.zeros((256, 256, 3), np.uint8)
    img[:] = bg[::-1]
    y = 60
    for txt, size, color in spec:
        scale = size / 34.0
        cv2.putText(img, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color[::-1], 2, cv2.LINE_AA)
        y += int(size * 1.5)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/train/act_dobot/checkpoints/last/pretrained_model")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=2026)
    ap.add_argument("--out", default="video/side_view.mp4")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    policy, pre, post, device = load_act(args.ckpt)

    import imageio_ffmpeg
    gen = imageio_ffmpeg.write_frames(
        args.out, (OW, OH), fps=FPS, codec="libx264", pix_fmt_in="rgb24",
        output_params=["-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p"])
    gen.send(None)
    nframes = 0

    def emit(frame_rgb, times=1):
        nonlocal nframes
        f = up(frame_rgb) if frame_rgb.shape[:2] != (OH, OW) else frame_rgb
        for _ in range(times):
            gen.send(np.ascontiguousarray(f))
            nframes += 1

    emit(make_slide([
        ("Side-view check (true side, yaw=0)", 30, (255, 255, 255)),
        ("cup-box contact at box-top height, dist 22cm", 20, (120, 200, 255)),
        ("camera tracks box top, horizontal", 20, (220, 220, 220)),
    ]), int(5.0 * FPS))

    env = DobotStackEnv(seed=args.seed0)
    for ep in range(args.episodes):
        color = "red" if ep % 2 == 0 else "blue"
        for attempt in range(4):
            e = env if attempt == 0 else DobotStackEnv(seed=args.seed0 + ep * 977 + attempt * 13)
            frames, ok, attach_idx = rollout_once(e, policy, pre, post, device, color)
            if ok or attempt == 3:
                break
        emit(make_slide([
            (f"Episode {ep + 1} / {args.episodes} ({color})", 30, (255, 255, 255)),
            ("recorded on success" if ok else "FAILED", 24, (110, 255, 140) if ok else (0, 0, 255)),
        ]), FPS)
        for f in frames:
            emit(f)
        if ok and attach_idx is not None:
            emit(make_slide([("Contact slow-mo (side view)", 30, (255, 255, 255))]), FPS)
            lo, hi = max(0, attach_idx - 5), min(len(frames), attach_idx + 7)
            for f in frames[lo:hi]:
                emit(f, 3)
        print(f"ep{ep} {color}: success={ok} frames={len(frames)} attach={attach_idx}", flush=True)
    env.close()
    gen.close()
    print(f"wrote {args.out}: {nframes} frames @ {FPS} fps = {nframes / FPS:.1f} s")


if __name__ == "__main__":
    main()
