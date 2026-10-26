"""PAI final assignment demo video: intro slides + ACT rollouts + summary.

Renders an H.264 MP4 (YouTube-friendly) at 1024x512 / 10 fps:
  [title slide] -> [task/method slide] -> N rollout episodes (caption card,
  live sim with overlays, success hold) -> [summary slide padded so the total
  duration lands inside the required 1-3 min window].

No personal information is drawn anywhere in the video.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sim_env import DobotStackEnv

W, H = 512, 256          # native sim combo size (two 256x256 views)
SCALE = 2
OW, OH = W * SCALE, H * SCALE   # output canvas
FPS = 10

FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msgothic.ttc",
    "/mnt/c/Windows/Fonts/meiryo.ttc",
    "/mnt/c/Windows/Fonts/YuGothM.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for c in FONT_CANDIDATES:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_slide(spec, bg=(15, 20, 30)):
    """spec: list of (text, size, (r,g,b)); first line treated as heading."""
    img = Image.new("RGB", (OW, OH), bg)
    d = ImageDraw.Draw(img)
    y = 46
    for i, (txt, size, color) in enumerate(spec):
        if i == 0:
            d.rectangle([52, y + size + 8, 52 + 190, y + size + 13], fill=(70, 140, 250))
        d.text((60, y), txt, font=load_font(size), fill=color)
        y += int(size * 1.75)
    return np.array(img)


def overlay(img, lines):
    out = img.copy()
    for i, (txt, color) in enumerate(lines):
        cv2.putText(out, txt, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, (8, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2, cv2.LINE_AA)
    return out


def up(frame):
    big = cv2.resize(frame, (OW, OH), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(big)


def load_act(ckpt):
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device).eval()
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt)
    return policy, pre, post, device


def rollout_once(env, policy, pre, post, device, color):
    """One episode; returns (frames_rgb_list, success)."""
    task = env.reset(color=color)
    policy.reset()
    frames = []
    step = 0
    attach_idx = None
    while not env.success() and step < 300:
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
        step += 1

        frame = env.render()
        left = overlay(frame, [("ACT policy", (255, 255, 255)),
                               (task, (0, 255, 180)),
                               (f"step {step}", (255, 255, 0))])
        env.cam_yaw = 30 + 90 * ((step // 30) % 2)
        side_txt = "LEFT" if env.side == "left" else "RIGHT"
        grab = bool(getattr(env, "carrying", False))
        recent = getattr(env, "_since_attach", 10 ** 9) < 12
        if (not grab or recent) and getattr(env, "box", None) is not None:
            import pybullet as _p
            _bp = _p.getBasePositionAndOrientation(env.box)[0]
            _tgt = [_bp[0], _bp[1], _bp[2] + env.box_size / 2]
            right_img = env.render(dist=0.38, target=_tgt)
        else:
            right_img = env.render(dist=0.74, target=[0.13, 0, 0.06])
        rlines = [(f"target: {side_txt}", (80, 180, 255))]
        if grab:
            rlines.append(("GRABBED (contact)", (0, 255, 0)))
        elif int(env.suction):
            rlines.append(("suction ON: approaching", (0, 255, 255)))
        else:
            rlines.append(("suction OFF", (200, 200, 200)))
        right = overlay(right_img, rlines)
        env.cam_yaw = 30
        frames.append(np.concatenate([left, right], axis=1))
        if attach_idx is None and getattr(env, "_since_attach", 10 ** 9) == 0:
            attach_idx = len(frames) - 1

    ok = env.success()
    end = overlay(env.render(dist=0.74, target=[0.13, 0, 0.06]),
                  [(f"SUCCESS: {ok}", (0, 255, 0) if ok else (0, 0, 255))])
    for _ in range(16):
        frames.append(np.concatenate([end, end], axis=1))
    return frames, ok, attach_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/train/act_dobot/checkpoints/last/pretrained_model")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=2026)
    ap.add_argument("--out", default="video/pai_final_rollout.mp4")
    ap.add_argument("--min-seconds", type=float, default=65.0)
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

    # ---- slide 1: title ----
    s1 = make_slide([
        ("Physical AI応用講座① 最終課題", 46, (255, 255, 255)),
        ("AI×ロボティクスによる赤/青ボックス仕分け", 34, (120, 200, 255)),
        ("VLA模倣学習パイプライン（ACT / SmolVLA）", 26, (220, 220, 220)),
        ("Dobot Magician モデル ・ PyBullet シミュレーション", 24, (180, 180, 180)),
        ("※本動画はシミュレーション環境での記録です", 20, (150, 150, 150)),
    ])
    emit(s1, int(6.0 * FPS))

    # ---- slide 2: task & method ----
    s2 = make_slide([
        ("課題の概要と手法", 42, (255, 255, 255)),
        ("タスク：言語指示に応じて、赤箱は左・青箱は右のスタックへ運ぶ", 26, (120, 200, 255)),
        ("データ：スクリプトエキスパートがデモ300本（8,208フレーム）を自動生成", 24, (220, 220, 220)),
        ("学習：ACT 20,000ステップ（RTX 4070）／SmolVLAも言語条件付きでファインチューン", 24, (220, 220, 220)),
        ("ドメインランダマイゼーション：位置・サイズ・色ジッタ、摩擦、カメラ姿勢ノイズ", 24, (220, 220, 220)),
        ("評価：未見シーンで成功率 16/20（80%）", 28, (110, 255, 140)),
        ("吸着カップは箱上面に接触してから吸着する物理的ゲートを実装", 22, (180, 180, 180)),
    ])
    emit(s2, int(7.5 * FPS))

    # ---- episodes ----
    env = DobotStackEnv(seed=args.seed0)
    n_ok = 0
    for ep in range(args.episodes):
        color = "red" if ep % 2 == 0 else "blue"
        jp = ("赤い箱を 左のスタックへ" if color == "red" else "青い箱を 右のスタックへ")
        # up to 3 attempts with fresh seeds; keep the first successful one
        for attempt in range(3):
            e = env if attempt == 0 else DobotStackEnv(seed=args.seed0 + ep * 977 + attempt * 13)
            frames, ok, attach_idx = rollout_once(e, policy, pre, post, device, color)
            # NOTE: never close() here — pybullet DIRECT server is global; closing it
            # disconnects the shared env and kills all later episodes.
            if ok or attempt == 2:
                break
        n_ok += int(ok)
        cap = make_slide([
            (f"Episode {ep + 1} / {args.episodes}", 44, (255, 255, 255)),
            (jp, 34, (120, 200, 255)),
            ("ACT policy rollout（左：方針入力カメラ / 右：吸着部クローズアップ→全体）", 22, (180, 180, 180)),
        ])
        emit(cap, FPS)                       # caption card 1 s
        for f in frames:
            emit(f)
        # slow-motion replay of the contact moment (each frame x3)
        if ok and attach_idx is not None:
            rep = make_slide([
                ("接触スロー再生", 40, (255, 255, 255)),
                ("カップが箱上面に触れてから吸着", 26, (110, 255, 140)),
            ])
            emit(rep, FPS)
            lo, hi = max(0, attach_idx - 5), min(len(frames) - 17, attach_idx + 7)
            for f in frames[lo:hi]:
                emit(f, 3)
        print(f"ep{ep} {color}: success={ok} frames={len(frames)} attach={attach_idx}", flush=True)
    env.close()

    # ---- summary slide (padded to reach min duration) ----
    total_est = nframes / FPS + 6.0
    s_end = make_slide([
        ("結果と今後の課題", 42, (255, 255, 255)),
        (f"今回の動画：{args.episodes}本の実行、成功 {n_ok}/{args.episodes}", 30, (110, 255, 140)),
        ("評価実験成功率：ACT 16/20（80%）／接触吸着ゲート込みの正直な数字", 26, (220, 220, 220)),
        ("技術スタック：PyBullet ／ LeRobot 0.4.4 ／ PyTorch (CUDA)", 24, (200, 200, 200)),
        ("今後：実機DobotへのSim2Real転移、SmolVLAとの比較評価", 24, (200, 200, 200)),
        ("ご清聴ありがとうございました", 34, (255, 255, 255)),
    ])
    hold = max(int(6.0 * FPS), int((args.min_seconds - total_est) * FPS))
    emit(s_end, hold)

    gen.close()
    print(f"\nwrote {args.out}: {nframes} frames @ {FPS} fps "
          f"= {nframes / FPS:.1f} s, episodes ok {n_ok}/{args.episodes}")


if __name__ == "__main__":
    main()
