# Dobot Magician 赤/青仕分け VLA — Sim2Real（PAI最終課題）

Physical AI 応用1講座 最終課題。Dobot Magician を想定した PyBullet シミュレーション環境で、
「赤い箱は左・青い箱は右のスタックへ置く」タスクを模倣学習（ACT / SmolVLA）させる Sim2Real パイプライン。

## 構成

```
sim_env.py          Dobot近似URDF + PyBullet環境（運動学アーム、吸着、カメラ描画、成功判定）
gen_data.py         スクリプトエキスパートによるデモ自動生成 -> LeRobotデータセット(300 eps)
train_act.sh        ACT 学習 (lerobot-train, RTX 4070 / CUDA 12.8)
train_smolvla.sh    SmolVLA ファインチューン（言語条件付き VLA）
eval_rollout.py     学習済みポリシーの Sim ロールアウト成功率
render_rollout.py   提出用動画のレンダリング（2画面 + テロップ）
```

## タスク定義

- 観測: 斜め俯瞰 RGB (256x256) + 状態 `[x, y, z, r, suction]`
- 行動: `[dx, dy, dz, dr, suction_target]`（デルタは 2cm/ステップにクリップ）
- 言語タスク: `red box to left stack` / `blue box to right stack`
- ドメインランダライゼーション: 箱の初期位置・サイズ・色ジッタ、摩擦係数、カメラ姿勢ノイズ

## 使い方

```bash
# 1. データ生成（約300本, CPU）
python gen_data.py --episodes 300 --root data/dobot_stack_sim

# 2. 学習（GPU）
bash train_act.sh          # ACT: 20k steps, ~2h on RTX 4070 Laptop
bash train_smolvla.sh      # SmolVLA base からファインチューン

# 3. 評価
python eval_rollout.py --policy-type act \
    --ckpt outputs/train/act_dobot/checkpoints/last/pretrained_model --episodes 20

# 4. 動画レンダリング
python render_rollout.py --policy-type act \
    --ckpt outputs/train/act_dobot/checkpoints/last/pretrained_model --out rollout_act.mp4
```

## 環境

- Ubuntu (WSL) / Python 3.11 / torch 2.10 cu128 / lerobot 0.4.4 / pybullet 3.2.7
- データは `use_videos=False`（PNG保存）: FFmpeg/torchcodec 不要

## Sim2Real メモ（実機 Dobot Magician への転移）

1. URDF のリンク長 L1/L2/H_BASE/GRIP_LEN を実測値に校正
2. カメラを Sim と同じ俯瞰アングルに固定し、ArUco で手→カメラ変換を求める
3. 制御は 1ステップ ≤2cm のデルタ PTP（ServoJ/MovJ）にマップ。吸着 ON/OFF は 0.5s ディレイ
4. 実機デモを 10〜20 本追加学習すると成功率が向上（Sim2Real gap 対策）

## 注意

- 本リポジトリは講座課題提出用。シミュレーション結果であり、実機映像は別途撮影。
- ライセンス: ゼロ（課題提出用途に限定しないパブリックドメイン相当で公開可）
