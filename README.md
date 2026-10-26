# pai-final-dobot-stack — Dobot Magician 仕分けピック＆プレースの模倣学習

Physical AI 応用講座 最終課題。Dobot Magician 風モデルを用いた PyBullet
シミュレーション環境で、「赤箱は左・青箱は右のスタックへ運ぶ」言語条件付き
仕分けタスクを模倣学習（ACT）で獲得するパイプライン。
実機所有を前提に Sim2Real を目標としたが、Sim の完成度向上を優先し、
実機転移は時間切れで未達。本リポジトリは Sim 段階の成果物である。

## 成果物

- 提出動画：`video/pai_final_rollout_v8.mp4`（166.6秒・16/16成功・説明スライド付き、
  接触クローズアップ＋GRABBED表示＋接触スロー再生×3）
- 真横検証映像：`video/side_view_v6.mp4`（水平視点で接触・中央寄せを確認）
- 説明資料：`説明資料_PAI最終課題.md`（PDF化して提出）
- 評価：厳格な接触ゲート条件で **16/20（80%）**（赤 8/10・青 8/10）

## 構成

```
sim_env.py            Dobot近似URDF + PyBullet環境（厳密リンク状態ベースの先端位置、
                      接触ゲート吸着、中央寄せアシスト、成功判定）
gen_data.py           スクリプトエキスパートによるデモ自動生成 -> LeRobotデータセット(300eps)
train_act.sh          ACT学習 (lerobot-train, 20k steps, RTX 4070 / CUDA)
train_smolvla.sh      SmolVLAファインチューン（言語条件付きVLA）
eval_rollout.py       学習済みポリシーのSimロールアウト成功率
render_final_video.py 提出用動画のレンダリング（2画面＋テロップ＋スロー再生）
side_view_video.py    真横カメラ検証映像のレンダリング
models/magician/      実機風Dobot Magicianモデル（URDF＋DAEメッシュ）
```

## タスク定義

- 観測：斜め俯瞰 RGB（256x256）＋ 状態 `[x, y, z, r, suction]`
- 行動：`[dx, dy, dz, dr, suction_target]`（デルタは 2cm/ステップにクリップ）
- 言語タスク：`red box to left stack` / `blue box to right stack`
- ドメインランダマイゼーション：箱の初期位置・サイズ（±10%）・色ジッタ、
  摩擦係数、カメラ姿勢ノイズ

## 工夫点

- 接触ゲート吸着：カップ底面〜箱上面ギャップ -15mm〜+8mm かつ xy が箱上面内の
  場合のみ吸着（遠方テレポート吸着を廃止）。運搬中は毎ステップ再固定。
- TCP実測校正：カップ底面の視覚的位置をオレンジ染色＋既知高さマーカー列および
  2視点ステレオ三角測量で実測し、解析FKの姿勢依存誤差を排除。
  先端位置は PyBullet リンク状態から直接計算（`CUP_LOCAL=[0,0,-64mm]`）。
- 中央寄せアシスト＋ゲートxy厳格化（24mm→12mm）：隅・外側吸着を防止。
- 成功判定の厳格化：持ち上げ履歴＋目標4cm以内を要求（置いただけは除外）。

## 使い方

```bash
# 1. データ生成（300本）
python gen_data.py --episodes 300 --root data/dobot_stack_sim

# 2. 学習（GPU）
bash train_act.sh          # ACT: 20k steps, ~2.5h on RTX 4070 Laptop

# 3. 評価
python eval_rollout.py --policy-type act \
    --ckpt outputs/train/act_dobot/checkpoints/last/pretrained_model --episodes 20

# 4. 動画レンダリング
python render_final_video.py --episodes 16 --out video/pai_final_rollout_v8.mp4
python side_view_video.py --episodes 2 --out video/side_view_v6.mp4
```

## 環境

- Ubuntu (WSL) / Python 3.11 / torch 2.10 cu128 / lerobot 0.4.4 / pybullet 3.2.7
- データは `use_videos=False`（PNG保存）

## 備考

- `data/`（115MB）・`outputs/`（学習済みチェックポイント約5GB）は容量のため
  リポジトリから除外。上記コマンドで再生成・再学習できる。
- 成果物に個人情報（氏名・所属・メールアドレス等）は含まない。

## Sim2Real メモ（実機 Dobot Magician への転移・未達）

1. URDF のリンク長を実測値に校正
2. 実画像でのドメイン適応（照明・背景差分）
3. 吸着圧・カップ変形の実機同定
