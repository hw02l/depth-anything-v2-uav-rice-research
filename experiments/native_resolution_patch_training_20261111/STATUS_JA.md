# 作業状況

| 項目 | 値 |
|---|---|
| 最終更新 | 2026-11-11 18:16 JST |
| 実験ID | `native_resolution_patch_training_20261111` |
| branch | `research/native-resolution-patch-training-20261111` |
| 基点 | PR #5 head `4369cf2` |
| 状態 | 実装・CPU検証済み、GPGPU4 pilot前。正式学習・正式評価は未実施。 |

## 完了

- PR #5を基点とした新規branchと専用実験ディレクトリを確認した。既存untrackedファイルは変更・stageしていない。
- C0/F1/F2対応crop loader、SiLog training/checkpoint、P0/P1/P2 tiled inference、共通Validation128評価、image-paired bootstrapを追加した。
- 8件のCPU統合テストがpassした。実画像 `2025/Day019_Corn/images/8.JPG` について、3群のRGB/GT crop、SiLog forward/backward、optimizer更新、保存・再読込を確認した。toy modelによるsmokeであり、研究結果ではない。
- 追加したpaired-bootstrap合成入力テスト1件もpassした。
- 実画像寸法監査はTrain384/Validation128の計512組すべてでpassした。全組3956×5280、RGB/GT寸法不一致0、Independent testを読み込んでいない。
- seed42で固定されたpilot Train8画像（crop×view各1画像）と、過去実験の固定Validation16画像をmanifest化した。
- F2 cropのTrain384 native-pixel unique coverageを事前計算した。480-update候補は中央値6.423%で10%未満、960-update候補は中央値12.082%。事前登録規則上の候補は960 updatesである。最終configへのlockはGPU pilotの完了後に行う。pilot validation値で予算を変更しない。
- 480/960候補のcrop座標を3群共通の乱数から展開し、sampling manifestを生成した。
- Codex右側ターミナルでGPGPU4の読み取り専用preflightを実行した。GeForce GTX 1080、総VRAM8192MiB/空き8098MiB、compute process 0、login session 1、`~/gpu_work`実体 `/localdata/s0623003`、40GB中約14GB使用・約26GB空き、Singularity CE 4.2.2を確認した。Formal training開始直前にもfresh preflightを行う。
- GPU上の既存epoch20 checkpoint copy `/localdata/s0623003/local_huber_project_20261105/checkpoint_epoch_0020/` の実SHA256は、model `52769e2e95704d0862e708e8772195b861c647dd4812058eaf698a44305c3bcb`、config `3c6ab05881e6a76edbfd33ff393fdfb082582084e9262826bb1ac0fd2b1dac7c`。記録値と一致した。既存checkpointは読み取りのみ。
- GPU用SIF、既存Python拡張環境、固定source_full_v1のパスが存在することを確認した。UAV3DCrop RGB/Depthはmigration_work_20261015配下のローカル領域にあり、再ダウンロードしない。

## 未完了

1. 新規専用GPU root `~/gpu_work/native_resolution_patch_training_20261111` を作り、初期checkpointを新rootへ複製する。既存run/data/checkpointは変更しない。
2. 実験コード・manifest・configをGPU専用rootへ配置し、train/validation/pair manifestのSHAを照合する。
3. fresh preflight JSONを保存し、pilot画像に限定してC0/F1/F2の実モデルCUDA AMP forward/backward、data loading throughput、VRAM、checkpoint save/resumeを確認する。
4. 固定16 validation画像でP0/P1/P2の推論速度・tile coverage・境界処理をprofileする。pilot重みは正式runへ流用しない。
5. 事前ルールに従い予算960 updateをconfigへlockし、source/config hashesを保存する。
6. GPUに競合がないことをfresh確認後、3群を同一初期checkpoint・共通条件で学習し、各epoch評価・checkpoint・再開情報を保存する。
7. 全9組合せのValidation128評価、paired bootstrap、図表、改善/悪化例、日本語詳細報告、checkpoint SHA/provenanceを作成する。
8. 大容量ファイルを含めず、専用branchをpushし、PR #5のbranchをbaseとするPRを作成してCodexへ添付する。

## 再開時の確認

- GPGPU4の接続はCodex右側ターミナルの `s0623003@GPGPU4` promptで確認する。接続済みのこのターミナルをComputer Useで操作する。
- GPU状態はtraining開始直前に再度確認する。過去のpreflightは開始許可の代用にしない。
- `/home1/eng-std/s0623003/gpu4_research_backup_local_huber_20261105_20261105T081949Z/checkpoint_epoch_0020/` は現在の確認では見つからなかった。利用する初期重みは上記localdata上の実体であり、SHA256は一致済み。正式結果のNFS backup rootは別の新規名を使う。
- 現在までformal trainingは一切開始していない。Independent test319も読み込んでいない。

## CPU確認コマンド

```powershell
& .\.venv\Scripts\python.exe -m compileall -q experiments\native_resolution_patch_training_20261111
& .\.venv\Scripts\python.exe experiments\native_resolution_patch_training_20261111\test_native_patch.py
& .\.venv\Scripts\python.exe experiments\native_resolution_patch_training_20261111\test_analysis.py
```
