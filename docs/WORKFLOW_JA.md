# ローカル中心の研究運用

通常のコード編集、設定作成、CPU分析、図表作成、LaTeX執筆、Git commit/pushは、9月20日の既存Windows研究リポジトリを正本として行う。9月22日のDA-2K用フォルダーは今回の統合先ではない。

1. 作業開始時に`git status --short`とブランチを確認し、既存の未コミット・未追跡ファイルを保護する。
2. 過去実験の`source_snapshot/`、`source_v4/`、`source_full_v1/`は固定スナップショットである。整形・改行変換・import修正を含め編集しない。`scripts/research_workspace.py verify`でSHA256を確認する。今後の補助開発は`src/uav_metric_research/`や新しい実験フォルダーで行う。既存`uav_metric_finetuning/`は移行前の準備版として保持している。
3. GitHubへcode/config/manifest/Notebook/小さい結果・図・論文をcommit/pushする。モデル、データ、全予測cache、仮想環境、認証情報は含めない。force push、hard reset、無断削除をしない。
4. GPU実験は明示的に許可された無料Colabまたは研究室GPUでのみ行う。Windowsの`C:\...`はリモート環境から直接参照できない。Colabでは公開GitHubブランチまたはbundleを使い、研究室GPUでは許可されたSSH経路で必要なコード・データをGPU機の`~/gpu_work`へ配置する。固定ソースのハッシュを元run metadataと照合し、ソースの流れは **local → GitHub/Drive → 実行環境** の一方向とする。
5. Driveは既存データ・model cache・完全checkpoint・Colab結果・バックアップの保存場所であり、研究室GPUの一時計算は`~/gpu_work`、重要checkpointの退避先は許可されたNFS等とする。実験結果は **実行環境 → Drive/NFS → local** の一方向で取り込み、新しい内容hash付きの場所へ保存して既存結果を上書きしない。比較・検証後にローカルで正規の表と図を更新する。Driveとlocalで同じソースを別々に編集する双方向同期は行わない。
6. 第3回の20epoch計画は完了済みであり、既存runを追加学習で再開しない。epoch 17のRMSE最良checkpointとepoch 20の最終checkpointを保持し、次の方法比較は新しいrun・新しい実験IDで行う。GPGPU4では共有GPUの事前確認、ローカル計算領域、checkpoint検証、利用後のプロセス・コンテナ・SSH終了確認を必ず行う。署名やruntime版不一致は迂回しない。
7. 実験後は実行先に応じてDriveまたは研究室NFSのcomplete marker/hash/同期を確認してからGPU・コンテナ・SSHを終了する。結果をlocalへ取り込み、全画素・全ペアと固定標本を区別して報告する。test319組・過去52組・稲高精度について未検証であることを保持する。

依頼例：「このローカルrepoを正本として、WORKFLOW_JA.mdを守り、既存runと固定sourceを確認してから○○を実施。GPU学習は別途明示した場合だけ行う。」
