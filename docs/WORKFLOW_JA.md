# ローカル中心の研究運用

通常のコード編集、設定作成、CPU分析、図表作成、LaTeX執筆、Git commit/pushは、9月20日の既存Windows研究リポジトリを正本として行う。9月22日のDA-2K用フォルダーは今回の統合先ではない。

1. 作業開始時に`git status --short`とブランチを確認し、既存の未コミット・未追跡ファイルを保護する。
2. 過去実験の`source_snapshot/`、`source_v4/`、`source_full_v1/`は固定スナップショットである。整形・改行変換・import修正を含め編集しない。`scripts/research_workspace.py verify`でSHA256を確認する。今後の補助開発は`src/uav_metric_research/`や新しい実験フォルダーで行う。既存`uav_metric_finetuning/`は移行前の準備版として保持している。
3. GitHubへcode/config/manifest/Notebook/小さい結果・図・論文をcommit/pushする。モデル、データ、全予測cache、仮想環境、認証情報は含めない。force push、hard reset、無断削除をしない。
4. GPU実験時のみColabを使う。Windowsの`C:\...`はColabから参照できない。公開GitHubブランチをColabの`/content/research`へcloneし、固定ソースのハッシュをDriveの元run metadataと照合する。別案としてローカルCLIの`bundle`でZIPを作り、通常のDriveアップロードで受け渡す。ソースの流れは **local → GitHub/bundle → Colab** の一方向とする。
5. Driveは既存データ・model cache・完全checkpoint・結果原本・バックアップの保存場所である。Colabが生成した結果は **Colab → Drive → local** へ取り込む。`research_workspace.py import-results`は新しい`imports/<内容hash>/`へ保存し、既存結果を上書きしない。比較・検証後にローカルで正規の表と図を更新する。Driveとlocalで同じソースを別々に編集する双方向同期は行わない。
6. 第3回は20epoch計画の5epochで停止中。再開には元の環境版・コード署名・分割・モデルキャッシュ・完全checkpointが必要である。残り15epochを実行する新しい指示と無料GPU割当が得られるまで開始しない。新GPUセッションではVRAM事前計測を行い、失敗・制限時は停止する。署名やruntime版不一致は迂回しない。
7. 実験後はDriveのcomplete marker/hash/同期を確認してからランタイムを終了する。結果をlocalへ取り込み、全画素・全ペアと固定標本を区別して報告する。test319組・過去52組・稲高精度について未検証であることを保持する。

依頼例：「このローカルrepoを正本として、WORKFLOW_JA.mdを守り、既存runと固定sourceを確認してから○○を実施。GPU学習は別途明示した場合だけ行う。」
