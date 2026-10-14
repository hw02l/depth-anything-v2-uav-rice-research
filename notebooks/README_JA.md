# Colab受け渡し

`colab_train_final_20261013_v2.ipynb`は第1回、`colab_metric_outdoor_full_20261014.ipynb`は第3回のNotebookから、セル出力・execution_count・個人metadataを除いた版である。元の実行履歴付きNotebookはローカルのGit対象外`work/local_migration_20261014/original_notebooks/`とDriveに保持した。セルソースは同一である。

今後の受け渡しはlocal → GitHub → Colabを基本とする。Colabでは`git clone --branch research/local-first-20261014 https://github.com/hw02l/depth-anything-v2-uav-rice-research.git /content/research`で本ブランチを取得できる。固定snapshotの全SHA256を必ずDriveのrun_metadataと照合する。

`scripts/research_workspace.py bundle --output work/colab_bundles/trial3_resume_code.zip`でも、固定source・署名metadata・ファイルhash一覧を含むZIPをローカルで生成できる。ZIPを通常のDrive操作でアップロードしてColabで展開する。zip内のファイル名はrepo相対パスであり、Windows固有の絶対パスを使用しない。

元の実行Notebookを移行作業として実行していない。GPUの新規割当、Drive mount、pip変更、train/resumeは今回一切行っていない。再開前には、元runtime版と同一署名、無料GPU、VRAM事前計測、Driveの完全checkpoint、既存512組を確認する。署名不一致を無効化しない。全epoch数20を維持し、承認された停止epochのみ指定する。

Colabの結果はDriveへ永続化した後localへ取得する。取得フォルダーに対し`research_workspace.py import-results --incoming <folder> --experiment <run-id>`を実行すると、内容hashごとの新しい`imports/`へ取り込み、既存結果を上書きしない。

新しい入口はresume_existing_full_run.ipynb。正式run署名・runtime・checkpointを検証し、新しいVRAM計測を行う構成である。元Notebookの初期作成セルを再実行する代わりにこちらを用いる。STOP_AFTER_EPOCH=5の既定値では追加学習を開始できず、次段階の明示的な許可後にだけ変更する。
