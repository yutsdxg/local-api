# 指示
別プロジェクトで開発したAnalyzeChordMelodyの資材を持っていきました。この資材はlcoal-apiに移植したいです。以下の要求に従って開発計画を立ててください。ユーザーが開発計画を確認して、問題がなければ実装に入るため、それまではソースコードに手を入れないでください。

- local-apiの中の一つのエンドポイントとして機能するようにする
- コーディングスタイルはlocal-apiの他の処理に合わせる
- AnalyzeChordMelodyはdockerコンテナ上で実行するようにしていたが、本プロジェクトのvenv仮想環境内で実行すればよい

## 実装メモ
- エンドポイント: `POST /audio/analyze/chord-melody`（パラメータなし、判定結果のみ返却）
- 入力ディレクトリ: `data/chord-melody/input`（固定、`LOCAL_API_CHORD_MELODY_INPUT_DIR`で上書き可）
- ログ出力: `data/chord-melody/logs/analysis.log`（`LOCAL_API_CHORD_MELODY_LOG_DIR`で上書き可）
- 依存パッケージ: `basic-pitch==0.2.6`, `tensorflow==2.12.0`, `scipy==1.9.3`, `pyyaml==6.0.1`
