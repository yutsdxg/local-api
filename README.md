# Local Utility API

Whisper 文字起こし・yt-dlp 音声抽出・コード/メロディ判定をローカルで扱うための FastAPI サーバーです。n8n 等から HTTP 経由で利用できます。

## 前提条件

- `uv`（例: `brew install uv`）
- `whisper-cli`（例: `brew install whisper-cpp`）
- `ffmpeg`
- `yt-dlp`
- モデルファイル（デフォルト: `model/ggml-medium.bin`）

Python 3.10 と `.venv` は `uv` が `.python-version` と `pyproject.toml` をもとに用意します。

## 事前準備

- 一時/出力ディレクトリを作成: `mkdir -p data/tmp/whisper data/tmp/yt-dlp data/chord-melody/input data/chord-melody/logs logs`
- Whisper 用モデルを `model/` 配下に配置（例: `model/ggml-medium.bin`）。
- パスを変えたい場合は環境変数で上書きします（`## 環境変数` を参照）。
- Whisper はデフォルトで `-ng -nt -np` を付けて CPU モードで実行します。Metal/GPU 経路を試す場合は `LOCAL_API_WHISPER_ARGS=""` を設定してください。

## セットアップ

```bash
uv sync
```

`uv sync` は `.venv` の作成、Python 3.10 の準備、依存パッケージのインストールをまとめて行います。

## 起動方法

サーバーを起動するときは以下を実行します。

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 5050
```

`uv run` が `.venv` を自動的に使うため、事前に `source .venv/bin/activate` する必要はありません。

## テスト

```bash
uv run python -m unittest discover -s tests
```

## エンドポイント

### 1. Whisper 文字起こし

- URL: `POST /whisper`
- Body: `multipart/form-data` の `file` フィールドに音声ファイルを付与

例:

```bash
curl -X POST "http://localhost:5050/whisper" \
  -F "file=@/path/to/audio.m4a"
```

### 2. YouTube 音声抽出

- URL: `POST /yt-dlp/audio`
- Query: `video_id`（YouTube の動画 ID）

例:

```bash
curl -X POST "http://localhost:5050/yt-dlp/audio?video_id=oCVLb374gQ0"
```

保存先: `data/yt-dlp/input_<uuid>.wav`

### 3. rsync 同期

- URL: `POST /rsync`
- Query: `src`（同期元の親ディレクトリ）, `dst`（同期先の親ディレクトリ）
- 挙動: `src` 直下の *ディレクトリのみ* を列挙し、各ディレクトリに対して `rsync -aiv --delete <src>/<name>/ <dst>/<name>/` を実行します。戻り値は同期したディレクトリ名と `rsync -i` の出力行を含む配列です。

例:

```bash
curl -X POST "http://localhost:5050/rsync?src=/path/to/source&dst=/path/to/dest"
```

パスにスペースが含まれる場合は URL エンコードしてください。curlなら`--data-urlencode`が便利です:

```bash
curl -X POST "http://localhost:5050/rsync" --get \
  --data-urlencode "src=/Users/yuts/Data/Sound Library/Ableton/User Library/Samples/Samplepacks" \
  --data-urlencode "dst=/Volumes/Backup/Samples/Samplepacks"
```

```bash
curl -X POST "http://localhost:5050/rsync" --get \
  --data-urlencode "src=/Users/yuts/Data/Sound Library/Ableton/User Library/Samples/References" \
  --data-urlencode "dst=/Volumes/Backup/Samples/References"
```

### 4. コード/メロディ判定（AnalyzeChordMelody 移植）

- URL: `POST /audio/analyze/chord-melody`
- パラメータ: なし（設定されたディレクトリ配下の `.wav` を走査）
- 挙動:
  - `data/chord-melody/input` 配下の `_MLD` / `_CHP` / `_CH1` で終わる `.wav` を再帰的に対象化
  - BasicPitch でCHORD/MELODY/CHORD1を判定し、結果に応じてファイル名末尾をリネーム
  - ログは `data/chord-melody/logs/analysis.log` に出力（`<相対パス>\t<判定結果>`）
- レスポンス: 判定結果のみを返す `{"results": [{"path": "...", "result": "CHORD|MELODY|CHORD1"}, ...]}`

例:

```bash
curl -X POST "http://localhost:5050/audio/analyze/chord-melody"
```

### 5. similar-tones（類似音色検索）

- URL: `POST /audio/similar-tones/index`
- Query: `preset_dir`（プリセット音源のディレクトリ）, `output_path`（インデックス出力パス）
- 挙動: `.wav` / `.ogg` を再帰的に探索し、CLAP埋め込みでインデックスを作成

例:

```bash
curl -X POST "http://localhost:5050/audio/similar-tones/index" --get \
  --data-urlencode "preset_dir=/Users/yuts/Data/Sound Library/Ableton/User Library/Samples/Preview/Factory Packs" \
  --data-urlencode "output_path=data/similar-tones/index/ableton_factory_packs_200ms.pkl"
```

- URL: `POST /audio/similar-tones/search`
- Query: `target_path`（対象音源ファイル）, `index_path`（インデックスファイル）, `top_k`（取得件数）
- レスポンス: `text`（1行1結果のランキング表）と `results`（詳細配列）

例:

```bash
curl -X POST "http://localhost:5050/audio/similar-tones/search" --get \
  --data-urlencode "target_path=/Users/yuts/Data/Sound Library/Ableton/User Library/Samples/Samplepacks/Native Instruments - Warped Symmetry/Samples/One Shots/Synth Note/Bell_D#_Obsidian.wav" \
  --data-urlencode "index_path=data/similar-tones/index/preset.pkl" \
  --data-urlencode "top_k=10" | jq -r '.text'
```

### 6. Obsidian ノート結合エクスポート

- URL: `POST /obsidian/merge`
- パラメータ: なし（環境変数で Vault/出力先を指定）
- 挙動:
  - Vault の `inbox` / `journal` 配下を対象に Markdown を抽出
  - 除外タグ `type/snippet`, `type/account` を含むノートは除外
  - `type/journal` は 1 ファイルに結合
  - `topic/*` は直下タグごとに結合
  - それ以外は `others.md` に結合
  - 毎回洗い替えで再生成

例:

```bash
curl -X POST "http://localhost:5050/obsidian/merge"
```

### 7. Obsidian 結合ファイル → Google ドキュメント変換

- URL: `POST /obsidian/exports/google-docs`
- Query:
  - `source_path`（変換対象の Markdown もしくはディレクトリ。`OBSIDIAN_EXPORT_DIR` 配下の相対/絶対パス）
  - `title`（Google Docs のタイトル上書き、任意）
  - `folder_id`（Google Drive のフォルダ ID、任意）
- 挙動:
  - Markdown の内容をそのまま Google Docs に挿入
  - `source_path` がディレクトリの場合は配下の `.md` をすべて変換（再帰的）
  - `folder_id` 未指定時は `LOCAL_API_GOOGLE_DOCS_FOLDER_ID` を使用
  - `folder_id` がある場合はそのフォルダ内に直接作成（同名の既存ドキュメントがあれば上書き）
  - OAuth 2.0（リフレッシュトークン）を使う場合は `LOCAL_API_GOOGLE_OAUTH_*` を設定

例:

```bash
curl -X POST "http://localhost:5050/obsidian/exports/google-docs" --get \
  --data-urlencode "source_path=others.md" \
  --data-urlencode "title=Obsidian Export - Others"
```

## 環境変数

`LOCAL_API_` プレフィックス付きで設定を上書きできます。

| 変数名 | デフォルト | 説明 |
| --- | --- | --- |
| `LOCAL_API_WHISPER_BIN` | `/opt/homebrew/.../whisper-cli` | whisper-cli のパス |
| `LOCAL_API_WHISPER_MODEL_PATH` | `/Users/.../ggml-medium.bin` | モデルファイル |
| `LOCAL_API_WHISPER_ARGS` | `-ng -nt -np` | whisper-cli 追加引数。デフォルトは CPU 安定運用用。空文字で追加引数なし |
| `LOCAL_API_WHISPER_TMP_DIR` | `data/tmp` | Whisper 一時ディレクトリ |
| `LOCAL_API_FFMPEG_BIN` | `ffmpeg` | ffmpeg コマンド |
| `LOCAL_API_YTDLP_BIN` | `yt-dlp` | yt-dlp コマンド |
| `LOCAL_API_YTDLP_OUTPUT_DIR` | `data/yt-dlp` | yt-dlp 出力先 |
| `LOCAL_API_RSYNC_BIN` | `rsync` | rsync コマンド |
| `LOCAL_API_CHORD_MELODY_INPUT_DIR` | `data/chord-melody/input` | 判定対象ディレクトリ |
| `LOCAL_API_CHORD_MELODY_LOG_DIR` | `data/chord-melody/logs` | 解析ログ出力先 |
| `LOCAL_API_CHORD_MELODY_TIME_UNIT` | `0.1` | 分析時間スライス（秒） |
| `LOCAL_API_CHORD_MELODY_POLY_THRESHOLD` | `0.4` | POLY率によるCHORD判定閾値 |
| `LOCAL_API_CHORD_MELODY_POLY_NOTE_COUNT` | `3` | POLY判定とする音数 |
| `LOCAL_API_CHORD_MELODY_STABILITY_THRESHOLD` | `0.7` | CHORD1判定用の最低音安定性閾値 |
| `LOCAL_API_SIMILAR_TONES_CACHE_DIR` | `data/similar-tones/cache` | Hugging Faceモデル/キャッシュ保存先 |
| `LOCAL_API_SIMILAR_TONES_DEVICE` | `cpu` | CLAP実行デバイス（将来のMPS対応用） |
| `LOCAL_API_OBSIDIAN_VAULT_ROOT` | `/Users/yuts/Obsidian` | Obsidian Vault ルート |
| `LOCAL_API_OBSIDIAN_EXPORT_DIR` | `/Users/yuts/My Drive/ObsidianExports` | Obsidian 結合ファイルの出力先 |
| `LOCAL_API_OBSIDIAN_TARGET_DIRS` | `inbox,journal` | 対象ディレクトリ（カンマ区切り） |
| `LOCAL_API_OBSIDIAN_EXCLUDE_TAGS` | `type/snippet,type/account` | 除外タグ（カンマ区切り） |
| `LOCAL_API_OBSIDIAN_JOURNAL_TAG` | `type/journal` | ジャーナル判定タグ |
| `LOCAL_API_OBSIDIAN_TOPIC_PREFIX` | `topic/` | トピック判定接頭辞 |
| `LOCAL_API_OBSIDIAN_OTHERS_GROUP_NAME` | `others` | その他グループ名 |
| `LOCAL_API_GOOGLE_DOCS_CREDENTIALS_PATH` | `` | Google API サービスアカウント JSON パス（未指定時は ADC） |
| `LOCAL_API_GOOGLE_DOCS_FOLDER_ID` | `` | Google Drive フォルダ ID（省略可） |
| `LOCAL_API_GOOGLE_OAUTH_CLIENT_ID` | `` | OAuth 2.0 クライアント ID（ユーザー認証） |
| `LOCAL_API_GOOGLE_OAUTH_CLIENT_SECRET` | `` | OAuth 2.0 クライアントシークレット |
| `LOCAL_API_GOOGLE_OAUTH_REFRESH_TOKEN` | `` | OAuth 2.0 リフレッシュトークン |
| `LOCAL_API_GOOGLE_OAUTH_TOKEN_URI` | `https://oauth2.googleapis.com/token` | OAuth 2.0 トークンエンドポイント |

## n8n からの利用メモ

- Whisper には必ずバイナリデータを送信する（HTTP Request ノードで "Send Binary Data" を ON）。
- `host.docker.internal:5050` を指定してコンテナからホストの API を呼び出す。
