# Local Utility API

Whisper 文字起こしと yt-dlp 音声抽出をローカルで扱うための FastAPI サーバーです。n8n 等から HTTP 経由で利用できます。

## 前提条件

- Python 3.11
- `whisper-cli`（例: `brew install whisper-cpp`）
- `ffmpeg`
- `yt-dlp`
- モデルファイル（デフォルト: `model/ggml-medium.bin`）

## 事前準備

- 一時/出力ディレクトリを作成: `mkdir -p data/tmp/whisper data/tmp/yt-dlp logs`
- Whisper 用モデルを `model/` 配下に配置（例: `model/ggml-medium.bin`）。
- パスを変えたい場合は環境変数で上書きします（`## 環境変数` を参照）。

## セットアップ

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 起動方法

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 5050
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

## 環境変数

`LOCAL_API_` プレフィックス付きで設定を上書きできます。

| 変数名 | デフォルト | 説明 |
| --- | --- | --- |
| `LOCAL_API_WHISPER_BIN` | `/opt/homebrew/.../whisper-cli` | whisper-cli のパス |
| `LOCAL_API_WHISPER_MODEL_PATH` | `/Users/.../ggml-medium.bin` | モデルファイル |
| `LOCAL_API_WHISPER_TMP_DIR` | `data/tmp` | Whisper 一時ディレクトリ |
| `LOCAL_API_FFMPEG_BIN` | `ffmpeg` | ffmpeg コマンド |
| `LOCAL_API_YTDLP_BIN` | `yt-dlp` | yt-dlp コマンド |
| `LOCAL_API_YTDLP_OUTPUT_DIR` | `data/yt-dlp` | yt-dlp 出力先 |
| `LOCAL_API_RSYNC_BIN` | `rsync` | rsync コマンド |

## n8n からの利用メモ

- Whisper には必ずバイナリデータを送信する（HTTP Request ノードで "Send Binary Data" を ON）。
- `host.docker.internal:5050` を指定してコンテナからホストの API を呼び出す。
