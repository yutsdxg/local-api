# Local Utility API

Whisper 文字起こし・yt-dlp 音声抽出・コード/メロディ判定をローカルで扱うための FastAPI サーバーです。n8n 等から HTTP 経由で利用できます。

## 前提条件

- Python 3.10
- `whisper-cli`（例: `brew install whisper-cpp`）
- `ffmpeg`
- `yt-dlp`
- モデルファイル（デフォルト: `model/ggml-medium.bin`）

## 事前準備

- 一時/出力ディレクトリを作成: `mkdir -p data/tmp/whisper data/tmp/yt-dlp data/chord-melody/input data/chord-melody/logs logs`
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
| `LOCAL_API_CHORD_MELODY_INPUT_DIR` | `data/chord-melody/input` | 判定対象ディレクトリ |
| `LOCAL_API_CHORD_MELODY_LOG_DIR` | `data/chord-melody/logs` | 解析ログ出力先 |
| `LOCAL_API_CHORD_MELODY_TIME_UNIT` | `0.1` | 分析時間スライス（秒） |
| `LOCAL_API_CHORD_MELODY_POLY_THRESHOLD` | `0.4` | POLY率によるCHORD判定閾値 |
| `LOCAL_API_CHORD_MELODY_POLY_NOTE_COUNT` | `3` | POLY判定とする音数 |
| `LOCAL_API_CHORD_MELODY_STABILITY_THRESHOLD` | `0.7` | CHORD1判定用の最低音安定性閾値 |
| `LOCAL_API_SIMILAR_TONES_CACHE_DIR` | `data/similar-tones/cache` | Hugging Faceモデル/キャッシュ保存先 |
| `LOCAL_API_SIMILAR_TONES_DEVICE` | `cpu` | CLAP実行デバイス（将来のMPS対応用） |

## n8n からの利用メモ

- Whisper には必ずバイナリデータを送信する（HTTP Request ノードで "Send Binary Data" を ON）。
- `host.docker.internal:5050` を指定してコンテナからホストの API を呼び出す。
