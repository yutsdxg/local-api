# `/whisper` の音声前処理と比較評価

AirPods などの録音について、無音・雑音対策を同じ原音で比較するための機能です。既定は従来の `legacy` のままです。`vad` と `deepfilter` は設定で選択します。API の入力と応答形式 `{"text": "..."}` は変わりません。

## 処理の選択

| `LOCAL_API_WHISPER_PREPROCESSING` | 処理 |
| --- | --- |
| `legacy`（既定） | 120 Hz high-pass → 8 kHz low-pass → `dynaudnorm=f=200:g=7` → 音量閾値による `silenceremove` → 16 kHz / mono / PCM s16le → Whisper |
| `vad` | 16 kHz / mono / PCM s16le → Silero VAD → Whisper |
| `deepfilter`（実験用） | 48 kHz / mono / PCM s16le → DeepFilterNet3 → 16 kHz 化 → Silero VAD → Whisper |

`vad` / `deepfilter` では既存の `silenceremove` を使わず、whisper.cpp の `--vad` と専用の Silero モデルで発話区間を検出します。VAD は発話区間の検出、DeepFilterNet3 は発話に重なる雑音の抑制を担当します。

`LOCAL_API_WHISPER_NORMALIZE=true` で、`vad` / `deepfilter` に `dynaudnorm=f=200:g=7` を追加できます。既定は `false` です。`deepfilter` の正規化はノイズ抑制後に行います。`legacy` は従来どおり常に正規化するため、この設定では変化しません。

ノイズを減らして聴きやすくなっても、文字起こしが改善するとは限りません。屋外の風・交通音を含む実録音で、発話の欠落、無音中の余計な出力、処理時間を比較してから使用する設定を決めます。

[2026-09-23の実録音評価](whisper-evaluation.md)では、試用候補を `vad`・正規化なしとしました。DeepFilterNetの追加効果は確認できず、既定は `legacy` を維持しています。

## 設定

環境変数はすべて `LOCAL_API_` 接頭辞付きです。相対パスはサーバーの作業ディレクトリを基準にします。

| 環境変数 | 既定値 |
| --- | --- |
| `LOCAL_API_WHISPER_PREPROCESSING` | `legacy` |
| `LOCAL_API_WHISPER_NORMALIZE` | `false` |
| `LOCAL_API_WHISPER_VAD_MODEL_PATH` | `data/models/ggml-silero-v6.2.0.bin` |
| `LOCAL_API_WHISPER_VAD_THRESHOLD` | `0.5` |
| `LOCAL_API_WHISPER_VAD_MIN_SPEECH_DURATION_MS` | `100` |
| `LOCAL_API_WHISPER_VAD_MIN_SILENCE_DURATION_MS` | `500` |
| `LOCAL_API_WHISPER_VAD_SPEECH_PAD_MS` | `200` |
| `LOCAL_API_WHISPER_DEEPFILTER_BIN` | `data/models/deepfilternet/deep-filter-0.5.6-aarch64-apple-darwin` |
| `LOCAL_API_WHISPER_DEEPFILTER_MODEL_PATH` | `data/models/deepfilternet/DeepFilterNet3_onnx.tar.gz` |
| `LOCAL_API_WHISPER_DEEPFILTER_ATTENUATION_LIMIT_DB` | `12` |

VAD の最短発話時間を上げすぎると短い返事を除外しやすくなり、余白を小さくすると語頭・語尾が切れやすくなります。上記は比較開始用の値です。DeepFilterNet の抑制上限は `0.01` dB 以上の有限値を指定します。小さい値ほど原音を多く残します。追加の post-filter は使用しません。

例: VAD を有効にして起動します。

```sh
LOCAL_API_WHISPER_PREPROCESSING=vad uv run uvicorn app.main:app --host 0.0.0.0 --port 5050
```

`LOCAL_API_WHISPER_PREPROCESSING=legacy` に戻すと従来の処理になります。選択した処理に必要なモデルや実行ファイルがない場合はエラーになります。

`LOCAL_API_WHISPER_ARGS` は最後に追加されます。同じ VAD オプションをここにも指定すると、上記専用設定より追加引数が優先されます。比較時は既定の `-ng -nt -np` を維持し、VAD は専用設定で調整してください。

## モデル・実行ファイルの準備

既存の FFmpeg、Whisper モデルと、`--vad` に対応した `whisper-cli` を使用します。以下はリポジトリルートで実行する取得例です。API はモデルの自動取得や依存パッケージのインストールを行いません。

Silero VAD v6.2.0 の取得先は [whisper.cpp 公式の取得スクリプト](https://github.com/ggml-org/whisper.cpp/blob/master/models/download-vad-model.sh)で使用している `ggml-org/whisper-vad` です。

```sh
mkdir -p data/models
curl --fail --location --proto '=https' --tlsv1.2 \
  --output data/models/ggml-silero-v6.2.0.bin \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
shasum -a 256 data/models/ggml-silero-v6.2.0.bin
```

DeepFilterNet は [公式 v0.5.6 リリース](https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6)の Apple Silicon 用 Rust CLI と、[同タグの DeepFilterNet3 モデル](https://github.com/Rikorose/DeepFilterNet/tree/v0.5.6/models)を使用します。Python ライブラリとしてのインストールは不要です。次の実行ファイルは macOS arm64 専用です。

```sh
mkdir -p data/models/deepfilternet
curl --fail --location --proto '=https' --tlsv1.2 \
  --output data/models/deepfilternet/deep-filter-0.5.6-aarch64-apple-darwin \
  https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-aarch64-apple-darwin
curl --fail --location --proto '=https' --tlsv1.2 \
  --output data/models/deepfilternet/DeepFilterNet3_onnx.tar.gz \
  https://raw.githubusercontent.com/Rikorose/DeepFilterNet/v0.5.6/models/DeepFilterNet3_onnx.tar.gz
shasum -a 256 data/models/deepfilternet/deep-filter-0.5.6-aarch64-apple-darwin \
  data/models/deepfilternet/DeepFilterNet3_onnx.tar.gz
```

取得元と下記の SHA256 を確認後、実行権限を付けてバージョンを確認します。

```sh
chmod u+x data/models/deepfilternet/deep-filter-0.5.6-aarch64-apple-darwin
data/models/deepfilternet/deep-filter-0.5.6-aarch64-apple-darwin --version
```

出力は `deep_filter 0.5.6` です。以下の SHA256 は今回取得したファイルから計算した値で、配布元の署名検証を表すものではありません。

| ファイル | SHA256 |
| --- | --- |
| `ggml-silero-v6.2.0.bin` | `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987` |
| `deep-filter-0.5.6-aarch64-apple-darwin` | `4601e7f4e4c03e59a4c5b5000216ef3add3e808799cfccd95e14e83ea4611081` |
| `DeepFilterNet3_onnx.tar.gz` | `c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616` |

### DeepFilterNet3 の遅延と末尾処理

今回のモデルは 48 kHz、hop 480 samples（10 ms）、遅延 1,440 samples（30 ms）です。サービスは末尾 1,920 samples（40 ms）を逆順に付加し、CLI の `-D` で遅延補正した後、元のサンプル数まで切り戻してから 16 kHz に変換します。短い音声では反射した末尾を繰り返して補います。

v0.5.6 は極小音量のフレームを内部でスキップするため、単純な無音の付加では末尾が欠落する場合があり、反射paddingを採用しています。また、抑制上限 `0.01` dB 未満は素通し処理となって `-D` と整合しないため許可しません。この補正は上記 DeepFilterNet3 モデル専用です。異なるモデル・CLI バージョンへ変更するときは遅延と末尾を再検証してください。発話を含む原音内部の極小音量区間への影響も含め、`deepfilter` は実験用として扱います。[参照実装](https://github.com/Rikorose/DeepFilterNet/blob/v0.5.6/libDF/src/tract.rs)

## 同じ抜粋で比較する

録音全体の原音ファイルを入力し、開始位置と長さを指定します。評価スクリプトは原音のサンプルレート・チャンネル数と最大32-bitの整数精度を保った同一抜粋を一度作り、各処理をその抜粋から個別に実行します。従来の処理済み 16 kHz ファイルから比較を始めると、除去済みの帯域や無音を戻せないため比較条件が変わります。

```sh
uv run python scripts/evaluate_whisper.py '/path/to/original.m4a' \
  --start 60 --duration 45 \
  --profiles legacy vad vad_normalized deepfilter deepfilter_normalized \
  --existing-transcript '/path/to/existing.txt' \
  --output-dir data/evaluations/whisper-comparison-01
```

`--existing-transcript` は省略できます。既存 TXT は過去の比較資料であり、正解文として扱いません。`vad_normalized` / `deepfilter_normalized` は評価スクリプト内の呼び名で、API の前処理設定はそれぞれ `vad` / `deepfilter` と `LOCAL_API_WHISPER_NORMALIZE=true` の組み合わせです。`--profiles` 省略時は `legacy vad vad_normalized deepfilter` を比較します。

出力先には原音の抜粋 `source.wav`、各処理の `transcript.txt` / `process.log` / 変換済み WAV、設定・ハッシュ・処理時間を記録した `report.json` が残ります。既存の出力ディレクトリは再利用せず、比較ごとに新しい名前を指定します。実録音と結果はローカルに保存され、外部サービスへ送信しません。`data/` は `.gitignore` 対象です。

文字数や連続した重複行の数は診断用であり、認識精度の点数ではありません。聞き取りで作った正解文がある場合に CER などを評価します。特に短い発話、語頭・語尾、長い無音、風の強い区間を聞き比べてください。抜粋評価は全文処理と文脈・フィルター境界が異なるため、採用候補は最後に全文でも確認します。
