# `/whisper` の認識工程を比較する

[Issue #13](https://github.com/yutsdxg/local-api/issues/13) では、前処理の比較に続き、実際に文字起こしする認識工程を評価する。現行の `whisper.cpp` と `medium` を基準に、CPU / Metal、デコード設定、モデル、MLX系ライブラリの違いを切り分ける。日本語の独話、AirPodsによる屋外録音、長い間を含む音声が主な対象になる。

これはローカル評価用の導入文書であり、実測と採用判断は [認識工程の評価結果](asr-recognition-results.md) にまとめる。`/whisper` の既定モデル `medium`、API契約、本番依存関係はこの評価ツールでは変更しない。前処理の採用経緯は [前処理の評価記録](whisper-evaluation.md)、運用設定は [前処理の説明](whisper-preprocessing.md) を参照する。

## 評価環境

評価用Pythonは `data/asr/venv` に分離する。APIの `.venv` や `pyproject.toml` にMLX関連パッケージを追加しない。

| 項目 | 固定・記録する内容 |
| --- | --- |
| 対象環境 | Apple Silicon、macOS 26、Python 3.12。今回の環境記録は macOS 26.6.2 / arm64 / Python 3.12.13 |
| Python依存 | [requirements-asr-evaluation.txt](../scripts/requirements-asr-evaluation.txt)。実環境の `data/asr/environment-freeze.txt` から全パッケージのバージョンを転記 |
| 主なライブラリ | `mlx==0.31.2`、`mlx-metal==0.31.2`、`mlx-whisper==0.4.3`、`mlx-audio==0.5.5` |
| 共通VAD | 現行 `whisper.cpp 1.8.2` の共有ライブラリと Silero v6.2.0。ASR本体に別版を使う条件でも共通VADの版は固定 |
| モデル・実行ファイル | ローカルに準備し、評価レポートにパス、サイズ、SHA256を保存。モデルディレクトリは `.cache` を除く各ファイルを記録 |

requirementsはこのOS・Python・CPUアーキテクチャ向けの版固定スナップショットである。配布wheelのハッシュを固定する形式ではなく、別OSや別Python版の互換性を保証しない。モデル重み、tokenizer、GGMLバイナリも別途必要になる。

以下はリポジトリルートで、新しい評価環境を準備するときだけ実行する。既存の評価環境がある場合はそのPythonを使う。`--no-project` と明示したインストール先により、API用Pythonの要求バージョンや環境と分離する。

```sh
uv venv --no-project --python 3.12.13 data/asr/venv
uv pip sync --python data/asr/venv/bin/python \
  scripts/requirements-asr-evaluation.txt
```

依存とモデルの取得は評価前に済ませる。workerはオフライン設定を有効にし、絶対パスで指定したローカルモデルだけを読み込む。重み以外のtokenizerなどが欠けている場合も、推論中の自動取得に頼らず準備段階で補う。

実験環境は次のように管理する。

| 場所 | 内容 |
| --- | --- |
| `data/asr/python/` | 評価用に取得したPythonランタイムを置く場合の保存先 |
| `data/asr/venv/` | 評価用仮想環境 |
| `data/asr/models/` | 追加のモデル重み・tokenizer |
| `data/asr/vendor/` | 比較用whisper.cppなどのソース・ビルド成果物 |
| `data/asr/hf-cache/` | モデル取得時のキャッシュ |
| `data/asr/evaluation/` | 入力manifest、共通VAD音声、文字起こし、計測ログ、レポート |

個人音声、正解文、文字起こし、個人のファイルパスを含むmanifestはGit対象外の `data/` に保存する。コミットする文書には発話内容や元録音を識別できる個人パスを転記しない。

## 評価モデルを明示的に取得する

[download_asr_models.py](../scripts/download_asr_models.py) は、カタログで選んだ公開モデルを固定revisionから取得する準備用CLIである。APIや推論workerからは呼び出さず、音声を読み込んだりアップロードしたりしない。取得内容だけを確認する `--list` は通信を行わない。

```sh
data/asr/venv/bin/python scripts/download_asr_models.py --list
```

カタログの名前、取得元、revisionは次のとおり。`main` の最新状態ではなく、このcommitを使う。

| `--models` の指定 | Hugging Faceの取得元 | 固定revision |
| --- | --- | --- |
| `whisper-ggml` | `ggerganov/whisper.cpp` | `5359861c739e955e79d9a303bcbc70fb988958b1` |
| `whisper-large-v3-mlx` | `mlx-community/whisper-large-v3-mlx` | `49e6aa286ad60c14352c404340ded53710378a11` |
| `qwen3-asr-1.7b-bf16` | `mlx-community/Qwen3-ASR-1.7B-bf16` | `e1f6c266914abc5a46e8756e02580f834a6cf8a7` |
| `parakeet-ja` | `mlx-community/parakeet-tdt_ctc-0.6b-ja` | `e3810190ff521dcd208bc444a71bf877f3864566` |

次はGGML版WhisperとMLX版Whisperを選ぶ例。QwenやParakeetも必要な場合は、そのカタログ名を `--models` に追加する。モデル指定を省略して全カタログを自動取得する動作はない。

```sh
data/asr/venv/bin/python scripts/download_asr_models.py \
  --models whisper-ggml whisper-large-v3-mlx \
  --output-dir data/asr/models
```

保存先は `<output-dir>/<カタログ名>/`。GGML条件の `model` はその中の `ggml-large-v3.bin` または `ggml-large-v3-turbo.bin` を指定し、MLX系workerの `model` はモデルディレクトリ自体の絶対パスを指定する。カタログには重みだけでなくconfig、tokenizer関連ファイル、READMEなど必要な取得パターンも固定してある。全ファイルの取得ではないため、詳細は `--list` の `files` を確認する。

取得には `huggingface_hub.snapshot_download` を使い、認証tokenを送らない公開モデル取得として実行する。`HF_HOME` 未指定時の取得キャッシュは、既定出力先なら `data/asr/hf-cache` になる。各モデルの `download-provenance.json` に取得元、revision、取得パターン、保存ファイルの相対パス・byte数・SHA256を記録する。`.cache` とprovenanceファイル自身はこの一覧から除く。ハッシュは保存した資材を後から照合するための記録であり、配布元の署名検証を表すものではない。

ダウンロードは同じ保存先を再利用できるが、以前の余分なファイルを削除する機能はない。revisionや取得パターンを変える実験では新しい `--output-dir` を使い、provenanceと評価レポートのファイル一覧を残す。推論前の取得を終えた後はworkerの `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` と絶対ローカルパスで評価する。モデル不足はエラーとして扱い、推論時に別revisionへ置き換えない。

## 比較用whisper.cppを隔離してビルドする

既存の本番whisper.cppを更新せず、比較対象を `data/asr/vendor/whisper.cpp-v1.9.4` に置く。今回の対象はタグ `v1.9.4`、commit `927cfce34f31707e17f2bff35c349632fb9e2c3a`。このcheckoutのCLI表示は `whisper.cpp version: 1.9.4-dev` なので、表示文字列だけで版を同定せずcommitも記録する。

Git、CMake、macOS用C/C++ビルド環境を準備したうえで、新しい保存先へ次のように取得する。既に同じcheckoutがある場合はcloneを繰り返さず、commitとビルド設定を確認する。

```sh
git clone --branch v1.9.4 --depth 1 \
  https://github.com/ggml-org/whisper.cpp.git \
  data/asr/vendor/whisper.cpp-v1.9.4
git -C data/asr/vendor/whisper.cpp-v1.9.4 checkout --detach \
  927cfce34f31707e17f2bff35c349632fb9e2c3a
git -C data/asr/vendor/whisper.cpp-v1.9.4 rev-parse HEAD
```

`rev-parse HEAD` が上記commitと一致することを確認し、Release、Metal有効、CoreML無効でCLIとserverをビルドする。

```sh
cmake -S data/asr/vendor/whisper.cpp-v1.9.4 \
  -B data/asr/vendor/whisper.cpp-v1.9.4/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DWHISPER_COREML=OFF \
  -DWHISPER_BUILD_EXAMPLES=ON \
  -DWHISPER_BUILD_SERVER=ON
cmake --build data/asr/vendor/whisper.cpp-v1.9.4/build \
  --config Release --target whisper-cli whisper-server --parallel 4
data/asr/vendor/whisper.cpp-v1.9.4/build/bin/whisper-cli --version
```

成果物は `build/bin/whisper-cli` と `build/bin/whisper-server`。runnerのwhisper.cpp条件では比較したいCLIの絶対パスを `binary` に設定する。Metalを有効にした同じバイナリでも `-ng` を指定すればCPU条件を作れる。今回のrunnerは毎回CLIを起動する方式であり、serverをビルドしただけではモデル常駐の評価にはならない。共有VADの準備には引き続き1.8.2の共有ライブラリを指定し、1.9.4のライブラリをctypesの既存ABIへ渡さない。

## 同一PCMと共通VADの準備

認識方式の比較では、各方式に同じ16 kHz / mono / PCM16 WAVを渡す。前処理評価で保存した通常90秒、雨音に言及する録音60秒、長い間を含む録音60秒のWAVを再利用できる。ここでの「雨」は録音の選定ラベルであり、測定したSNRや正解ラベルではない。

比較入力には次の2種類がある。名前をmanifestに明示して混同を避ける。

| caseのキー | 音声 | 使用する条件 |
| --- | --- | --- |
| `prepared_audio` | 形式変換後、VADの区間抽出前 | 現行処理の基準など、whisper.cpp内部VADを使う条件 |
| `speech_audio` | 共通VADで区間抽出・連結したWAV | 認識ライブラリ間で同じPCMと同じVAD境界を使う条件 |
| `vad_manifest` | 共通VADが出力した `manifest.json` のパス | 取得元時刻、連結境界、モデルと入力のハッシュを追跡する補助情報 |

共通VAD音声は [prepare_asr_evaluation.py](../scripts/prepare_asr_evaluation.py) で作る。次の `/absolute/path/...` は実在する手元のファイルへ置き換える。出力ディレクトリは空であっても既存なら拒否される。

```sh
data/asr/venv/bin/python scripts/prepare_asr_evaluation.py \
  /absolute/path/to/prepared-before-vad.wav \
  --output-dir data/asr/evaluation/common-vad/normal-new \
  --whisper-lib /absolute/path/to/libwhisper.1.8.2.dylib \
  --vad-model /absolute/path/to/ggml-silero-v6.2.0.bin
```

`prepared.wav` と `manifest.json` が保存され、標準出力にはmanifestの絶対パスが出る。VADはCPUで実行する。共有ライブラリの `whisper_version()` が `1.8.2` 以外なら、構造体ABIを利用する前に停止する。コンテキストとVADの初期値はライブラリのdefault関数から取得し、次の条件を上書きする。

| 設定 | 値 |
| --- | ---: |
| 発話確率の閾値 | 0.5 |
| 最短発話 | 100 ms |
| 最短無音 | 500 ms |
| 発話前後の余白 | 200 ms |
| 最後以外の検出区間に加える末尾音声 | 最大100 ms |
| 連結区間の間に挿入するゼロ音声 | 100 ms |

PCM16のサンプル値は再量子化せずにコピーする。manifestには入力・出力WAV、共有ライブラリ、VADモデルのSHA256、実設定、検出時のcentisecond区間、コピー後の元・出力sample範囲、挿入無音を記録する。sample範囲は半開区間 `[start, end)` で、コピー音声は元時刻と1対1に対応し、挿入無音には元時刻を割り当てない。元ファイル全体からの抜粋なら、その開始時刻は別途caseの `start_seconds` に記録する。

現行whisper.cpp内部処理との次の差は、評価用manifestにも明記する。

- 1.8.2内部処理には、確保長の計算では `n_samples - 1` まで、実コピーでは `n_samples` までを使う不一致がある。評価スクリプトは一貫した排他的終端を使い、EOFに達した区間ごとの1 sample分の不足を再現しない。本番whisper.cppの修正は行わない。
- 内部時刻補間は延長前のVAD時刻を使うが、評価用mappingは追加100 msを含む実際のコピー元sampleを記録する。
- WAV外へ丸められた終端はファイル境界へ切り詰め、長さ0の区間は省略する。元入力が空なら `empty_input`、発話が残らなければ `no_speech` とし、どちらも有効な0 frame WAVを出力する。

0 frame入力ではrunnerが両方式ともデコードをスキップし、空textと `skipped_empty: true` を記録する。worker条件ではモデル起動が先に完了しているため、起動時間まで0になるわけではない。workerを直接呼ぶ場合は空音声をエラーにする。`empty_input` / `no_speech` は準備結果として保存し、認識性能の計測対象には含めない。

共通VADと内部VADの結果一致は別途確認する。両者の波形が完全一致すると仮定せず、同じmedium・同じデコード設定で得た文字起こしの一致/非一致も記録する。`speech_audio` を渡す条件では追加のVADを有効にしない。

## manifestで比較条件を指定する

[evaluate_asr.py](../scripts/evaluate_asr.py) は、JSONの `cases` と `conditions` を順に実行する。パスは絶対パスで記載する。以下は書式例であり、モデルの種類・保存先は実際に準備したものに置き換える。

```json
{
  "cases": [
    {
      "name": "normal",
      "language": "ja",
      "prepared_audio": "/absolute/path/to/prepared-before-vad.wav",
      "speech_audio": "/absolute/path/to/common-vad/prepared.wav",
      "vad_manifest": "/absolute/path/to/common-vad/manifest.json",
      "start_seconds": 0.0,
      "duration_seconds": 90.0,
      "reference": {
        "path": "/absolute/path/to/reference.txt",
        "human_verified": false
      }
    }
  ],
  "conditions": [
    {
      "name": "medium-cpu-common-vad",
      "backend": "whisper.cpp",
      "binary": "/absolute/path/to/whisper-cli",
      "model": "/absolute/path/to/ggml-medium.bin",
      "input_key": "speech_audio",
      "vad_mode": "common-silero",
      "args": ["-ng", "-nt", "-np", "-t", "4", "-bs", "5", "-bo", "5"]
    },
    {
      "name": "mlx-whisper-common-vad",
      "backend": "mlx-whisper",
      "python": "/absolute/path/to/data/asr/venv/bin/python",
      "model": "/absolute/path/to/local-mlx-whisper-model",
      "input_key": "speech_audio",
      "vad_mode": "common-silero",
      "options": {
        "temperature": 0.0,
        "condition_on_previous_text": false
      }
    }
  ]
}
```

`name` は英数字・`_`・`-` のみを使い、case内・condition内でそれぞれ重複させない。`reference` は任意で、`human_verified` が `false` の例ではファイルを正解文として読まない。`duration_seconds` などのcase補助情報は記録用であり、runner自身が抜粋を作り直す機能はない。

whisper.cpp条件ではrunnerが `-m`、`-f`、`-otxt`、`-of`、`-l` を設定し、末尾に `args` を追加する。出力先や入力を変える引数を `args` に重ねない。内部VADを比較する条件は `input_key: "prepared_audio"` とし、`args` にVADモデル・閾値・最短発話・最短無音・余白を明記する。CPU / Metal、`-nt` の有無、バージョン、モデルを同時に変えず、独立したconditionとして比較する。

`input_key` の既定はwhisper.cppで `prepared_audio`、workerで `speech_audio` なので、公平な共通入力比較では必ず `input_key: "speech_audio"` と `vad_mode: "common-silero"` を明示する。`common-silero` 条件ではcaseの `vad_manifest` が必須になる。`speech_audio` と `vad_manifest` が指定されている場合、runnerは入力WAVのSHA256とframe数をVAD manifestの出力記録と照合し、不一致をエラーにする。VAD manifest自体のSHA256も実行結果へ保存する。

runnerは選択されたWAVの形式、長さ、SHA256を実行ごとに記録するが、異なるconditionの入力一致を強制しない。比較時にハッシュを確認する。認識結果の時刻をVAD mappingで原録音へ自動変換する機能はなく、mappingの利用は評価結果の確認側で行う。

例として、このJSONを `data/asr/evaluation/manifest-example.json` に保存した場合は次のように実行する。

```sh
data/asr/venv/bin/python scripts/evaluate_asr.py \
  data/asr/evaluation/manifest-example.json \
  --output-dir data/asr/evaluation/comparison-new \
  --only medium-cpu-common-vad mlx-whisper-common-vad \
  --cases normal --repeat 2 --timeout 1200
```

`--only` / `--cases` は省略時に全条件・全caseを実行する。`--repeat` の既定は1、`--timeout` は1200秒。1つのrunner内では条件、反復、caseの順に逐次実行する。並行負荷を避けるため、別のrunnerやASR処理も同時に起動しない。出力ディレクトリは既存なら拒否される。

`report.json` は条件・モデル・入力manifest・各実行のハッシュ、所要時間、状態を保存し、各実行後に更新する。文字起こしは `<condition>/<case>-<repeat>/transcript.txt`、whisper.cppログは同ディレクトリの `process.log`、workerログはconditionディレクトリの `worker.log` に出る。失敗も結果として残し、いずれかの条件や実行が失敗した場合は終了コード1になる。

## 常駐workerの契約

[asr_worker.py](../scripts/asr_worker.py) は1モデルを読み込んだまま、標準入力のJSON Linesを逐次処理する。通常はrunnerが起動する。直接確認する場合のCLIは次のとおり。

```sh
data/asr/venv/bin/python scripts/asr_worker.py \
  --backend mlx-whisper --model /absolute/path/to/local-mlx-whisper-model
```

`--backend` は `mlx-whisper`、`qwen`、`parakeet`。モデル読み込み後に `ready` が出る。その後、1行につき1要求を渡し、同じ `id` の `result` または `error` を受け取ってから次へ進む。

```json
{"id":"normal-1","audio_path":"/absolute/path/to/common-vad/prepared.wav","language":"ja","options":{"temperature":0.0,"condition_on_previous_text":false}}
{"action":"shutdown"}
```

この2行はmlx-whisper用の要求と終了要求を示す。`audio_path` とモデルは絶対ローカルパスが必要で、入力は空でない16 kHz / mono / PCM16 WAVに限る。`language` の既定は `ja`、`auto` または `null` は自動判定。Qwenと今回の日本語Parakeet条件では日本語と自動判定だけを受け付ける。`prompt` は任意文字列で、mlx-whisperにはinitial prompt、Qwenにはsystem promptとして渡す。Parakeetにはプロンプトを指定しない。

| backend | 指定できる `options` と既定値 |
| --- | --- |
| `mlx-whisper` | `temperature: 0.0`、`condition_on_previous_text: false` |
| `qwen` | `temperature: 0.0`、`max_tokens: 8192`、`chunk_duration: 1200.0`、`min_chunk_duration: 1.0` |
| `parakeet` | `chunk_duration: null` |

未知のoptionはエラーになる。worker自身はVADやノイズ抑制を行わない。Qwenの `segments` は入力チャンクの時刻であり、語や発話のalignmentではない。Whisperはsegment、Parakeetはsentence単位の時刻を返し、いずれも入力WAV上の相対時刻になる。原録音へ戻すには共通VAD manifestと抜粋開始時刻を使う。方式間のtimestamp粒度を同じ精度の指標として比較しない。

## 時間・メモリの測定範囲

測定値は以下のscopeと一緒に扱う。モデル常駐の有無とVADの有無を混ぜた1つの速度順位にまとめない。

| 値 | 含まれる範囲 |
| --- | --- |
| whisper.cppの `elapsed_seconds` | CLI起動、モデル読み込み、VADが有効ならVAD、デコードを含む壁時計時間。各実行で新しいプロセス |
| workerの `startup_seconds` | プロセス起動から `ready` 受信まで。importとモデル読み込みを含む |
| `worker_ready.load_seconds` | worker内部でbackendを構築する時間。Pythonプロセス起動そのものは含まない |
| workerの `elapsed_seconds` | 各要求のWAV読込、PCMの変換、推論、結果作成。モデルの初期読み込みは含まない |
| workerの `wall_seconds` | runnerからの要求送信から応答受信まで。IPCの時間も含む。送信前の入力ハッシュ計算は含まない |
| `peak_process_rss_bytes` | macOSの `/usr/bin/time -l` による、そのCLI実行の最大RSS |
| `peak_worker_process_rss_bytes` | workerプロセス全体の最大RSS。モデル読み込みと全case・全反復を含むcondition単位の値 |
| workerの `peak_memory_bytes` | 要求ごとにpeakをリセットしたMLXのactive allocation。ホストRSSやallocator cacheは含まない |

常駐workerの初回要求には初回演算などの影響が残り得る。2回目以降は同じモデルを保持したwarm実行になるが、case間の順序やキャッシュも記録して解釈する。ここでのcoldは新規プロセス/初回要求という区別であり、OSのファイルキャッシュを消した状態を保証しない。`--repeat` を増やしてもwhisper.cppは毎回CLIを起動し直す。

MLXのpeakには常駐モデルのactive allocationも含まれ得るため、「要求で増えたメモリ量」や「GPU全体の使用量」とは扱わない。Apple Siliconの共有メモリ上ではMLX割当とOS RSSが重なるため、両者を加算しない。RSSを取得できない環境では値は `null` になり、0 byteとして比較しない。

RTFを算出するときは `経過秒 / 入力音声秒` とし、分母が共通VAD後の秒数か原抜粋の秒数かを記載する。0秒の音声では算出しない。準備工程のVAD、ハッシュ計算、モデル取得の時間は認識時間に含めないため、API全体の応答時間とは別に扱う。

## 正解文と精度の解釈

CERを出すのは、caseの `reference.human_verified` がJSONの `true` で、原音を人が確認した正解文がある場合だけにする。以前のASR出力や文脈から推測した文章にはこのラベルを付けない。同じcaseでは各条件を同じ正解文と比較し、VADで落ちた発話も評価から都合よく除かない。

CERは参照・仮説の両方にNFKCを適用し、空白文字だけを除去してから文字単位のLevenshtein距離を計算し、正規化後の参照文字数で割る。句読点、数字、符号は削除しない。例えば `12.5` と `125`、`-3` と `3` は区別する。参照が正規化後に空ならCERは `null` とし、0%とは扱わない。正解ファイルのSHA256と文字数も結果に残す。

正解文がないcaseではCERを出さず、文字数、空でない行数、隣接する重複行数を診断情報として記録する。さらにNFKC・空白除去後に `。!?` で区切り、文数、同一文の最大出現回数、最大連続回数を数える。小数点は区切りに使わない。行内の反復も検出できるが、実際の発話の繰り返しも数えるため、精度の点数や自動棄却の根拠にはしない。自然な文章、短い出力、速い処理を認識精度の向上とはみなさない。短い語の欠落、語尾、数値、固有名詞、無音中の誤生成、長い反復、長尺での打ち切りを原音で確認して採用を判断する。Qwenが生成token上限に達した場合のwarningも確認対象になる。

## 機能確認

次のテストはASRモデルのダウンロードや実推論を行わず、CLI契約、PCM区間抽出、例外処理、計測・採点の扱いを確認する。

```sh
.venv/bin/python -m unittest \
  tests.test_prepare_asr_evaluation \
  tests.test_asr_worker \
  tests.test_evaluate_asr
```

実録音での結果と採用判断は、入力・モデル・設定のハッシュと上記の制限を伴って記録する。本書の手順や依存スナップショットの追加だけを根拠に、既定のmediumを別モデルへ切り替えない。
