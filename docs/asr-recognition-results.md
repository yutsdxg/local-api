# 認識工程の評価結果（2026-09-24）

[Issue #13](https://github.com/yutsdxg/local-api/issues/13) / [PR #14](https://github.com/yutsdxg/local-api/pull/14) の実測記録。セットアップ、モデルrevision、測定範囲は [評価手順](asr-recognition.md) を参照する。個人音声・文字起こし・詳細manifestはGit対象外の `data/asr/evaluation/` に保存する。

**精度を最優先にした調査対象全体の判断は、[精度重視の全体評価](asr-accuracy-assessment.md)を参照。** 本書のParakeetに関する推奨は速度・メモリ面の評価であり、精度優先の主候補はMLX large-v3とcpp large-v3のgreedy・履歴なし設定へ整理し直した。

## 実装と採用判断

APIの重い変換・認識処理をイベントループの外へ移し、同じAPIプロセス内では1件ずつ実行する。文字起こし中も他の非同期処理が進み、同時要求によるモデルの重複ロードを防ぐ。開始済みの処理は要求キャンセル後もロックと一時ファイルを保持して完了する。複数のUvicornプロセスにまたがる排他や、実行中の子プロセス中断は実装していない。

比較用には共通VAD生成、条件別実行、常駐MLX worker、明示的モデル取得、入力・モデルのハッシュ記録を追加した。モデル常駐workerは評価CLIから使う独立プロセスであり、APIへの常駐バックエンド接続は今回採用しない。APIの入力・`{"text":"..."}`応答、前処理、`medium`とCPUの既定構成は維持する。

正解文の聴取確認が済んでいないため、精度順位やCERは確定できない。Whisperのモデルを大きくするだけでは、今回の長尺・雨音条件を安定して改善できるとは言えない。Parakeetの60秒分割を速度・メモリ面の有力候補、QwenをWhisperと異なる出力を確認する候補として残す。既定モデルの切替には、ローカル比較文書の数字・固有名詞・抜けを原音と照合する必要がある。

アプリケーションを含む次の候補も調べたが、今回の実測順位には含めない。採否はこのAPIとMacでの運用上の判断であり、未測定の精度を評価したものではない。

| 候補 | 調査で確認した点と今回の扱い |
| --- | --- |
| WhisperKit / Argmax OSS | Apple向けSwift実装。OSSにもローカルHTTPサーバーと `/v1/audio/transcriptions` がある。今回は同じWhisper重みを扱うcpp/MLX比較と異なるモデルの検証を先に行い、CoreML変換版を含む追加実測は見送った。[公式](https://github.com/argmaxinc/argmax-oss-swift#local-server) |
| MacWhisper | `mw` CLIからアプリとローカルsocketで通信できる。GUIでの確認には候補になるが、APIの専用常駐処理にはアプリの起動・選択モデル・設定も管理対象となるため、今回は直接ライブラリを検証した。[公式CLI文書](https://docs.macwhisper.com/article/57-macwhisper-command-line-tool) |
| faster-whisper / CTranslate2 | CPUとCUDA向けの選択肢。macOS ARM64 wheelはGPU実行の対象ではなく、このM1 ProのGPUを活用する候補としてはMetal/MLXを優先した。[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[CTranslate2の対応環境](https://opennmt.net/CTranslate2/installation.html) |

外部文字起こしAPIへの送信は今回の対象外である。QwenとParakeetはそれぞれ [Qwen公式モデル](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)、[日本語ParakeetのMLX変換モデル](https://huggingface.co/mlx-community/parakeet-tdt_ctc-0.6b-ja) に対応するローカル資材を使った。一般的なParakeet v3の対応言語と日本語専用モデルを混同しない。

## 条件

- Apple M1 Pro、10 CPU cores、32 GiB、macOS 26.6.2 / arm64。同時に複数のASR推論を走らせず、条件ごとに逐次実行した。OSのファイルキャッシュは消去していない。
- whisper.cpp旧版1.8.2、新版はv1.9.4タグのcommit `927cfce34f31707e17f2bff35c349632fb9e2c3a`（CLI表示1.9.4-dev）。Metal有効、CoreML無効。
- GGMLのmedium / large-v3 / large-v3-turbo、MLX版large-v3、Qwen3-ASR-1.7B BF16、日本語Parakeet `parakeet-tdt_ctc-0.6b-ja` を実行した。量子化条件の比較ではない。
- 通常90秒、雨60秒、長い間60秒の3抜粋。共通Silero後はそれぞれ56.93 / 30.52 / 6.79秒。同じ共通VAD入力を使う条件では音声SHA256を照合した。
- 全文は同じ3録音を形式変換した約248 / 743 / 629秒。全文の共通VADは全文から別途抽出し、抜粋の結果を連結していない。
- 全文の共通VAD後は126.42 / 486.50 / 275.56秒。無音の削除で短縮した時間を、認識エンジン単体の高速化と混同しない。
- 通常のwhisper.cpp条件はthreads 4、beam size 5、best of 5。`-nt`有無を別条件で記録した。MLX Whisperはtemperature 0、previous text無効。Qwenはtemperature 0、max tokens 8192、chunk duration 1200秒、batch 1。Parakeetは全入力を1チャンクとして処理する条件。

## 速度の観測

旧版mediumの内部VAD・`-nt`条件でCPUからMetalだけを変えた測定値。各1回で、CLI起動・モデル読み込み・内部VAD・認識を含む。形式変換は含まない。

| 条件 | 通常90秒 | 雨60秒 | 間60秒 |
| --- | ---: | ---: | ---: |
| 1.8.2 medium CPU | 21.78秒 | 16.77秒 | 7.57秒 |
| 1.8.2 medium Metal | 6.12秒 | 4.36秒 | 2.26秒 |

この3本では3.34〜3.85倍の速度差があった。ただしCPU/Metalで一部の出力も変わったため、文字列が同一になる設定変更とは扱わない。旧版→新版、`-nt`解除にも一貫した出力改善を保証する根拠はなかった。新版CPUの初回は54.31秒と長く、少数測定から版全体の速度順位を付けない。

次は共通VAD済み入力の測定。上段のCLIは毎回起動、下段のworkerはモデルを常駐させた2巡目なので、同じ時間範囲のランキングにはしない。

| 方式・計測範囲 | 通常 | 雨 | 間 |
| --- | ---: | ---: | ---: |
| 新cpp large-v3 Metal / timestamp有効、CLI全体・1回 | 14.78秒 | 7.24秒 | 3.74秒 |
| 新cpp large-v3 Metal / `-nt`、CLI全体・1回 | 8.53秒 | 7.05秒 | 3.75秒 |
| 新cpp turbo Metal / timestamp有効、CLI全体・1回 | 5.11秒 | 3.50秒 | 1.89秒 |
| MLX Whisper large-v3、常駐2巡目 | 7.21秒 | 4.38秒 | 1.48秒 |
| MLX Qwen3-ASR-1.7B、常駐2巡目 | 4.93秒 | 2.72秒 | 0.85秒 |
| MLX日本語Parakeet、常駐2巡目 | 0.86秒 | 0.41秒 | 0.11秒 |

各workerは同じ抜粋を2巡実行し、この範囲では各巡の出力ハッシュが一致した。初回要求はMLX Whisper 10.17 / 4.58 / 1.72秒、Qwen 7.01 / 2.90 / 0.85秒、Parakeet 2.59 / 0.50 / 0.14秒だった。CLIとの差には起動・モデル読み込み時間を除外した効果が含まれる。workerの1巡目と2巡目はどちらも起動時間を除いており、その差には初回演算などの影響もあり得る。

起動からreadyまでは最初の評価時にMLX Whisper 70.10秒、Qwen 20.65秒、Parakeet 4.34秒を要した。別プロセスで再起動したMLX Whisperは3.17秒、Qwenは6.76秒だった。新規インストール直後のimport等とOSキャッシュの影響を分離していないため、最初の値を毎回必要な起動時間とは扱わない。

MLX Whisperとcppのgreedy・履歴条件を近付けた比較も保存したが、完全なデコード同条件ではない。cppのtemperature fallbackは既定の刻み0.2、MLX側はtemperature 0のみで、純粋なエンジン差に帰属できない。

全文の測定値は以下。旧mediumは内部VAD込み、他は共通VAD後で、workerは起動時間を除く。各1回の観測である。

| 条件 | 通常全文 | 雨全文 | 間のある全文 |
| --- | ---: | ---: | ---: |
| 旧cpp medium CPU / `-nt`、CLI全体 | 54.66秒 | 196.82秒 | 103.77秒 |
| 新cpp large-v3 Metal / `-nt`、CLI全体 | 17.48秒 | 77.04秒 | 34.32秒 |
| MLX Whisper large-v3、常駐認識 | 14.20秒 | 53.77秒 | 28.41秒 |
| MLX Qwen3-ASR-1.7B、常駐認識 | 12.32秒 | 53.39秒 | 29.70秒 |
| MLX日本語Parakeet / 一括、常駐認識 | 2.02秒 | 20.58秒 | 10.26秒 |

全文3本を処理したworkerのOS最大RSSはMLX Whisper 3.06 GiB、Qwen 2.21 GiB、Parakeet 2.74 GiBだった。一方、要求ごとのMLX active allocation最大値はそれぞれ3.85 / 8.00 / 14.29 GiBで、特にParakeet一括処理では長さによって大きく増えた。RSSとMLXの値は共有メモリ上で重なり、対象も異なるため合計しない。OS RSSが小さいことを根拠に、モデルの総必要メモリが小さいと判断しない。

Parakeetを `options: {"chunk_duration": 60.0}` にした別条件では、全文3本の認識が1.92 / 8.26 / 4.62秒、MLX active allocationの最大が3.76 / 3.78 / 3.76 GiBとなった。起動は4.06秒、worker全体の最大RSSは2.76 GiB。mlx-audio 0.5.5の非ストリーミング処理の既定で、チャンク間は2秒重複させる。この3本では長尺一括よりメモリを抑えられたため、次の採用候補は60秒分割を推奨する。ただし3本とも一括処理とは出力文字列が変わり、分割境界での語の抜け・重複を含めた精度確認が必要である。

## 全文と出力品質

旧medium CPUと新large-v3 Metal・`-nt`は3本すべて正常終了した。ただし雨の全文ではlarge-v3出力の後半に短文の反復が集中した。句点で区切った同一文が37回連続し、短文の部分文字列としては38回出現する。旧medium出力の後半にある複数の話題に対応する記述が見当たらず、後半脱落の疑いがある。原音との聴取照合前なので、旧medium出力を正解と断定しない。

この出力でも従来の「隣接重複行」は0だったため、今回のrunnerには行内も対象にする文単位の反復診断を追加した。繰り返しの数は警告を見つける補助であり、実際に繰り返した発話や句読点の有無の影響を受ける。正常終了・文字数・反復数0はいずれも精度の保証ではない。

雨全文の追加確認ではlarge-v3のtimestampを有効にしたbeam 5条件でも進捗出力の反復が続いたため、当該評価プロセスを手動停止した。これは品質の一次確認による打切りであり、エンジンの自然な異常終了や、全文処理時間の測定結果として扱わない。`rain-diagnostics/manual-stop.json` に理由、途中ログを保存した。このrunはreport上でexit -15のerrorとして残る。

同じ雨全文でlarge-v3のgreedy・履歴なし（`-bs 1 -bo 1 -mc 0`、timestamp有効）は66.39秒、turbo・beam 5・timestamp有効は53.28秒で完了した。こちらの保存出力には同じ長い反復列が見当たらなかった。大モデルの一律採用や `-nt`解除だけの採用を避け、履歴・探索条件まで含めて評価する必要がある。その後、両条件について通常・長い間の全文も追加確認し、いずれも完了した。各3本を通じて同様の反復列は見当たらなかったが、他の録音も含む解消を保証するものではない。

追加の通常・長い間の全文は、cpp large-v3のgreedy・履歴なしで15.46 / 31.52秒、turboで9.75 / 21.50秒だった。共通VAD後の入力を使い、CLI起動とモデル読み込みを含む。認識内容は[精度重視の全体評価](asr-accuracy-assessment.md)へまとめた。

MLX Whisper、Qwen、日本語Parakeetも3本ずつ全文処理を完了した。これらの出力では同じ種類の句点区切りの連続文反復を検出しなかったが、Parakeetの出力には句点が少なく、この指標の検出力も限られる。Qwenの生成数は全文で325 / 1225 / 686 tokensとなり、設定上限8192には達していない。数字や言い回しのモデル間差は残っており、読みやすさや文字数の増加を正解率の向上とは扱わない。

共通VADの検証として、旧medium CPU・同じデコード設定で内部VADと共通VADを比較し、3抜粋すべての文字起こしSHA256が一致した。これにより今回の抜粋では共通入力化による出力差がないことを確認したが、異なる境界処理まで同一と保証するものではない。

## 安定性と実装の確認

空入力、10秒の無音、10秒の低レベル疑似雑音、0.1秒の正弦波は共通VADで0 frameとなり、cpp large-v3 / MLX Whisper / Qwen / Parakeetの4条件で認識をスキップして空文字を返した。0.7秒の発話抜粋はVADを通過し、4方式すべてが実推論を完了した。これは短尺処理の動作確認であり、発話の完全な聴き取りを採点したものではない。VADを使わず無音を直接モデルへ渡した試験とも区別する。

リポジトリのテストは `.venv/bin/python -m unittest discover -s tests -q` で89件成功した。APIイベントループの進行、同時要求の排他、要求キャンセル後の一時ファイル寿命、workerの起動失敗・誤った応答ID・timeout、CLI割込み時の子プロセス終了、共通PCMの境界、検証済み参照だけのCER計算などを確認した。モデルの精度検証をこのテスト件数に含めない。

## 保存した記録

`data/asr/evaluation/` の以下の各ディレクトリに `report.json`、モデル・入力のハッシュ、TXT、実行ログがある。

| ディレクトリ | 内容 |
| --- | --- |
| `medium-matrix` | 旧/新版 × CPU/Metal × `-nt`有無 × 3抜粋 |
| `whisper-models` | 共通VAD基準、medium / large-v3 / turbo |
| `whisper-mlx` | MLX large-v3、cppの近いgreedy設定、各2巡 |
| `other-models` | Qwen、日本語Parakeet、各2巡 |
| `full-whisper` | 旧mediumと新large-v3・`-nt`の全文 |
| `full-other-models` | MLX Whisper、Qwen、Parakeetの全文 |
| `rain-diagnostics` | 雨全文のtimestamp有効・greedy履歴なし・turboの追加確認。手動打切り1件を含む |
| `smoke` | 共通VADと空入力・無音・雑音・極短音・短い発話の確認 |
| `parakeet-chunk60` | 日本語Parakeetの全文を60秒分割し、一括処理の時間・メモリ・出力と比較 |
| `accuracy-full-completion` | cpp large-v3のgreedy・履歴なしとturboで、通常・長い間の全文を追加確認 |

反復診断追加前の結果は元reportを上書きせず、保存TXTから再計算した値を `diagnostics.json` に記録した。比較用の原音抜粋と文字起こしをまとめた `review.md` もローカルに保存した。

初期の `baseline-cpu` はmacOSサンドボックス内で `/usr/bin/time -l` のsysctlが失敗した診断記録であり、成功した計測には含めない。その後の実測は必要なローカル実行権限の下で行った。モデルダウンロード時も音声を送信せず、認識はすべてローカルで実行した。
