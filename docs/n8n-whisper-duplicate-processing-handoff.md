# n8n Whisper重複処理・エラー事象 引き継ぎ

- 作成日: 2026-08-27（JST）
- 対象ワークフロー: `journal-voice-to-text`
- 調査対象のエクスポート: `/Users/yuts/Downloads/journal-voice-to-text.json`
- 関連API: `/Users/yuts/Data/Dev/local-api` の `POST /whisper`

## 1. 要約

`journal-voice-to-text` ワークフローでは、Dropboxの `/Temporary/Journal Audio` に置かれた音声ファイルを10分間隔で列挙し、ローカルのWhisper APIで文字起こししている。

音声ファイルを `/Temporary/Backup` へ移動するのは、Whisper、2回のOpenAI処理、Obsidianへの追記がすべて完了した後である。そのため、1回の処理が10分を超えると、次のスケジュール実行が元フォルダに残っている同じファイルを再取得する。

2026-08-27朝のローカル一時ファイルをハッシュで調査したところ、4つのユニークな音声が合計21回処理されていた。同一音声ごとの処理回数は8回、5回、5回、3回だった。したがって、複数ファイルを処理しているだけでなく、同じファイルを複数のワークフロー実行が重複処理していることが確認できた。

主な対策箇所はn8nワークフローである。ファイルを選択した直後にDropbox上のProcessingフォルダへ移動して処理対象を取得済み状態にし、その後に長時間処理を実行する必要がある。

## 2. 現在のワークフロー

確認した処理順は次のとおり。

```text
Schedule Trigger（10分間隔）
  → List a folder（/Temporary/Journal Audio）
  → Download a file
  → Edit Fields
  → WhisperApi
  → GenerateText
  → GenerateTags
  → ObsidianContent
  → Convert to File
  → Read/Write Files from Disk（Obsidianへ追記）
  → Move a file（/Temporary/Backupへ移動）
```

重要な設定は以下。

| 項目 | 現在の値・状態 |
| --- | --- |
| スケジュール | 10分間隔 |
| 監視元 | `/Temporary/Journal Audio` |
| Whisper API | `POST http://host.docker.internal:5050/whisper` |
| HTTPタイムアウト | 600,000ミリ秒（10分） |
| 処理済みファイルの移動 | ワークフロー末尾 |
| ワークフローのエクスポート時状態 | `active: true` |

## 3. 確認できた事象

### 3.1 同一音声の重複処理

2026-08-27朝に生成されたWhisper入力WAVをSHA-1で比較した結果、次の重複を確認した。

| 同一内容のグループ | 処理回数 |
| --- | ---: |
| 音声A | 8回 |
| 音声B | 5回 |
| 音声C | 5回 |
| 音声D | 3回 |
| 合計 | 21回 |

一時WAVの生成時刻は、おおむね10分間隔のスケジュール実行と、その実行内での複数アイテム処理に対応していた。

### 3.2 Whisper処理の滞留

調査時点のLocal APIは、次のコマンドによる単一Uvicornワーカーで動作していた。

```text
uv run uvicorn app.main:app --host 0.0.0.0 --port 5050
```

APIの `async` エンドポイント内では、`ffmpeg` と `whisper-cli` を同期的な `subprocess.run()` で実行している。単一ワーカーのイベントループが処理中にブロックされるため、後続リクエストは実質的に順番待ちになる。

この状態でn8nから複数のリクエストや重複実行が到着すると、後続リクエストがHTTP Requestノードの10分タイムアウトに達する可能性が高い。

### 3.3 一時ファイルの蓄積

調査時点で、`data/tmp/whisper` に以下のファイルが存在した。

- Whisper入力WAV: 90個
- Whisper出力TXT: 89個
- 使用量: 約2.8GB

入力と出力の差分1件は、確認時に実行中だった `whisper-cli` のセッションと一致していた。

Local APIはUUIDを使用して入力WAVと出力TXTの名前を分離しているため、複数リクエスト間の一時ファイル名衝突は確認されていない。一方、処理終了後にWAVとTXTを削除していないため、ファイルは継続的に蓄積する。

ホストのデータボリュームは調査時点で使用率98%、空き約24GiBだった。直ちに容量不足が今回の直接原因とまでは断定できないが、今後 `ffmpeg` やWhisperが書き込みに失敗する運用リスクがある。

## 4. 想定される失敗の流れ

```text
時刻 T
  実行1が音声A・B・CをListする
  音声はJournal Audioフォルダに残ったまま処理される

時刻 T+10分
  実行1はまだ処理中
  実行2が同じ音声A・B・Cを再びListする

以降
  Whisper APIへの処理要求が重複・滞留する
  後続リクエストが10分でタイムアウトする可能性がある
  先行実行がObsidianへ書き込み、DropboxファイルをBackupへ移動する
  後続実行も同じ内容をObsidianへ追記する可能性がある
  後続実行のDropbox移動時には、元ファイルが存在せず失敗する可能性がある
```

## 5. 原因の評価

### 確認済みの主原因

処理対象ファイルを取得済み状態へ変更するタイミングが遅く、スケジュール実行間で同じDropboxファイルが再取得されている。

### 事象を増幅している要因

- 1回のListで複数ファイルが返る。
- Whisperと後続OpenAI処理に時間がかかる。
- スケジュール間隔とHTTPタイムアウトがともに10分である。
- Local APIが単一ワーカーで同期的にWhisperを実行している。
- Local APIの一時ファイルが削除されない。

### 現時点で主原因ではないもの

Local APIはリクエストごとにUUIDを生成しているため、一時ファイル名の衝突が今回の重複処理を発生させているわけではない。

## 6. n8n側の推奨対策

### 6.1 推奨ワークフロー

```text
Schedule Trigger
  → List a folder（Journal Audio）
  → 対象を1件選択、または Loop Over Items（Batch Size: 1）
  → Move（Journal Audio → Processing）
  → Download（Processing側の新しいパスから取得）
  → WhisperApi
  → GenerateText
  → GenerateTags
  → Obsidianへ書き込み
  → Move（Processing → Backup）
  → 次のアイテム
```

推奨するDropboxフォルダ構成例。

```text
/Temporary/Journal Audio   未取得
/Temporary/Processing      処理中・取得済み
/Temporary/Backup          処理成功
/Temporary/Error           処理失敗
```

### 6.2 Processingへの移動を先に行う理由

`Loop Over Items` のBatch Sizeを1にするだけでは、同一ワークフロー内の処理は直列化できても、10分ごとに開始する別のワークフロー実行との重複は防げない。

長時間処理の前にファイルをProcessingへ移動することで、次のスケジュール実行がJournal AudioをListした際に同じファイルを取得しなくなる。

複数の実行がほぼ同時に同じファイルをListした場合は、ProcessingへのMoveに最初に成功した実行だけが処理を継続する設計が必要である。後続実行の「元ファイルが存在しない」エラーは、別実行が取得済みであることを示す競合として扱い、そのアイテムを終了させる。

### 6.3 失敗時の扱い

- Whisperまたは後続処理が失敗した場合は、Processingに残すかErrorへ移動する。
- 自動再試行する場合は、同一ファイルがすでに処理成功していないことを識別できるようにする。
- Obsidianへの追記後、Backup移動だけが失敗したケースでは、再実行による二重追記を防ぐ必要がある。
- ファイルパス、Dropbox ID、コンテンツハッシュなどを処理済みキーとして保存できると、より確実に冪等性を担保できる。

### 6.4 応急処置

恒久対策を反映するまで、以下を検討する。

1. ワークフローを一時停止し、実行中・待機中の処理を確認する。
2. 重複実行のキューが解消するまで、失敗実行を安易にRetryしない。
3. スケジュール間隔を、最長の処理時間より十分長い値へ延長する。

スケジュール間隔の延長は、処理時間がさらに長くなれば再発するため、恒久対策にはならない。

### 6.5 対策として不十分な変更

- HTTPタイムアウトだけを延長する。
- `Loop Over Items` だけを追加する。
- スケジュール間隔だけを延長する。
- BackupへのMoveを現在の末尾に残したままにする。

これらは滞留や発生頻度を減らす可能性はあるが、実行間で同じファイルを再取得できる状態を解消しない。

## 7. Local API側で推奨される補助対策

n8n側が根本対策であり、以下はLocal API側の防御・運用改善として別途検討する。

1. Whisper処理に明示的なSemaphoreまたはLockを設け、同時実行数を1に制限する。
2. 無制限に待たせず、キュー上限超過時は429または503を返す。
3. `subprocess.run()` をワーカースレッド等へ逃がし、FastAPIのイベントループをブロックしない。
4. `finally` で入力WAVと出力TXTを削除する。
5. `ffmpeg` と `whisper-cli` の標準エラー出力をログへ保存する。
6. n8nへ返すエラーに、失敗工程と診断可能な範囲の詳細を含める。
7. Whisperプロセス自体にもタイムアウトを設定する。

## 8. 受け入れ条件案

n8n側の修正後、少なくとも次を確認する。

- Journal Audioに複数ファイルを置いても、各ファイルのWhisper呼び出しが1回だけになる。
- 1件の処理時間が10分を超えても、次のスケジュール実行が同じファイルを処理しない。
- Whisper開始前に対象ファイルがProcessingへ移動している。
- 成功時はProcessingからBackupへ移動する。
- 失敗時はProcessingまたはErrorに残り、Journal Audioへ暗黙に戻らない。
- 同一ファイルのObsidian本文が二重追記されない。
- DropboxのMove競合が発生しても、別実行による取得済みとして安全に終了できる。
- n8nの実行履歴で、実行中・成功・失敗と対象ファイルを追跡できる。

## 9. 未確認事項

- n8nに表示された実際のエラー本文は共有されていない。
- エラーがHTTPタイムアウト、Whisperの500応答、Dropbox Move失敗のどれだったかは未確定。
- n8nのバージョン、実行モード、インスタンス側の同時実行数設定は未確認。
- 重複した内容がObsidianファイルへ実際に何回追記されたかは未確認。
- Dropbox側に現在残っているJournal Audio、Processing、Backupの状態は未確認。
- 添付JSONではワークフローがActiveだが、調査後の現在状態は未確認。

引き継ぎ先では、まずn8nの該当Executionを開き、失敗ノード、エラー全文、開始・終了時刻、対象ファイル名を確認すること。これにより、確認済みの重複処理に加えて、今回報告されたエラーの直接の発生箇所を確定できる。

## 10. 調査時に変更していないもの

- n8nワークフローは変更していない。
- Local APIのコードは変更していない。
- 実行中のn8nまたはWhisperプロセスは停止していない。
- `data/tmp/whisper` の一時ファイルは削除していない。
- Obsidianの重複記録は削除・修正していない。
- Dropbox上のファイルは移動していない。

## 11. 関連ファイル

- n8nワークフローエクスポート: `/Users/yuts/Downloads/journal-voice-to-text.json`
- Whisper APIエンドポイント: `/Users/yuts/Data/Dev/local-api/app/main.py`
- Whisper処理実装: `/Users/yuts/Data/Dev/local-api/app/services/whisper.py`
- Local API設定: `/Users/yuts/Data/Dev/local-api/app/config.py`
- Whisper一時ディレクトリ: `/Users/yuts/Data/Dev/local-api/data/tmp/whisper`
