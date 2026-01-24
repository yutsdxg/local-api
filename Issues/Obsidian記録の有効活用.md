---
date: 2026-01-21
tags:
  - topic/ai
  - type/idea
internal_links:
ai_thread:
---
## 背景
- Obsidianでジャーナリングだったりインプット情報のテキスト記録をしているが、イマイチそれらを有効活用できていない
- Codexなどのコーディングエージェントはローカルファイルを見に行けるので、情報を集約して活用できるかもと思っていたが、散らばった情報を検索したりする部分は人間がやっていることと同じなのであまり得意ではないかもと感じているし、一つ一つのタスクに時間がかかっている
- NotebookLMの方がこの目的には合っているのでは
- Obsidianにある情報を全てNotebookLMにインプットしてみたい
- NotebookLMへのインプットの仕方も気になる
	- Noteを分けた方が良いのか？全部一緒が良いのか？
		- Obsidianのファイルはタグをつけているのでそれで分けることはできそう(inbox/タグ構成.md)
	- 一つのNoteにはファイル数制限があるので、Obsidianのファイルは結合などの処理をしてからNotebookLMにインポートする必要がありそう
	- Obsidianのファイルは日々増えていくので、自動的にNoteに反映する仕組みがあると尚良い
		- Obsidianのファイルの結合
		- NotebookLMにインポートされたGoogleドキュメントは自動で最新を読み取ってくれるらしいので、Obsidian→Googleドキュメントで連携すると良いかもしれない（他のアイデアも求む）
- ここまで書いてみて気付いたこと。Obsidianの散らばったファイルをある程度結合しておけばCodexでも十分に扱えるのではないか？

## 結合スクリプト要件（APIサーバ側）
- 目的: Vault内のノートを条件で抽出し、用途別に結合ファイルを生成する
- 対象: Vaultの`inbox`、`journal`、（除外タグ `type/snippet`, `type/account` を含むノートは対象外）
- 出力: ローカルのGoogle Drive同期フォルダ（My Drive配下）に結合ファイルを保存する
- 結合単位:
    - `type/journal` は1ファイルにまとめる
    - `topic/*` は直下タグごとに1ファイルにまとめる
- 更新方式: 毎日実行し、全ファイルを洗い替えで再生成する（差分更新はしない）

### obsidian vault内のディレクトリ構成
```
- inbox
    - xxx.md
    - yyy.md
- journal
    - zzz.md
```
他のディレクトリは現行無視する

### obsidian vault内のマークダウンファイルのサンプル
```
---
date: 2025-11-21
tags:
  - topic/tools/mac
internal_links:
ai_thread:
---
## バックアップ
- Dataディレクトリ
	- Samplepacksはrsyncでバックアップする仕組みを作った
	- まるっと全部コピーして前回分を削除する運用でもよさそう
- Karabiner Elementsの設定
	- [[Karabiner-Elementsスクリプト]]
- /Library/Audio/Presets
	- VSTを使わないようにしたので不要
- /Users/yuts/Library/Audio/Presets
	- VSTを使わないようにしたので不要
- Control Surface Studio
## その他
```

## システム全般の構成（提案）
- 概要: ローカルで結合し、Google Driveを経由してNotebookLMへ取り込む流れにする
- ローカル側:
    - APIサーバに結合スクリプトを配置
    - n8n（Docker）からAPIを実行して結合処理を走らせる
- Google Drive側:
    - 出力先は `My Drive/ObsidianExports` 配下
    - NotebookLMはGoogleドキュメント形式のみ表示される想定
    - `.md` をGoogleドキュメントへ変換する経路を用意する（Google Apps Scriptで自動変換）
- NotebookLM側:
    - Googleドキュメント化されたファイルをソースとして参照
    - 自動更新が必要な場合はDrive側での変換・更新を前提とする
