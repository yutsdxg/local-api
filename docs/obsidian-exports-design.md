# Obsidian 結合エクスポート設計

## 目的
Obsidian Vault 内の `inbox` / `journal` 配下のノートを条件で抽出し、用途別に結合した Markdown を Google Drive 同期フォルダへ出力する。NotebookLM で参照可能な Google ドキュメントへ変換する前段として、結合済みの Markdown を生成する。

## 要件（確定事項）
- Vault ルートは `LOCAL_API_OBSIDIAN_VAULT_ROOT` で指定する。
- 出力先は `LOCAL_API_OBSIDIAN_EXPORT_DIR`（例: `/.../My Drive/ObsidianExports`）で指定する。
- 対象ディレクトリ: `inbox` と `journal` のみ。
- 除外タグ: `type/snippet`, `type/account` を含むノートは除外。
- 結合単位:
  - `type/journal` を含むノートは 1 ファイルにまとめる。
  - `topic/*` は直下タグごとに 1 ファイルにまとめる（例: `topic/tools/mac` → `topic/tools`）。
  - `type/journal` と `topic/*` が同時にある場合は **type/journal のみ** に含める。
  - `type/journal` でも `topic/*` でもない場合は `others.md` にまとめる。
- 更新方式: 毎回全件洗い替えで再生成する。
- 結合時はフロントマターは除去し、簡易ヘッダを付ける。
- 並び順: `date` 降順 → ファイル名昇順（新しい順）。

## エンドポイント
- `POST /obsidian/exports/merge`
- パラメータ: なし（Settings で解決）
- 役割: 結合処理を実行し、出力ファイルを更新する。

### レスポンス例
```
{
  "vault_root": "/path/to/vault",
  "output_dir": "/path/to/My Drive/ObsidianExports",
  "generated": [
    {"group": "type/journal", "output_path": ".../journal.md", "note_count": 123},
    {"group": "topic/tools", "output_path": ".../topic_tools.md", "note_count": 45},
    {"group": "others", "output_path": ".../others.md", "note_count": 3}
  ],
  "skipped": {"excluded_tag": 12, "no_target_tag": 8, "parse_error": 1}
}
```

## 設定値（環境変数）
`LOCAL_API_` プレフィックスで指定する。

- `LOCAL_API_OBSIDIAN_VAULT_ROOT`: Obsidian Vault のルート。
- `LOCAL_API_OBSIDIAN_EXPORT_DIR`: 出力先ディレクトリ（Google Drive 同期フォルダ）。
- `LOCAL_API_OBSIDIAN_TARGET_DIRS`: 対象ディレクトリ（デフォルト: `inbox,journal`）。
- `LOCAL_API_OBSIDIAN_EXCLUDE_TAGS`: 除外タグ（デフォルト: `type/snippet,type/account`）。
- `LOCAL_API_OBSIDIAN_JOURNAL_TAG`: ジャーナル判定タグ（デフォルト: `type/journal`）。
- `LOCAL_API_OBSIDIAN_TOPIC_PREFIX`: トピック判定接頭辞（デフォルト: `topic/`）。
- `LOCAL_API_OBSIDIAN_OTHERS_GROUP_NAME`: その他グループ名（デフォルト: `others`）。

## 結合処理の詳細

### 1. 対象ファイルの収集
- `vault_root/<target_dir>` 配下を再帰的に探索し `*.md` を収集。
- それ以外のディレクトリは無視。

### 2. フロントマター解析
- 先頭の `---` ブロックのみを YAML として解析。
- `tags` は `list[str]` を想定（文字列の場合は単一タグとして扱う）。
- パースに失敗した場合は `parse_error` としてスキップ。

### 3. グルーピング
- `type/journal` が含まれる場合 → `type/journal` グループ。
- それ以外で `topic/*` が含まれる場合 → `topic/<直下>` グループ。
- どちらも無ければ `others` グループ。

### 4. 出力ファイル名
- `type/journal` → `journal.md`
- `topic/tools` → `topic_tools.md`
- `others` → `others.md`
- `/` は `_` に変換し、英数字と `_` `-` 以外は `_` に置換。

### 5. ノートの結合フォーマット
- フロントマターは除去し、先頭に簡易ヘッダを付与。

```
---
source: inbox/xxx.md
title: xxx
date: 2025-11-21
tags:
  - topic/tools/mac
---

本文…
```

### 6. 並び順
- `date` 降順 → `filename` 昇順。
- `date` が無い場合は `0000-00-00` 扱いで末尾へ。

## エラー方針
- Vault ルートや出力先ディレクトリが無効 → 400。
- その他予期しないエラー → 500。

## テスト方針
- サービス層テスト: グルーピング、除外、並び順、出力内容。
- エンドポイントテスト: 正常系 / 400 / 500。
