from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.config import Settings
from app.services import obsidian_exports


class TestObsidianExportsService(unittest.TestCase):
    def test_merge_exports_raises_when_target_dir_missing(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            vault = base / "vault"
            export_dir = base / "exports"
            vault.mkdir(parents=True)

            settings = Settings(
                whisper_bin="whisper",
                whisper_model_path="model",
                ffmpeg_bin="ffmpeg",
                whisper_tmp_dir=base / "tmp/whisper",
                ytdlp_bin="ytdlp",
                ytdlp_output_dir=base / "tmp/yt",
                rsync_bin="rsync",
                chord_melody_input_dir=base / "input",
                chord_melody_log_dir=base / "logs",
                chord_melody_time_unit=1.0,
                chord_melody_poly_threshold=0.5,
                chord_melody_poly_note_count=1,
                chord_melody_stability_threshold=0.5,
                similar_tones_cache_dir=base / "cache",
                similar_tones_device="cpu",
                similar_tones_segment_seconds=0.4,
                similar_tones_rms_window_seconds=0.05,
                similar_tones_target_db_offset=14.0,
                similar_tones_peak_weight=0.7,
                similar_tones_target_weight=0.3,
                obsidian_vault_root=vault,
                obsidian_export_dir=export_dir,
                obsidian_target_dirs=("inbox", "journal"),
                obsidian_exclude_tags=("type/snippet", "type/account"),
                obsidian_journal_tag="type/journal",
                obsidian_topic_prefix="topic/",
                obsidian_others_group_name="others",
                google_docs_credentials_path=None,
                google_docs_folder_id=None,
                google_oauth_client_id=None,
                google_oauth_client_secret=None,
                google_oauth_refresh_token=None,
                google_oauth_token_uri="https://oauth2.googleapis.com/token",
            )

            with self.assertRaises(obsidian_exports.ObsidianExportPathError):
                obsidian_exports.merge_exports(settings)

    def test_merge_exports_groups_and_orders_notes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            vault = base / "vault"
            export_dir = base / "exports"
            inbox = vault / "inbox"
            journal = vault / "journal"
            inbox.mkdir(parents=True)
            journal.mkdir(parents=True)

            (journal / "journal.md").write_text(
                """---\ndate: 2025-02-01\ntags:\n  - type/journal\n  - topic/tools/mac\n---\n\nJournal body\n""",
                encoding="utf-8",
            )
            (inbox / "topic_old.md").write_text(
                """---\ndate: 2025-01-01\ntags:\n  - topic/tools/mac\n---\n\nOld topic\n""",
                encoding="utf-8",
            )
            (inbox / "topic_new.md").write_text(
                """---\ndate: 2025-03-01\ntags:\n  - topic/tools/mac\n---\n\nNew topic\n""",
                encoding="utf-8",
            )
            (inbox / "others.md").write_text(
                """---\ndate: 2024-01-01\n---\n\nOther body\n""",
                encoding="utf-8",
            )
            (inbox / "excluded.md").write_text(
                """---\ndate: 2024-02-01\ntags:\n  - type/snippet\n---\n\nSkip me\n""",
                encoding="utf-8",
            )

            settings = Settings(
                whisper_bin="whisper",
                whisper_model_path="model",
                ffmpeg_bin="ffmpeg",
                whisper_tmp_dir=base / "tmp/whisper",
                ytdlp_bin="ytdlp",
                ytdlp_output_dir=base / "tmp/yt",
                rsync_bin="rsync",
                chord_melody_input_dir=base / "input",
                chord_melody_log_dir=base / "logs",
                chord_melody_time_unit=1.0,
                chord_melody_poly_threshold=0.5,
                chord_melody_poly_note_count=1,
                chord_melody_stability_threshold=0.5,
                similar_tones_cache_dir=base / "cache",
                similar_tones_device="cpu",
                similar_tones_segment_seconds=0.4,
                similar_tones_rms_window_seconds=0.05,
                similar_tones_target_db_offset=14.0,
                similar_tones_peak_weight=0.7,
                similar_tones_target_weight=0.3,
                obsidian_vault_root=vault,
                obsidian_export_dir=export_dir,
                obsidian_target_dirs=("inbox", "journal"),
                obsidian_exclude_tags=("type/snippet", "type/account"),
                obsidian_journal_tag="type/journal",
                obsidian_topic_prefix="topic/",
                obsidian_others_group_name="others",
                google_docs_credentials_path=None,
                google_docs_folder_id=None,
                google_oauth_client_id=None,
                google_oauth_client_secret=None,
                google_oauth_refresh_token=None,
                google_oauth_token_uri="https://oauth2.googleapis.com/token",
            )

            summary = obsidian_exports.merge_exports(settings)

            self.assertEqual(str(vault), summary["vault_root"])
            self.assertEqual(str(export_dir), summary["output_dir"])
            self.assertEqual(
                {item["group"] for item in summary["generated"]},
                {"type/journal", "topic/tools", "others"},
            )
            self.assertEqual({"excluded_tag": 1, "no_target_tag": 0, "parse_error": 0}, summary["skipped"])

            journal_output = (export_dir / "journal.md").read_text(encoding="utf-8")
            self.assertIn("source: journal/journal.md", journal_output)
            self.assertNotIn("topic_old.md", journal_output)

            topic_output = (export_dir / "topic_tools.md").read_text(encoding="utf-8")
            self.assertLess(
                topic_output.index("source: inbox/topic_new.md"),
                topic_output.index("source: inbox/topic_old.md"),
            )

            others_output = (export_dir / "others.md").read_text(encoding="utf-8")
            self.assertIn("source: inbox/others.md", others_output)
