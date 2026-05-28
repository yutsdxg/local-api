from __future__ import annotations

import asyncio
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi import UploadFile

from app import main
from app.config import Settings
from app.services import whisper as whisper_service


class TestWhisperService(unittest.TestCase):
    def test_transcribe_upload_passes_configured_whisper_args(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            whisper_tmp_dir = tmp_path / "whisper"
            whisper_tmp_dir.mkdir()
            settings = _settings(whisper_tmp_dir, whisper_args=("-ng", "-nt", "-np"))
            upload = UploadFile(filename="audio.m4a", file=io.BytesIO(b"audio"))
            calls: list[list[str]] = []

            def fake_run(cmd: list[str], check: bool) -> None:
                self.assertTrue(check)
                calls.append(cmd)
                if "-of" in cmd:
                    output_prefix = Path(cmd[cmd.index("-of") + 1])
                    output_prefix.with_suffix(".txt").write_text("transcribed", encoding="utf-8")

            with mock.patch.object(whisper_service.subprocess, "run", side_effect=fake_run):
                text = asyncio.run(
                    whisper_service.transcribe_upload(upload, settings, language="ja")
                )

            self.assertEqual("transcribed", text)
            self.assertEqual(2, len(calls))
            whisper_cmd = calls[1]
            self.assertIn("-ng", whisper_cmd)
            self.assertIn("-nt", whisper_cmd)
            self.assertIn("-np", whisper_cmd)
            self.assertEqual(["-l", "ja", "-ng", "-nt", "-np"], whisper_cmd[-5:])

    def test_whisper_endpoint_response_contract_is_unchanged(self) -> None:
        upload = mock.Mock(spec=UploadFile)

        with mock.patch.object(
            main.whisper_service,
            "transcribe_upload",
            new_callable=mock.AsyncMock,
            return_value="transcribed",
        ) as mocked_transcribe:
            result = asyncio.run(main.whisper(file=upload, language="ja"))

        mocked_transcribe.assert_awaited_once_with(upload, main.settings, language="ja")
        self.assertEqual({"text": "transcribed"}, result)


def _settings(
    whisper_tmp_dir: Path,
    whisper_args: tuple[str, ...],
) -> Settings:
    base = whisper_tmp_dir.parent
    return Settings(
        whisper_bin="whisper",
        whisper_model_path="model",
        whisper_args=whisper_args,
        ffmpeg_bin="ffmpeg",
        whisper_tmp_dir=whisper_tmp_dir,
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
        obsidian_vault_root=base / "vault",
        obsidian_export_dir=base / "exports",
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
