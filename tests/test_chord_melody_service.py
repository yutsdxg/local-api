from __future__ import annotations

import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from app.config import Settings
from app.services import chord_melody


class TestChordMelodyService(unittest.TestCase):
    def test_analyze_chord_melody_classifies_and_renames(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            input_dir = base / "input"
            log_dir = base / "logs"
            input_dir.mkdir()

            (input_dir / "song_MLD.wav").write_bytes(b"fake-audio")
            (input_dir / "other_CHP.wav").write_bytes(b"fake-audio")
            (input_dir / "ignore.wav").write_bytes(b"fake-audio")

            settings = Settings(
                whisper_bin="whisper",
                whisper_model_path="model",
                ffmpeg_bin="ffmpeg",
                whisper_tmp_dir=base / "tmp/whisper",
                ytdlp_bin="ytdlp",
                ytdlp_output_dir=base / "tmp/yt",
                rsync_bin="rsync",
                chord_melody_input_dir=input_dir,
                chord_melody_log_dir=log_dir,
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

            with mock.patch.object(chord_melody, "_predict_audio", side_effect=_fake_predict):
                results = chord_melody.analyze_chord_melody(settings)

            self.assertEqual(
                [
                    {"path": "other_CHP.wav", "result": "CHORD1"},
                    {"path": "song_MLD.wav", "result": "CHORD1"},
                ],
                sorted(results, key=lambda r: r["path"]),
            )

            self.assertTrue((input_dir / "song_CH1.wav").exists())
            self.assertTrue((input_dir / "other_CH1.wav").exists())
            self.assertFalse((input_dir / "ignore.wav").with_stem("ignore_CH1").exists())

            log_file = log_dir / "analysis.log"
            self.assertTrue(log_file.exists())
            content = log_file.read_text()
            self.assertIn("song_MLD.wav\tCHORD1", content)
            self.assertIn("other_CHP.wav\tCHORD1", content)


def _fake_predict(_: str) -> tuple:
    class FakeNote:
        def __init__(self, start: float, end: float, pitch: int) -> None:
            self.start = start
            self.end = end
            self.pitch = pitch

    class FakeInstrument:
        def __init__(self, notes) -> None:
            self.notes = notes

    class FakeOnsets:
        def __init__(self, instruments) -> None:
            self.instruments = instruments

    notes = [
        FakeNote(0.0, 2.0, 60),
        FakeNote(0.0, 2.0, 64),
        FakeNote(1.0, 2.0, 67),
    ]
    return (None, FakeOnsets([FakeInstrument(notes)]), None)
