from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from app.config import Settings


class TestSettings(unittest.TestCase):
    def test_whisper_args_default_to_cpu_stable_flags(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = Settings.load()

        self.assertEqual(("-ng", "-nt", "-np"), settings.whisper_args)

    def test_whisper_args_are_parsed_with_shell_quoting(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"LOCAL_API_WHISPER_ARGS": '-ng --prompt "hello world"'},
            clear=True,
        ):
            settings = Settings.load()

        self.assertEqual(("-ng", "--prompt", "hello world"), settings.whisper_args)

    def test_whisper_args_allow_empty_override(self) -> None:
        with mock.patch.dict(os.environ, {"LOCAL_API_WHISPER_ARGS": ""}, clear=True):
            settings = Settings.load()

        self.assertEqual((), settings.whisper_args)

    def test_preprocessing_defaults_to_vad_without_normalization(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = Settings.load()

        self.assertEqual("vad", settings.whisper_preprocessing)
        self.assertFalse(settings.whisper_normalize)
        self.assertEqual(Path("data/models/ggml-silero-v6.2.0.bin"), settings.whisper_vad_model_path)
        self.assertEqual(0.5, settings.whisper_vad_threshold)
        self.assertEqual(100, settings.whisper_vad_min_speech_duration_ms)
        self.assertEqual(500, settings.whisper_vad_min_silence_duration_ms)
        self.assertEqual(200, settings.whisper_vad_speech_pad_ms)
        self.assertEqual(12, settings.whisper_deepfilter_attenuation_limit_db)

    def test_preprocessing_environment_overrides(self) -> None:
        with mock.patch.dict(os.environ, {
            "LOCAL_API_WHISPER_PREPROCESSING": " DEEPFILTER ",
            "LOCAL_API_WHISPER_NORMALIZE": "true",
            "LOCAL_API_WHISPER_VAD_MODEL_PATH": "/models/vad.bin",
            "LOCAL_API_WHISPER_VAD_THRESHOLD": "0.4",
            "LOCAL_API_WHISPER_VAD_MIN_SPEECH_DURATION_MS": "80",
            "LOCAL_API_WHISPER_VAD_MIN_SILENCE_DURATION_MS": "800",
            "LOCAL_API_WHISPER_VAD_SPEECH_PAD_MS": "300",
            "LOCAL_API_WHISPER_DEEPFILTER_BIN": "/bin/deep-filter",
            "LOCAL_API_WHISPER_DEEPFILTER_MODEL_PATH": "/models/dfn.tar.gz",
            "LOCAL_API_WHISPER_DEEPFILTER_ATTENUATION_LIMIT_DB": "6",
        }, clear=True):
            settings = Settings.load()

        self.assertEqual("deepfilter", settings.whisper_preprocessing)
        self.assertTrue(settings.whisper_normalize)
        self.assertEqual(Path("/models/vad.bin"), settings.whisper_vad_model_path)
        self.assertEqual(0.4, settings.whisper_vad_threshold)
        self.assertEqual(80, settings.whisper_vad_min_speech_duration_ms)
        self.assertEqual(800, settings.whisper_vad_min_silence_duration_ms)
        self.assertEqual(300, settings.whisper_vad_speech_pad_ms)
        self.assertEqual("/bin/deep-filter", settings.whisper_deepfilter_bin)
        self.assertEqual(Path("/models/dfn.tar.gz"), settings.whisper_deepfilter_model_path)
        self.assertEqual(6, settings.whisper_deepfilter_attenuation_limit_db)

    def test_normalize_boolean_does_not_treat_false_as_true(self) -> None:
        for value in ("0", "false", "no", "off"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"LOCAL_API_WHISPER_NORMALIZE": value}, clear=True
            ):
                self.assertFalse(Settings.load().whisper_normalize)

    def test_invalid_normalize_boolean_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {"LOCAL_API_WHISPER_NORMALIZE": "invalid"}, clear=True):
            with self.assertRaisesRegex(ValueError, "WHISPER_NORMALIZE"):
                Settings.load()
