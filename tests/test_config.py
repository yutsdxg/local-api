from __future__ import annotations

import os
import unittest
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
