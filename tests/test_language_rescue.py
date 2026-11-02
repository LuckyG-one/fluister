"""Tests for the foreign-script guard and the persisted language setting.

Run: .venv/bin/python -m unittest tests.test_language_rescue
"""
import tempfile
import unittest
from pathlib import Path

from fluister import config as config_module
from fluister.asr import ASR, has_foreign_script


class ForeignScriptTests(unittest.TestCase):
    def test_detects_cyrillic(self):
        self.assertTrue(has_foreign_script("Вэй вэ вэйпнтгугл пункт ком."))

    def test_latin_text_passes(self):
        self.assertFalse(has_foreign_script("Eén, twee, drie."))
        self.assertFalse(has_foreign_script("www.google.com 123"))

    def test_streaming_only_when_language_is_auto(self):
        asr = ASR({"asr": {"backend": "parakeet", "language": None}})
        self.assertTrue(asr.streaming)
        asr.cfg["language"] = "nl"
        self.assertFalse(asr.streaming)


class SaveLanguageTests(unittest.TestCase):
    def test_rewrites_only_the_language_line(self):
        original = config_module.CONFIG_PATH
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "asr:\n  backend: parakeet\n  language: null  # auto\n", encoding="utf-8"
            )
            config_module.CONFIG_PATH = path
            try:
                config_module.save_asr_language("nl")
                self.assertIn("  language: nl", path.read_text())
                self.assertIn("backend: parakeet", path.read_text())
                config_module.save_asr_language(None)
                self.assertIn("  language: null", path.read_text())
            finally:
                config_module.CONFIG_PATH = original


if __name__ == "__main__":
    unittest.main()
