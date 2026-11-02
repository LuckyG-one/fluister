"""Tests for the foreign-script guard and the persisted language setting.

Run: .venv/bin/python -m unittest tests.test_language_rescue
"""
import tempfile
import unittest
from pathlib import Path

import yaml

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
    def _roundtrip(self, text, language):
        """Write text as config.yaml, save the language, return the parsed
        asr.language and the new file content."""
        original = config_module.CONFIG_PATH
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            config_module.CONFIG_PATH = path
            try:
                config_module.save_asr_language(language)
                content = path.read_text()
            finally:
                config_module.CONFIG_PATH = original
        return yaml.safe_load(content)["asr"]["language"], content

    def test_keeps_inline_comment_and_parses(self):
        text = "asr:\n  backend: parakeet\n  language: null  # auto\n"
        value, content = self._roundtrip(text, "nl")
        self.assertEqual(value, "nl")
        self.assertIn("  language: nl  # auto", content)
        self.assertIn("backend: parakeet", content)
        value, _ = self._roundtrip(content, None)
        self.assertIsNone(value)

    def test_four_space_indentation(self):
        value, _ = self._roundtrip("asr:\n    backend: parakeet\n    language: en\n", None)
        self.assertIsNone(value)

    def test_adds_missing_key(self):
        value, content = self._roundtrip("asr:\n  backend: parakeet\nllm:\n  model: x\n", "nl")
        self.assertEqual(value, "nl")
        self.assertIn("llm:\n  model: x", content)

    def test_adds_missing_asr_block(self):
        value, _ = self._roundtrip("llm:\n  model: x\n", "en")
        self.assertEqual(value, "en")

    def test_ignores_language_key_in_other_sections(self):
        text = "asr:\n  backend: parakeet\nother:\n  language: fr\n"
        original = config_module.CONFIG_PATH
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            config_module.CONFIG_PATH = path
            try:
                config_module.save_asr_language("nl")
                data = yaml.safe_load(path.read_text())
            finally:
                config_module.CONFIG_PATH = original
        self.assertEqual(data["asr"]["language"], "nl")
        self.assertEqual(data["other"]["language"], "fr")


if __name__ == "__main__":
    unittest.main()
