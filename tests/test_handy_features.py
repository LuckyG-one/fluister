"""Tests for the features borrowed from Handy: VAD speech gate,
hallucination filter, pre-roll recorder and auto-submit app matching.

Run: .venv/bin/python -m unittest tests.test_handy_features
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fluister import config, vad
from fluister.audio import BLOCK_SIZE, SAMPLE_RATE, Recorder
from fluister.main import _app_matches

VAD_CFG = config.DEFAULTS["vad"]


def _block(value: float) -> np.ndarray:
    return np.full((BLOCK_SIZE, 1), value, dtype=np.float32)


class HallucinationFilterTest(unittest.TestCase):
    def test_drops_empty_transcript(self):
        self.assertTrue(vad.is_hallucination("", 2.0, VAD_CFG))
        self.assertTrue(vad.is_hallucination("  ", None, VAD_CFG))

    def test_drops_known_phrase_with_little_speech(self):
        self.assertTrue(vad.is_hallucination("Thank you.", 0.6, VAD_CFG))
        self.assertTrue(vad.is_hallucination("You", 0.4, VAD_CFG))
        self.assertTrue(vad.is_hallucination("Mm-hmm.", None, VAD_CFG))

    def test_keeps_known_phrase_with_enough_speech(self):
        self.assertFalse(vad.is_hallucination("Thank you.", 1.5, VAD_CFG))

    def test_keeps_real_short_dictation(self):
        self.assertFalse(vad.is_hallucination("Executive search.", 0.8, VAD_CFG))
        self.assertFalse(vad.is_hallucination("De knopjes.", 0.5, VAD_CFG))


class SpeechGateTest(unittest.TestCase):
    def test_too_little_speech(self):
        self.assertTrue(vad.has_too_little_speech(0.1, VAD_CFG))
        self.assertFalse(vad.has_too_little_speech(0.5, VAD_CFG))

    def test_fails_open_without_vad(self):
        self.assertFalse(vad.has_too_little_speech(None, VAD_CFG))

    def test_silence_has_no_speech(self):
        self.assertEqual(vad.speech_seconds(np.zeros(SAMPLE_RATE * 2, np.float32)), 0.0)

    def test_synthetic_speech_is_detected(self):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "speech.wav"
            subprocess.run(
                ["say", "-o", str(wav), "--data-format=LEI16@16000", "Dit is een test van Fluister."],
                check=True,
            )
            audio, _ = sf.read(wav, dtype="float32")
        self.assertGreater(vad.speech_seconds(audio), 1.0)


class PrerollRecorderTest(unittest.TestCase):
    def _recorder(self, always_on=True, preroll_ms=300):
        recorder = Recorder(always_on=always_on, preroll_ms=preroll_ms)
        recorder._ensure_stream = lambda: None  # no real audio device
        return recorder

    def test_preroll_is_prepended_and_streamed_first(self):
        recorder = self._recorder()
        for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):  # 6 blocks, ring holds 5
            recorder._callback(_block(value), BLOCK_SIZE, None, None)
        fed = []
        recorder.on_chunk = lambda chunk: fed.append(float(chunk[0]))
        recorder.start()
        recorder._callback(_block(0.9), BLOCK_SIZE, None, None)
        audio = recorder.stop()

        expected = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
        self.assertEqual([round(v, 1) for v in fed], expected)
        self.assertEqual(len(audio), len(expected) * BLOCK_SIZE)
        self.assertAlmostEqual(float(audio[0]), 0.2, places=5)

    def test_no_preroll_when_not_always_on(self):
        recorder = self._recorder(always_on=False)
        recorder._callback(_block(0.1), BLOCK_SIZE, None, None)
        recorder.start()
        recorder._callback(_block(0.9), BLOCK_SIZE, None, None)
        audio = recorder.stop()
        self.assertEqual(len(audio), BLOCK_SIZE)

    def test_ring_is_cleared_between_recordings(self):
        recorder = self._recorder()
        recorder._callback(_block(0.1), BLOCK_SIZE, None, None)
        recorder.start()
        recorder.stop()
        recorder.start()
        self.assertEqual(len(recorder.stop()), 0)

    def test_stop_without_start_returns_empty(self):
        self.assertEqual(len(self._recorder().stop()), 0)


class AutoSubmitMatchTest(unittest.TestCase):
    def test_matches_case_insensitive_and_strips_ltr_mark(self):
        self.assertTrue(_app_matches("‎WhatsApp", ["whatsapp"]))
        self.assertTrue(_app_matches("Claude", ["Claude"]))

    def test_no_match(self):
        self.assertFalse(_app_matches("Safari", ["Claude"]))
        self.assertFalse(_app_matches("", ["Claude"]))
        self.assertFalse(_app_matches("Claude", []))


if __name__ == "__main__":
    unittest.main()
