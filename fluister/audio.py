"""Microphone capture at 16 kHz mono, with live RMS levels for the pill waveform.

With `always_on`, the input stream stays open between dictations and keeps
a short ring buffer (pre-roll), so the first syllable is never lost to the
time it takes to open the mic (Handy: always_on_microphone). macOS then
shows the orange mic indicator permanently. Without it, the stream opens
on key press and closes on release, as before.
"""
import math
import threading
from collections import deque

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024


class Recorder:
    def __init__(self, always_on: bool = False, preroll_ms: int = 0):
        self.always_on = always_on
        self._frames = []
        self._stream = None
        self._capturing = False
        self._lock = threading.Lock()
        preroll_blocks = math.ceil(preroll_ms / 1000 * SAMPLE_RATE / BLOCK_SIZE)
        self._ring = deque(maxlen=max(1, preroll_blocks))
        self._use_preroll = always_on and preroll_ms > 0
        self.levels = deque(maxlen=64)
        # Optional per-recording tap for streaming ASR; set/cleared by the
        # controller around each recording.
        self.on_chunk = None

    @property
    def recording(self) -> bool:
        return self._capturing

    def open(self):
        """Open the always-on stream at startup. No-op otherwise."""
        if self.always_on:
            self._ensure_stream()

    def _ensure_stream(self):
        # Never called with the lock held: the audio callback takes that
        # lock, and stream.stop() waits for a running callback.
        stream = self._stream
        if stream is not None and stream.active:
            return
        if stream is not None:
            # A stream that died (e.g. after an audio device change).
            _close(stream)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def start(self):
        if self._capturing:
            return
        self._ensure_stream()
        with self._lock:
            preroll = list(self._ring) if self._use_preroll else []
            self._frames = preroll
            self.levels.clear()
            # Fed under the lock so pre-roll reaches the streaming ASR
            # before any live chunk does.
            self._emit(preroll)
            self._capturing = True

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        with self._lock:
            if not self._capturing:
                self._ring.append(chunk)
                return
            self._frames.append(chunk)
            self.levels.append(float(np.sqrt(np.mean(chunk ** 2))))
            self._emit([chunk])

    def _emit(self, chunks):
        callback = self.on_chunk
        if callback is None:
            return
        for chunk in chunks:
            try:
                callback(chunk)
            except Exception:
                return  # a streaming bug must never kill audio capture

    def stop(self) -> np.ndarray:
        with self._lock:
            if not self._capturing:
                return np.zeros(0, dtype=np.float32)
            self._capturing = False
            frames, self._frames = self._frames, []
            self._ring.clear()
        if not self.always_on and self._stream is not None:
            stream, self._stream = self._stream, None
            _close(stream)
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames)


def _close(stream):
    try:
        stream.stop()
        stream.close()
    except Exception:
        pass
