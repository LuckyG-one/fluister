"""Start/stop sounds and muting system audio while recording (Handy:
audio_feedback, mute_while_recording). macOS only; elsewhere a no-op.

All calls run on the Qt main thread.
"""
import platform

IS_MAC = platform.system() == "Darwin"


class Feedback:
    def __init__(self, cfg: dict):
        self.cfg = cfg["feedback"]
        self._muted_by_us = False

    def recording_started(self):
        if self.cfg["sounds"]:
            _play(self.cfg["start_sound"], self.cfg["volume"])

    def mute(self):
        """Called shortly after the start sound so it is not cut off."""
        if self.cfg["mute_while_recording"] and not _output_muted():
            self._muted_by_us = _set_output_muted(True)

    def recording_stopped(self):
        if self._muted_by_us:
            _set_output_muted(False)
            self._muted_by_us = False
        if self.cfg["sounds"]:
            _play(self.cfg["stop_sound"], self.cfg["volume"])


def _play(name: str, volume: float):
    if not IS_MAC or not name:
        return
    try:
        from AppKit import NSSound

        sound = NSSound.soundNamed_(name)
        if sound is None:
            return
        sound.setVolume_(volume)
        sound.play()
    except Exception:
        pass


def _run_applescript(source: str):
    from Foundation import NSAppleScript

    result, error = NSAppleScript.alloc().initWithSource_(source).executeAndReturnError_(None)
    if error is not None:
        raise RuntimeError(str(error))
    return result


def _output_muted() -> bool:
    if not IS_MAC:
        return False
    try:
        result = _run_applescript("output muted of (get volume settings)")
        return bool(result.booleanValue())
    except Exception:
        return False


def _set_output_muted(muted: bool) -> bool:
    if not IS_MAC:
        return False
    try:
        _run_applescript(f"set volume output muted {'true' if muted else 'false'}")
        return True
    except Exception as exc:
        print(f"fluister: could not change output mute ({exc})")
        return False
