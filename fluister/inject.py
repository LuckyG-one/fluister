"""Put the final text into whatever app has focus, and detect which app
that is (used for the per-app style hint).

Injection strategy: a two-tier cascade. First try inserting directly at
the cursor via the Accessibility API (fluister.ax), which never touches
the system clipboard, so clipboard managers never see the dictation.
If that fails (unsupported app, secure field, permission missing), fall
back to the proven clipboard swap plus a synthetic Cmd+V (Ctrl+V on
Windows/Linux), the same approach VoiceInk and Handy use. On macOS both
paths need the Accessibility permission.
"""
import platform
import threading
import time

import pyperclip
from pynput.keyboard import Controller, Key

from . import ax

_keyboard = Controller()
IS_MAC = platform.system() == "Darwin"


def paste_text(text: str, cfg: dict | None = None, restore_clipboard: bool = True):
    if not text:
        return

    ax_insert_enabled = (cfg or {}).get("injection", {}).get("ax_insert", True)
    if IS_MAC and ax_insert_enabled and _try_ax_insert(text):
        return

    _paste_via_clipboard(text, restore_clipboard)


def _try_ax_insert(text: str) -> bool:
    element = ax.focused_element()
    if element is None:
        return False
    if not ax.insert_at_cursor(element, text):
        return False
    threading.Thread(target=_verify_ax_insert, args=(element, text), daemon=True).start()
    return True


def _verify_ax_insert(element, text: str):
    time.sleep(0.3)
    try:
        value = ax.read_value(element)
        if value is None or text not in value:
            print(f"fluister: AX insert verification failed for {frontmost_app()!r}")
    except Exception:
        pass


def _paste_via_clipboard(text: str, restore_clipboard: bool = True):
    previous = None
    if restore_clipboard:
        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None

    pyperclip.copy(text)
    time.sleep(0.08)

    modifier = Key.cmd if IS_MAC else Key.ctrl
    with _keyboard.pressed(modifier):
        _keyboard.press("v")
        _keyboard.release("v")

    if restore_clipboard and previous:
        def _restore():
            time.sleep(0.5)
            try:
                pyperclip.copy(previous)
            except Exception:
                pass

        threading.Thread(target=_restore, daemon=True).start()


def press_return_later(delay_s: float = 0.15):
    """Press Return once the paste has landed (Handy: auto_submit)."""
    def _press():
        time.sleep(delay_s)
        _keyboard.press(Key.enter)
        _keyboard.release(Key.enter)

    threading.Thread(target=_press, daemon=True).start()


def frontmost_app() -> str:
    """Name of the app that currently has focus, or an empty string."""
    try:
        if IS_MAC:
            from AppKit import NSWorkspace

            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return str(app.localizedName()) if app else ""
        if platform.system() == "Windows":
            import ctypes

            handle = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(handle)
            buffer = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(handle, buffer, length + 1)
            return buffer.value
    except Exception:
        pass
    return ""
