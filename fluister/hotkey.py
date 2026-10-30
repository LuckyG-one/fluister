"""Global push-to-talk listener.

Two modes:
- pynput: a normal key or modifier (default: right Option / alt_r). Needs the
  Input Monitoring permission on macOS.
- fn (macOS only): a Quartz event tap on flagsChanged so the fn/globe key
  behaves exactly like in Wispr Flow. Needs the Accessibility permission.
"""
import threading

from pynput import keyboard

KEY_MAP = {
    "alt_r": keyboard.Key.alt_r,
    "alt_l": keyboard.Key.alt_l,
    "cmd_r": keyboard.Key.cmd_r,
    "ctrl_r": keyboard.Key.ctrl_r,
    "shift_r": keyboard.Key.shift_r,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
}


class HoldKeyListener:
    """Fires on_press_start when the key goes down, on_press_stop on release,
    and on_cancel when Esc is hit while the key is held."""

    def __init__(self, key_name, on_press_start, on_press_stop, on_cancel=None):
        self.key = KEY_MAP.get(key_name, keyboard.Key.alt_r)
        self.on_press_start = on_press_start
        self.on_press_stop = on_press_stop
        self.on_cancel = on_cancel
        self._down = False
        self._listener = keyboard.Listener(
            on_press=self._press, on_release=self._release
        )

    def start(self):
        self._listener.start()

    def _press(self, key):
        if key == self.key and not self._down:
            self._down = True
            self.on_press_start()
        elif key == keyboard.Key.esc and self._down and self.on_cancel:
            self._down = False
            self.on_cancel()

    def _release(self, key):
        if key == self.key and self._down:
            self._down = False
            self.on_press_stop()


def start_fn_listener(on_press_start, on_press_stop):
    """macOS only: watch the fn/globe modifier via a Quartz event tap.

    Runs in its own daemon thread with its own run loop. If the tap cannot
    be created, the Accessibility permission is missing.
    """
    import Quartz

    state = {"down": False}

    def _tap(proxy, event_type, event, refcon):
        flags = Quartz.CGEventGetFlags(event)
        fn_down = bool(flags & Quartz.kCGEventFlagMaskSecondaryFn)
        if fn_down and not state["down"]:
            state["down"] = True
            on_press_start()
        elif not fn_down and state["down"]:
            state["down"] = False
            on_press_stop()
        return event

    def _run():
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
            _tap,
            None,
        )
        if tap is None:
            print("fluister: could not create fn event tap, check Accessibility permission")
            return
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(
            Quartz.CFRunLoopGetCurrent(), source, Quartz.kCFRunLoopCommonModes
        )
        Quartz.CGEventTapEnable(tap, True)
        Quartz.CFRunLoopRun()

    threading.Thread(target=_run, daemon=True, name="fn-listener").start()
