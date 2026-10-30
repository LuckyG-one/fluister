"""Learns from corrections you make right after a dictation lands, the
same idea as Wispr Flow's personal dictionary training. Polling-based by
design: no AXObserver, no Cocoa run loop, just a delayed re-read on a
plain daemon thread, matching this codebase's existing threading model
(see inject.py's clipboard-restore thread).

Never writes to config.yaml and never promotes a correction straight
into the LLM `dictionary:` block; see the dictionary_learning note in config.example.yaml on why unscoped names
in the dictionary over-correct unrelated people.
"""
import json
import re
import string
import threading
import time

from . import ax
from . import config as config_module

_STRIP = string.whitespace + string.punctuation
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_BOUNDARY_RE = re.compile(r"""[\s.,!?;:"'()\[\]{}<>]""")


def _expand_to_token(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) to the surrounding word(s), so a single
    differing character ('c' -> 'k') becomes the whole corrected word
    ('Sarah' -> 'Sara') instead of a meaningless letter swap."""
    while start > 0 and not _BOUNDARY_RE.match(text[start - 1]):
        start -= 1
    while end < len(text) and not _BOUNDARY_RE.match(text[end]):
        end += 1
    return start, end


def find_correction(before: str, after: str, region: tuple[int, int]):
    """Pure diff: compares the field value right after paste (`before`)
    against a later read (`after`). `region` is the (start, end) span
    where the pasted text originally sat in `before`. Returns
    (heard, corrected) if the change is a short, meaningfully different,
    alphabetic substitution fully inside `region`; else None."""
    if before == after:
        return None

    prefix = 0
    max_prefix = min(len(before), len(after))
    while prefix < max_prefix and before[prefix] == after[prefix]:
        prefix += 1

    suffix = 0
    max_suffix = min(len(before) - prefix, len(after) - prefix)
    while (
        suffix < max_suffix
        and before[len(before) - suffix - 1] == after[len(after) - suffix - 1]
    ):
        suffix += 1

    old_start, old_end = prefix, len(before) - suffix
    new_start, new_end = prefix, len(after) - suffix
    if old_start >= old_end and new_start >= new_end:
        return None

    if old_start == old_end:
        # Nothing was replaced, only inserted: the user kept typing
        # (continuing a sentence, adding a word after) rather than
        # correcting something Fluister pasted. A genuine correction
        # always removes at least one character of what was heard.
        return None

    old_start, old_end = _expand_to_token(before, old_start, old_end)
    new_start, new_end = _expand_to_token(after, new_start, new_end)

    region_start, region_end = region
    if not (old_start >= region_start and old_end <= region_end):
        return None

    heard = before[old_start:old_end].strip(_STRIP)
    corrected = after[new_start:new_end].strip(_STRIP)
    return _validate(heard, corrected)


def _validate(heard: str, corrected: str):
    if not heard or not corrected or heard == corrected:
        return None
    if heard.lower() == corrected.lower():
        return None
    if len(heard) + len(corrected) > 70:
        return None
    for value in (heard, corrected):
        words = value.split()
        if not words or len(words) > 3:
            return None
        if not _WORD_RE.search(value):
            return None
    return heard, corrected


def watch_for_correction(cfg: dict, pasted_text: str, tray=None):
    """Call once, right after paste_text() returns. Best-effort: does
    nothing if AX is unavailable or the pasted text can't be located in
    the focused field's current value."""
    settings = cfg.get("dictionary_learning") or {}
    if not ax.IS_MAC or not settings.get("enabled", True) or not pasted_text:
        return
    baseline = _capture_baseline(pasted_text)
    if baseline is None:
        return
    delay = settings.get("check_delay_seconds", 4)
    threading.Thread(
        target=_check_later, args=(cfg, baseline, delay, tray), daemon=True
    ).start()


def _capture_baseline(pasted_text: str):
    element = ax.focused_element()
    if element is None:
        return None
    value = ax.read_value(element)
    if value is None:
        return None
    start = value.rfind(pasted_text)
    if start < 0:
        return None
    return {
        "element": element,
        "pid": ax.element_pid(element),
        "value": value,
        "region": (start, start + len(pasted_text)),
    }


def _check_later(cfg: dict, baseline: dict, delay: float, tray):
    time.sleep(delay)
    try:
        element = baseline["element"]
        if ax.element_pid(element) != baseline["pid"]:
            return
        current = ax.read_value(element)
        if current is None:
            return
        result = find_correction(baseline["value"], current, baseline["region"])
        if result is None:
            return
        heard, corrected = result
        record_candidate(cfg, heard, corrected, tray=tray)
    except Exception:
        pass


def _key(heard: str, corrected: str) -> str:
    return f"{heard.lower()}\x1f{corrected.lower()}"


def _load() -> dict:
    path = config_module.LEARNED_CORRECTIONS_PATH
    if not path.exists():
        return {"pairs": {}}
    try:
        return json.loads(path.read_text()) or {"pairs": {}}
    except Exception:
        return {"pairs": {}}


def _save(data: dict):
    path = config_module.LEARNED_CORRECTIONS_PATH
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(path)
    except Exception:
        pass


def _already_configured(cfg: dict, heard: str, corrected: str) -> bool:
    heard_l, corrected_l = heard.lower(), corrected.lower()
    for wrong, right in (cfg.get("replacements") or {}).items():
        if heard_l in wrong.lower() and corrected_l in right.lower():
            return True
    for term in cfg.get("dictionary") or []:
        if term.lower() == corrected_l:
            return True
    return False


def record_candidate(cfg: dict, heard: str, corrected: str, tray=None):
    settings = cfg.get("dictionary_learning") or {}
    if not settings.get("enabled", True):
        return
    threshold = settings.get("threshold", 2)
    data = _load()
    data.setdefault("pairs", {})
    key = _key(heard, corrected)
    entry = data["pairs"].setdefault(
        key,
        {"heard": heard, "corrected": corrected, "count": 0, "notified": False},
    )
    entry["count"] += 1
    entry["last_seen"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save(data)

    if (
        entry["count"] >= threshold
        and not entry["notified"]
        and not _already_configured(cfg, heard, corrected)
    ):
        entry["notified"] = True
        _save(data)
        if tray is not None:
            try:
                from PySide6.QtWidgets import QSystemTrayIcon

                tray.showMessage(
                    "Fluister leerde een correctie",
                    f'"{heard}" werd steeds veranderd naar "{corrected}". '
                    "Voeg toe aan replacements in config.yaml als dit klopt.",
                    QSystemTrayIcon.MessageIcon.Information,
                    6000,
                )
            except Exception:
                pass
