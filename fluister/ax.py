"""Shared Accessibility (AX) primitives, used by both inject.py (insert
text at the cursor without touching the clipboard) and corrections.py
(read the focused field back to detect a manual correction). Every
function fails silently (None/False), matching the defensive style
already used elsewhere (inject.py.frontmost_app, llm.py.format_transcript).
"""
import platform

IS_MAC = platform.system() == "Darwin"

MAX_VALUE_LENGTH = 200_000
EDITABLE_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}


def enabled() -> bool:
    if not IS_MAC:
        return False
    try:
        import ApplicationServices as AS

        return bool(AS.AXIsProcessTrusted())
    except Exception:
        return False


def focused_element():
    if not IS_MAC:
        return None
    try:
        import ApplicationServices as AS

        system_wide = AS.AXUIElementCreateSystemWide()
        err, element = AS.AXUIElementCopyAttributeValue(
            system_wide, "AXFocusedUIElement", None
        )
        if err != 0 or element is None:
            return None
        return element
    except Exception:
        return None


def element_pid(element) -> int | None:
    if element is None:
        return None
    try:
        import ApplicationServices as AS

        err, pid = AS.AXUIElementGetPid(element, None)
        if err != 0 or not pid:
            return None
        return int(pid)
    except Exception:
        return None


def role(element) -> str | None:
    if element is None:
        return None
    try:
        import ApplicationServices as AS

        err, value = AS.AXUIElementCopyAttributeValue(element, "AXRole", None)
        if err != 0:
            return None
        return str(value) if value else None
    except Exception:
        return None


def is_secure(element) -> bool:
    if element is None:
        return False
    try:
        import ApplicationServices as AS

        err, value = AS.AXUIElementCopyAttributeValue(element, "AXSubrole", None)
        if err != 0 or not value:
            return False
        subrole = str(value)
        return subrole == "AXSecureTextField" or "secure" in subrole.lower()
    except Exception:
        return False


def read_value(element) -> str | None:
    if element is None:
        return None
    try:
        import ApplicationServices as AS

        err, value = AS.AXUIElementCopyAttributeValue(element, "AXValue", None)
        if err != 0 or not isinstance(value, str):
            return None
        if len(value) > MAX_VALUE_LENGTH:
            return None
        return value
    except Exception:
        return None


def read_selected_range(element) -> tuple[int, int] | None:
    if element is None:
        return None
    try:
        import ApplicationServices as AS

        err, value = AS.AXUIElementCopyAttributeValue(
            element, "AXSelectedTextRange", None
        )
        if err != 0 or value is None:
            return None
        ok, cf_range = AS.AXValueGetValue(value, AS.kAXValueCFRangeType, None)
        if not ok:
            return None
        return (int(cf_range.location), int(cf_range.length))
    except Exception:
        return None


def write_value(element, new_value: str) -> bool:
    if element is None:
        return False
    try:
        import ApplicationServices as AS

        err = AS.AXUIElementSetAttributeValue(element, "AXValue", new_value)
        return err == 0
    except Exception:
        return False


def write_caret(element, location: int) -> bool:
    if element is None:
        return False
    try:
        import ApplicationServices as AS
        from Foundation import NSMakeRange

        ax_range = AS.AXValueCreate(
            AS.kAXValueCFRangeType, NSMakeRange(location, 0)
        )
        if ax_range is None:
            return False
        err = AS.AXUIElementSetAttributeValue(
            element, "AXSelectedTextRange", ax_range
        )
        return err == 0
    except Exception:
        return False


def insert_at_cursor(element, text: str) -> bool:
    """Read the current value + selection, splice `text` in, write both
    back. Refuses secure fields and non-editable roles outright."""
    if element is None or not text:
        return False
    if is_secure(element):
        return False
    if role(element) not in EDITABLE_ROLES:
        return False

    value = read_value(element)
    selected = read_selected_range(element)
    if value is None or selected is None:
        return False

    location, length = selected
    location = max(0, min(location, len(value)))
    length = max(0, min(length, len(value) - location))

    new_value = value[:location] + text + value[location + length :]
    if not write_value(element, new_value):
        return False

    write_caret(element, location + len(text))
    return True
