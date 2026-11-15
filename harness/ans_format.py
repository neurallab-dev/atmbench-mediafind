"""Deterministic answer formatter for number-type answers: calendar dates are written with the full month name
("April 20, 2023") instead of ISO "2023-04-20"; clock times drop seconds and a timezone suffix that follows them;
"5:30 pm" is written compactly as "5:30pm". Pure string rewriting; no model call."""
import re
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
_ISO = re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b")
_SECS = re.compile(r"\b(\d{1,2}:\d{2}):\d{2}(?:\.\d+)?(?:\s*(?:[+-]\d{2}:?\d{2}|Z\b|UTC\b))?")
_AMPM = re.compile(r"\b(\d{1,2}(?::\d{2})?)\s+([AaPp])\.?([Mm])\.?(?![A-Za-z])")
def _iso(m):
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31): return m.group(0)
    return f"{MONTHS[mo-1]} {d}, {y}"
def format_number_answer(s: str) -> str:
    s = _SECS.sub(r"\1", s)
    s = _ISO.sub(_iso, s)
    s = _AMPM.sub(lambda m: f"{m.group(1)}{m.group(2).lower()}{m.group(3).lower()}", s)
    return s
if __name__ == "__main__":
    for t in ["2023-04-20 14:40", "2022-06-19 12:12:28+00:00", "13:17-14:17", "between 08:17-09:17", "from 3 pm to 5:30 pm", "12:15 PM", "2024-10-20 at 14:49 (2:49 PM)", "August 12, 2024, from 19:00-22:00", "9:30am to 10:30am"]:
        print(f"  {t!r:45s} -> {format_number_answer(t)!r}")
