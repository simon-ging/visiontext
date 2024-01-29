from __future__ import annotations

import re

RE_ALNUM = re.compile(r"[^a-zA-Z0-9 ]+")


def preprocess_text_simple(in_str: str, lowercase=True, regexp: re.Pattern | str = RE_ALNUM) -> str:
    """Remove non-alphanumeric characters, lowercase, and strip duplicate whitespaces."""
    if not isinstance(regexp, re.Pattern):
        regexp = re.compile(regexp)
    intermed_str = regexp.sub(" ", in_str).strip()
    if lowercase:
        intermed_str = intermed_str.lower()
    return " ".join(intermed_str.split())
