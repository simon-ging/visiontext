import re

RE_ALNUM = re.compile(r"[^a-zA-Z0-9 ]+")


def preprocess_text_simple(in_str: str, lowercase=True) -> str:
    """Remove non-alphanumeric characters, lowercase, and strip duplicate whitespaces."""
    intermed_str = RE_ALNUM.sub(" ", in_str).strip()
    if lowercase:
        intermed_str = intermed_str.lower()
    return " ".join(intermed_str.split())
