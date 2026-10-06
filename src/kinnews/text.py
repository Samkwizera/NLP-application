import re
import unicodedata

# Kinyarwanda marks vowel elision with an apostrophe (ya + abantu -> y'abantu).
# The corpus mixes several apostrophe characters for this, so map them all to one.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})
_QUOTES = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"'})
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_APOSTROPHES).translate(_QUOTES)
    text = _URL.sub(" ", text)
    return _WS.sub(" ", text).strip()


_WORD = re.compile(r"[a-z]+")


def tokenize(text: str, split_elision: bool = True) -> list[str]:
    # splitting "y'abantu" -> "y", "abantu" means "abantu" gets one token with or without the prefix
    text = normalize(text).lower()
    if split_elision:
        return _WORD.findall(text)
    return re.findall(r"[a-z]+(?:'[a-z]+)*", text)
