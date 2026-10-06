"""
utils.py
Small helper to clean raw scraped HTML/text before it's sent to the LLM —
strips extra whitespace, HTML leftovers, and links, so fewer tokens are
wasted on noise.
"""

import re


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]*?>", "", text)           # strip leftover HTML tags
    text = re.sub(r"http\S+|www\.\S+", "", text)    # strip URLs
    text = re.sub(r"[^a-zA-Z0-9 .,\n]", " ", text)  # strip special characters
    text = re.sub(r"\s{2,}", " ", text)             # collapse repeated whitespace
    return text.strip()
