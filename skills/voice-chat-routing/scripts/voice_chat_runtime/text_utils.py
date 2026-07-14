from __future__ import annotations

import re
from pathlib import Path


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def sanitize_tts_text(text: str) -> str:
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]", "", text)
    text = re.sub(r"(\^_?\^|T_T|QAQ|QwQ|x_x|X_X)", "。", text)
    text = text.replace("~", "。").replace("～", "。")
    text = text.replace("[笑]", "。").replace("（笑）", "。").replace("(笑)", "。")
    text = re.sub(r"\s*[\r\n]+\s*", "。", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "，", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([，。！？；：、,.!?;:])\s*", r"\1", text)
    text = re.sub(r"[。]{2,}", "。", text)
    text = re.sub(r"[，]{2,}", "，", text)
    text = re.sub(r"([。！？])([，。！？])+", r"\1", text)
    text = text.strip()
    if text and text[-1] not in "。！？.!?":
        text += "。"
    return text
