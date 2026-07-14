import csv
import re
from pathlib import Path


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def read_test_cases(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_csv_cell(value: object) -> object:
    if not isinstance(value, str):
        return value
    return " ".join(value.split())


def normalize_csv_row(row: dict[str, object]) -> dict[str, object]:
    return {key: normalize_csv_cell(value) for key, value in row.items()}


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


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalize_csv_row(row) for row in rows)
