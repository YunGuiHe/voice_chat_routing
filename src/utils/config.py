import os
from pathlib import Path


def project_file(root_dir: str | Path | None, filename: str) -> Path:
    return Path(root_dir) / filename if root_dir is not None else Path(filename)


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_doubao_key_from_api_md(path: str | Path = "api.md") -> str | None:
    api_path = Path(path)
    if not api_path.exists():
        return None

    lines = [line.strip() for line in api_path.read_text(encoding="utf-8").splitlines()]
    for index, line in enumerate(lines):
        if "豆包" in line:
            for candidate in lines[index + 1 :]:
                if candidate and not candidate.startswith("#"):
                    return candidate
    return None


def read_api_section_value(
    section_name: str,
    label: str,
    path: str | Path = "api.md",
) -> str | None:
    api_path = Path(path)
    if not api_path.exists():
        return None

    in_section = False
    for raw_line in api_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("###"):
            in_section = section_name.lower() in line.lower()
            continue
        if not in_section:
            continue
        if line.startswith(f"{label}：") or line.startswith(f"{label}:"):
            return line.split("：", 1)[1].strip() if "：" in line else line.split(":", 1)[1].strip()
    return None


def get_doubao_api_key(root_dir: str | Path | None = None) -> str:
    load_dotenv(project_file(root_dir, ".env"))
    key = os.getenv("DOUBAO_API_KEY") or read_doubao_key_from_api_md(
        project_file(root_dir, "api.md")
    )
    if not key:
        raise RuntimeError("未找到豆包 API Key。请在 .env 中设置 DOUBAO_API_KEY，或在 api.md 的“豆包”下一行填写。")
    return key


def get_doubao_base_url(root_dir: str | Path | None = None) -> str:
    load_dotenv(project_file(root_dir, ".env"))
    return os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")


def get_doubao_model(root_dir: str | Path | None = None) -> str:
    load_dotenv(project_file(root_dir, ".env"))
    return os.getenv("DOUBAO_MODEL", "doubao-seed-1-6-vision-250815")


def get_deepseek_api_key(root_dir: str | Path | None = None) -> str:
    load_dotenv(project_file(root_dir, ".env"))
    key = os.getenv("DEEPSEEK_API_KEY") or read_api_section_value(
        "DeepSeek",
        "API",
        project_file(root_dir, "api.md"),
    )
    if not key:
        raise RuntimeError(
            "未找到 DeepSeek API Key。请在 .env 中设置 DEEPSEEK_API_KEY，"
            "或在 api.md 的“DeepSeek”部分填写 API。"
        )
    return key


def get_deepseek_base_url(root_dir: str | Path | None = None) -> str:
    load_dotenv(project_file(root_dir, ".env"))
    base_url = (
        os.getenv("DEEPSEEK_BASE_URL")
        or read_api_section_value(
            "DeepSeek",
            "网址",
            project_file(root_dir, "api.md"),
        )
        or "https://api.deepseek.com/v1"
    )
    return f"{base_url.rstrip('/')}/chat/completions"


def get_deepseek_model(root_dir: str | Path | None = None) -> str:
    load_dotenv(project_file(root_dir, ".env"))
    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
