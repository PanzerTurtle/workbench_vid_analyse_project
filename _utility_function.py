import re
import json
from datetime import datetime


def read_num_in_filename(filename: str) -> int:
    match = re.findall(r"\d+", filename)
    return int(match[0]) if match else 0


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def parse_json_response(text: str) -> dict | None:
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        text = text.strip()

        if text and not text.endswith("}"):
            text += "}" * (text.count("{") - text.count("}"))

        if text.count('"') % 2 != 0:
            text += '"'

        return json.loads(text)
    except Exception:
        return None


def log_parse_error(frame: str, error: str, output="error.log"):
    with open(output, "a") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Frame: {frame}\n")
        f.write(f"Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{error}\n\n")