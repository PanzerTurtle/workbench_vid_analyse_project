import os
import ollama
from _utility_function import parse_json_response, format_timestamp

from _config import SYSTEM_CONTEXT, USER_CONTEXT, OLLAMA_OPTIONS


def analyze_frame_worker(frame, timestamps, extracted_frames_dir):
    path = os.path.join(extracted_frames_dir, frame)

    if not os.path.exists(path):
        return None, f"Missing file: {frame}"

    try:
        response = ollama.chat(
            model="llava:7b",
            messages=[
                {"role": "system", "content": SYSTEM_CONTEXT},
                {"role": "user", "content": USER_CONTEXT, "images": [path]},
            ],
            options=OLLAMA_OPTIONS,
        )

        parsed = parse_json_response(response["message"]["content"])
        if not parsed:
            return None, response["message"]["content"]

        parsed.update({
            "frame_file": frame,
            "timestamp": format_timestamp(timestamps.get(frame, 0)),
        })

        return parsed, None

    except Exception as e:
        return None, str(e)
    