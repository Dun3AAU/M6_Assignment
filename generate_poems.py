from __future__ import annotations

import json
import os
import re
import html
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from groq import Groq

from weather_queries import get_tomorrow_weather

OUTPUT_FILE = Path("outputs") / "poems.md"
HTML_OUTPUT_FILE = Path("docs") / "index.html"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
load_dotenv()


def _build_weather_context(summary: list[dict[str, Any]]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _extract_json_payload(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        raise ValueError("Groq returned empty content.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Groq response did not contain valid JSON.")



def generate_bilingual_poems(period: str | None = None) -> dict[str, Any]:
    """
    Generate Danish and English poems for the selected tomorrow period.

    period supports:
    - None (default: now + 24h, 1-hour window)
    - "HH:MM"
    - "HH:MM-HH:MM"
    """
    window, rows, summary = get_tomorrow_weather(period)

    if not rows:
        raise ValueError(
            "No weather rows found for the requested tomorrow period. "
            "Run fetch.py first or choose another period."
        )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    system_prompt = (
        "You are a poetic weather assistant. "
        "Write two separate short poems comparing weather across all provided locations. "
        "One poem must be in Danish and one in English. "
        "Both poems must describe differences and explicitly recommend where it is nicest to kitesurf tomorrow. "
        "Keep each poem to 6-10 lines."
    )

    user_prompt = (
        f"Time window: {window.start.isoformat()} to {window.end.isoformat()}\n"
        "Weather summary per location (JSON):\n"
        f"{_build_weather_context(summary)}\n\n"
        "Output STRICTLY as JSON with keys: danish_poem, english_poem, surf_recommendation."
    )

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.8,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or ""
    payload = _extract_json_payload(content)

    result = {
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "summary": summary,
        "danish_poem": payload.get("danish_poem", ""),
        "english_poem": payload.get("english_poem", ""),
        "surf_recommendation": payload.get("surf_recommendation", ""),
    }
    return result



def write_markdown_output(result: dict[str, Any], destination: Path = OUTPUT_FILE) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Weather Poems",
        "",
        f"Time window: {result['window_start']} → {result['window_end']}",
        "",
        "## Weather Summary",
        "",
        "| Location | Avg Temp (°C) | Avg Precip (%) | Avg Wind (m/s) | Min Temp | Max Temp |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for item in result["summary"]:
        lines.append(
            f"| {item['location_name']} | {item['avg_temperature_c']} | "
            f"{item['avg_precipitation_probability']} | {item['avg_wind_speed_ms']} | "
            f"{item['min_temperature_c']} | {item['max_temperature_c']} |"
        )

    lines.extend(
        [
            "",
            "## Danish Poem",
            "",
            result["danish_poem"],
            "",
            "## English Poem",
            "",
            result["english_poem"],
            "",
            "## Surf Recommendation",
            "",
            result["surf_recommendation"],
            "",
        ]
    )

    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _poem_to_html(poem_text: str) -> str:
    escaped = html.escape(poem_text.strip())
    return escaped.replace("\n", "<br>")


def write_html_output(result: dict[str, Any], destination: Path = HTML_OUTPUT_FILE) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = "\n".join(
        (
            "<tr>"
            f"<td>{html.escape(str(item['location_name']))}</td>"
            f"<td>{item['avg_temperature_c']}</td>"
            f"<td>{item['avg_precipitation_probability']}</td>"
            f"<td>{item['avg_wind_speed_ms']}</td>"
            f"<td>{item['min_temperature_c']}</td>"
            f"<td>{item['max_temperature_c']}</td>"
            "</tr>"
        )
        for item in result["summary"]
    )

    html_content = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Weather Poems</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 2rem auto; max-width: 960px; padding: 0 1rem; }}
        h1, h2 {{ margin-top: 1.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: right; }}
        th:first-child, td:first-child {{ text-align: left; }}
        .meta {{ color: #444; margin-bottom: 1rem; }}
        .poem {{ white-space: normal; background: #fafafa; border: 1px solid #eee; padding: 0.75rem; border-radius: 8px; }}
    </style>
</head>
<body>
    <h1>Weather Poems</h1>
    <p class=\"meta\">Time window: {html.escape(result['window_start'])} → {html.escape(result['window_end'])}</p>

    <h2>Weather Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Location</th>
                <th>Avg Temp (°C)</th>
                <th>Avg Precip (%)</th>
                <th>Avg Wind (m/s)</th>
                <th>Min Temp</th>
                <th>Max Temp</th>
            </tr>
        </thead>
        <tbody>
            {summary_rows}
        </tbody>
    </table>

    <h2>Danish Poem</h2>
    <div class=\"poem\">{_poem_to_html(result['danish_poem'])}</div>

    <h2>English Poem</h2>
    <div class=\"poem\">{_poem_to_html(result['english_poem'])}</div>

    <h2>Surf Recommendation</h2>
    <p>{html.escape(result['surf_recommendation'])}</p>
</body>
</html>
"""

    destination.write_text(html_content, encoding="utf-8")
    return destination



def main() -> None:
    period = os.getenv("WEATHER_PERIOD")
    result = generate_bilingual_poems(period=period)
    markdown_path = write_markdown_output(result)
    html_path = write_html_output(result)
    print(f"Poems written to {markdown_path}")
    print(f"HTML written to {html_path}")


if __name__ == "__main__":
    main()
