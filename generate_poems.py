from __future__ import annotations

import json
import os
import re
import html
import shutil
import tempfile
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from groq import Groq
import wandb

try:
    import weave
except Exception:  # pragma: no cover - weave is optional at runtime
    weave = None

from weather_queries import get_tomorrow_weather
from config import SQL_DB_PATH

OUTPUT_FILE = Path("outputs") / "poems.md"
HTML_OUTPUT_FILE = Path("docs") / "index.html"
load_dotenv()
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "m6-assignment")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")


class TrackingSession:
    def __init__(self, run: Any | None, artifact_dir: Path | None = None):
        self.run = run
        self.artifact_dir = artifact_dir

    @property
    def enabled(self) -> bool:
        return self.run is not None and self.artifact_dir is not None

    def write_json(self, name: str, payload: Any) -> Path:
        if not self.enabled:
            raise RuntimeError("TrackingSession is not enabled.")
        path = self.artifact_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
        if self.artifact_dir is not None:
            shutil.rmtree(self.artifact_dir, ignore_errors=True)


def _init_tracking_session(period: str | None) -> TrackingSession:
    mode = os.getenv("WANDB_MODE")
    api_key = os.getenv("WANDB_API_KEY")

    if mode not in ("offline", "disabled") and not api_key:
        return TrackingSession(run=None, artifact_dir=None)

    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="groq-poem-generation",
            tags=["groq", "weather", "poems"],
            config={
                "groq_model": DEFAULT_MODEL,
                "weather_period": period or "auto",
            },
            mode=mode,
        )
    except Exception:
        return TrackingSession(run=None, artifact_dir=None)

    if weave is not None and os.getenv("WEAVE_DISABLED", "0") != "1":
        weave_project = WANDB_PROJECT if not WANDB_ENTITY else f"{WANDB_ENTITY}/{WANDB_PROJECT}"
        try:
            weave.init(weave_project)
        except Exception:
            pass

    artifact_dir = Path(tempfile.mkdtemp(prefix="wandb-artifacts-"))
    return TrackingSession(run=run, artifact_dir=artifact_dir)


def _to_serializable(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return {"raw": str(response)}


def _log_input_artifact(
    tracking: TrackingSession,
    period: str | None,
    window_start: str,
    window_end: str,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
) -> None:
    if not tracking.enabled:
        return

    payload = {
        "period": period,
        "window_start": window_start,
        "window_end": window_end,
        "rows": rows,
        "summary": summary,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    input_path = tracking.write_json("input_context.json", payload)

    artifact = wandb.Artifact(name="groq-weather-inputs", type="dataset")
    artifact.add_file(str(input_path), name="input_context.json")

    db_path = Path(SQL_DB_PATH)
    if db_path.exists():
        artifact.add_file(str(db_path), name="weather.db")

    tracking.run.log_artifact(artifact)


def _log_groq_artifact(
    tracking: TrackingSession,
    response: Any,
    parsed_payload: dict[str, Any],
    raw_content: str,
) -> None:
    if not tracking.enabled:
        return

    response_data = _to_serializable(response)
    response_path = tracking.write_json("groq_response.json", response_data)
    parsed_path = tracking.write_json("parsed_payload.json", parsed_payload)
    raw_path = tracking.artifact_dir / "raw_content.txt"
    raw_path.write_text(raw_content, encoding="utf-8")

    artifact = wandb.Artifact(name="groq-generation-response", type="model-output")
    artifact.add_file(str(response_path), name="groq_response.json")
    artifact.add_file(str(parsed_path), name="parsed_payload.json")
    artifact.add_file(str(raw_path), name="raw_content.txt")
    tracking.run.log_artifact(artifact)

    usage = response_data.get("usage") if isinstance(response_data, dict) else None
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        metrics: dict[str, float | int] = {}
        if isinstance(prompt_tokens, (int, float)):
            metrics["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, (int, float)):
            metrics["completion_tokens"] = completion_tokens
        if isinstance(total_tokens, (int, float)):
            metrics["total_tokens"] = total_tokens
        if metrics:
            tracking.run.log(metrics)


def _log_output_artifact(
    tracking: TrackingSession,
    result: dict[str, Any],
    markdown_path: Path,
    html_path: Path,
) -> None:
    if not tracking.enabled:
        return

    result_path = tracking.write_json("result.json", result)

    artifact = wandb.Artifact(name="groq-weather-poems", type="result")
    artifact.add_file(str(result_path), name="result.json")
    artifact.add_file(str(markdown_path), name=markdown_path.name)
    artifact.add_file(str(html_path), name=html_path.name)
    tracking.run.log_artifact(artifact)

    tracking.run.log(
        {
            "rows_in_window": len(result.get("rows", [])),
            "locations_in_summary": len(result.get("summary", [])),
        }
    )


if weave is not None:
    @weave.op()
    def _call_groq_completion(
        api_key: str,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return _to_serializable(response)
else:
    def _call_groq_completion(
        api_key: str,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return _to_serializable(response)


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



def generate_bilingual_poems(period: str | None = None, tracking: TrackingSession | None = None) -> dict[str, Any]:
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

    api_key = os.environ["GROQ_API_KEY"]

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

    if tracking is not None:
        _log_input_artifact(
            tracking=tracking,
            period=period,
            window_start=window.start.isoformat(),
            window_end=window.end.isoformat(),
            rows=rows,
            summary=summary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    response_data = _call_groq_completion(
        api_key=api_key,
        model=DEFAULT_MODEL,
        temperature=0.8,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    choices = response_data.get("choices", []) if isinstance(response_data, dict) else []
    if not choices:
        raise ValueError("Groq returned no choices.")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") or ""
    payload = _extract_json_payload(content)

    if tracking is not None:
        _log_groq_artifact(
            tracking=tracking,
            response=response_data,
            parsed_payload=payload,
            raw_content=content,
        )

    result = {
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
        "rows": rows,
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
    tracking = _init_tracking_session(period=period)
    try:
        result = generate_bilingual_poems(period=period, tracking=tracking)
        markdown_path = write_markdown_output(result)
        html_path = write_html_output(result)
        _log_output_artifact(tracking, result, markdown_path, html_path)
        print(f"Poems written to {markdown_path}")
        print(f"HTML written to {html_path}")
    finally:
        tracking.finish()


if __name__ == "__main__":
    main()
