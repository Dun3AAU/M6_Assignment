from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
from typing import Any

from config import SQL_DB_PATH

DEFAULT_TZ = ZoneInfo("Europe/Copenhagen")


@dataclass(frozen=True)
class TimeWindow:
    start: datetime
    end: datetime



def _parse_time(value: str) -> time:
    return datetime.strptime(value.strip(), "%H:%M").time()



def resolve_tomorrow_window(period: str | None = None, tz: ZoneInfo = DEFAULT_TZ) -> TimeWindow:
    """
    Resolve a query window for tomorrow.

    Args:
        period:
            - None: use current time + 24 hours, rounded down to the nearest hour (1-hour window).
            - "HH:MM": use that hour tomorrow (1-hour window).
            - "HH:MM-HH:MM": use that full range tomorrow (end is exclusive).
    """
    now_local = datetime.now(tz)
    tomorrow = now_local.date() + timedelta(days=1)

    if period is None:
        target = now_local + timedelta(hours=24)
        start = target.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
        return TimeWindow(start=start, end=end)

    if "-" in period:
        start_text, end_text = period.split("-", maxsplit=1)
        start_time = _parse_time(start_text)
        end_time = _parse_time(end_text)
        start = datetime.combine(tomorrow, start_time, tzinfo=tz)
        end = datetime.combine(tomorrow, end_time, tzinfo=tz)
        if end <= start:
            raise ValueError("For a range HH:MM-HH:MM, end must be after start.")
        return TimeWindow(start=start, end=end)

    start_time = _parse_time(period)
    start = datetime.combine(tomorrow, start_time, tzinfo=tz)
    end = start + timedelta(hours=1)
    return TimeWindow(start=start, end=end)



def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQL_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn



def fetch_weather_rows(period: str | None = None, include_unknown: bool = False) -> tuple[TimeWindow, list[dict[str, Any]]]:
    """
    Fetch hourly weather rows in the selected time window for tomorrow.
    """
    window = resolve_tomorrow_window(period)
    tomorrow = window.start.date()

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                date,
                location_name,
                hour,
                day,
                month,
                year,
                temperature_2m,
                precipitation_probability,
                wind_speed_10m,
                wind_direction_10m,
                latitude,
                longitude,
                elevation,
                timezone
            FROM weather_data
            WHERE year = ? AND month = ? AND day = ?
            ORDER BY location_name, date
            """,
            (tomorrow.year, tomorrow.month, tomorrow.day),
        )
        rows = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        row_dt = datetime(
            int(row["year"]),
            int(row["month"]),
            int(row["day"]),
            int(row["hour"]),
            tzinfo=DEFAULT_TZ,
        )
        if not include_unknown and (row.get("location_name") in (None, "", "Unknown")):
            continue
        if window.start <= row_dt < window.end:
            filtered_rows.append(row)

    return window, filtered_rows



def summarize_by_location(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate rows to one summary per location for prompt building.
    """
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        location_name = row.get("location_name") or "Unknown"
        summary = grouped.setdefault(
            str(location_name),
            {
                "location_name": str(location_name),
                "samples": 0,
                "temperature_sum": 0.0,
                "precipitation_sum": 0.0,
                "wind_speed_sum": 0.0,
                "temp_min": float("inf"),
                "temp_max": float("-inf"),
            },
        )

        temperature = float(row["temperature_2m"])
        precipitation = float(row["precipitation_probability"])
        wind_speed = float(row["wind_speed_10m"])

        summary["samples"] += 1
        summary["temperature_sum"] += temperature
        summary["precipitation_sum"] += precipitation
        summary["wind_speed_sum"] += wind_speed
        summary["temp_min"] = min(summary["temp_min"], temperature)
        summary["temp_max"] = max(summary["temp_max"], temperature)

    result: list[dict[str, Any]] = []
    for summary in grouped.values():
        samples = int(summary["samples"])
        if samples == 0:
            continue

        result.append(
            {
                "location_name": summary["location_name"],
                "samples": samples,
                "avg_temperature_c": round(summary["temperature_sum"] / samples, 1),
                "avg_precipitation_probability": round(summary["precipitation_sum"] / samples, 1),
                "avg_wind_speed_ms": round(summary["wind_speed_sum"] / samples, 1),
                "min_temperature_c": round(summary["temp_min"], 1),
                "max_temperature_c": round(summary["temp_max"], 1),
            }
        )

    return sorted(result, key=lambda item: item["location_name"])



def get_tomorrow_weather(period: str | None = None, include_unknown: bool = False) -> tuple[TimeWindow, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    High-level helper used by poem generation.
    Returns: (time_window, raw_rows, per_location_summary)
    """
    window, rows = fetch_weather_rows(period, include_unknown=include_unknown)
    summary = summarize_by_location(rows)
    return window, rows, summary
