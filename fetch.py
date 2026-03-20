import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import sqlite3
from pathlib import Path
from typing import Any

from config import SQL_DB_PATH

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

KNOWN_COORDINATES = [
    (57.1155, 8.6172, "Hanstholm"),
    (57.0386, 10.0027, "Aalborg Øst"),
    (56.5670, 9.0271, "Skive"),
]

COORDINATE_TOLERANCE = 0.03


def resolve_place_name(latitude: float, longitude: float) -> str:
    for known_lat, known_lon, place_name in KNOWN_COORDINATES:
        if abs(latitude - known_lat) <= COORDINATE_TOLERANCE and abs(longitude - known_lon) <= COORDINATE_TOLERANCE:
            return place_name
    return "Unknown"


def get_responses():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": [57.1155, 57.0386, 56.567],
        "longitude": [8.6172, 10.0027, 9.0271],
        "hourly": ["temperature_2m", "precipitation_probability", "wind_speed_10m", "wind_direction_10m"],
        "timezone": "Europe/Berlin",
        "forecast_days": 2,
        "wind_speed_unit": "ms",
    }
    return openmeteo.weather_api(url, params=params)


def init_db() -> sqlite3.Connection:
    db_path = Path(SQL_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            location_name TEXT,
            hour INTEGER,
            day INTEGER,
            month INTEGER,
            year INTEGER,
            temperature_2m REAL,
            precipitation_probability REAL,
            wind_speed_10m REAL,
            wind_direction_10m REAL,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            timezone TEXT
        )
        """
    )

    existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(weather_data)")}
    if "location_name" not in existing_columns:
        cursor.execute("ALTER TABLE weather_data ADD COLUMN location_name TEXT")

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_weather_data_date_location
        ON weather_data (date, latitude, longitude, location_name)
        """
    )

    conn.commit()
    return conn


def store_data(conn: sqlite3.Connection, responses: list[Any]) -> int:
    sql = """
        INSERT OR IGNORE INTO weather_data (
            date, location_name, hour, day, month, year, temperature_2m, precipitation_probability,
            wind_speed_10m, wind_direction_10m, latitude, longitude, elevation, timezone
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows_inserted = 0
    cursor = conn.cursor()

    for response in responses:
        latitude = float(response.Latitude())
        longitude = float(response.Longitude())
        location_name = resolve_place_name(latitude, longitude)
        elevation = float(response.Elevation())
        timezone = str(response.Timezone())

        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation_probability = hourly.Variables(1).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(3).ValuesAsNumpy()

        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        for index, date_value in enumerate(dates):
            cursor.execute(
                sql,
                (
                    date_value.isoformat(),
                    location_name,
                    date_value.hour,
                    date_value.day,
                    date_value.month,
                    date_value.year,
                    float(hourly_temperature_2m[index]),
                    float(hourly_precipitation_probability[index]),
                    float(hourly_wind_speed_10m[index]),
                    float(hourly_wind_direction_10m[index]),
                    latitude,
                    longitude,
                    elevation,
                    timezone,
                ),
            )
            rows_inserted += cursor.rowcount

    conn.commit()
    return rows_inserted


def main() -> None:
    responses = get_responses()
    conn = init_db()
    try:
        inserted = store_data(conn, responses)
        print(f"Inserted {inserted} new rows into weather_data")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
