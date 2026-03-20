import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SQL_DB_PATH = os.getenv("SQL_DB_PATH", str(DATA_DIR / "weather.db"))