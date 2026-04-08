"""App configuration from environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RULES_DIR = DATA_DIR / "rules"
MANUALS_CACHE_DIR = DATA_DIR / "manuals_cache"
SCRAPED_MANUALS_DIR = DATA_DIR / "scraped_manuals"

for d in (DATA_DIR, RULES_DIR, MANUALS_CACHE_DIR, SCRAPED_MANUALS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Unstructured (optional, for robust multi-format manual parsing)
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")
UNSTRUCTURED_API_URL = os.getenv(
    "UNSTRUCTURED_API_URL",
    "https://api.unstructuredapp.io/general/v0/general",
)

# Google Places (optional, for nearby mechanics links)
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# Allowed extraction components (closed schema)
ALLOWED_COMPONENTS = {
    "engine_oil",
    "oil_filter",
    "tire_rotation",
    "brake_inspection",
    "battery",
    "coolant",
    "transmission_fluid",
    "air_filter",
    "fuel_filter",
    "belt_inspection",
    "hose_inspection",
    "daily_check",
    "weekly_check",
}

VALID_DISTANCE_UNITS = {"km", "miles"}
VALID_TIME_UNITS = {"months", "weeks", "hours", "days", "years"}
