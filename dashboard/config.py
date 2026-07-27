from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

DATABASE_PATH = PROJECT_ROOT / "riftbound.db"
API_BASE_URL = "http://127.0.0.1:8000"
