from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

DATABASE_PATH = PROJECT_ROOT / "riftbound_test_inventory.db"