from pathlib import Path

# from dotenv import load_dotenv
from loguru import logger
import sys

# Load environment variables from .env file if it exists
# load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

DASHBOARD_DIR = PROJ_ROOT / "dashboard_data"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Loguru idempotent konfigurieren (kein remove(0) mehr)
logger.configure(handlers=[{
    "sink": sys.stderr,
    "level": "INFO",
    "enqueue": True,
    "backtrace": False,
    "diagnose": False,
}])

# Optional: tqdm-Integration, ohne remove()
try:
    from tqdm import tqdm
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
