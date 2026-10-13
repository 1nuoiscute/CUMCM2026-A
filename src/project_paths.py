"""Repository paths for the post-contest publication; numerical models are unchanged."""
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
