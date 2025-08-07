import logging
from src.utils.path import get_project_root
from pathlib import Path


root_dir = get_project_root()
log_dir = root_dir / "logs"

def setup_logging(log_path:str)-> logging.Logger:
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True) 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{log_path}.log"),
            logging.StreamHandler()
        ]
    )
    