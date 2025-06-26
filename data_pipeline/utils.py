import yaml
import logging
from pathlib import Path

def load_config(path: str) -> dict:
    """
    Load YAML config from given path.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that prints to console.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
