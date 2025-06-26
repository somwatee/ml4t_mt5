# data_pipeline/utils.py
import yaml
import logging
from pathlib import Path

def load_config(path: str) -> dict:
    """โหลดค่าต่าง ๆ จาก config.yaml เป็น dict (อ่านด้วย encoding UTF-8)"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} ไม่พบไฟล์")
    text = p.read_text(encoding='utf-8')
    return yaml.safe_load(text)

def get_logger(name: str) -> logging.Logger:
    """สร้าง logger ชื่อ name พร้อมตั้งระดับ INFO และ format"""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log
