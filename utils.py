import json, os, time, hashlib
from pathlib import Path
from typing import Dict, Any

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def sha1(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

def append_jsonl(path, record: Dict):
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def now_iso():
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def human_pct(x):
    try:
        return f"{float(x):.0f}%"
    except Exception:
        return str(x)
