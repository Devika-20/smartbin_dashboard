from pathlib import Path
from typing import List, Dict
from utils import append_jsonl, now_iso

OUTBOX = Path(__file__).parent / "logs" / "dispatch_outbox.jsonl"

def notify_municipality(bin_records: List[Dict], channel: str = "webhook") -> int:
    """Append a dispatch event to outbox. Replace with real SMTP/SMS/Webhook integration."""
    if not bin_records:
        return 0
    event = {
        "ts": now_iso(),
        "channel": channel,
        "count": len(bin_records),
        "bins": [
            {"bin_id": r["bin_id"], "lat": r["lat"], "lon": r["lon"], "area": r.get("area_type", "unknown"), "reason": r.get("alert_reason", "overflow")}
            for r in bin_records
        ],
        "status": "queued"
    }
    append_jsonl(OUTBOX, event)
    return len(bin_records)
