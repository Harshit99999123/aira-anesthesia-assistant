import json
import time
from typing import Any

from config.settings import settings


def log_event(event: str, **fields: Any) -> None:
    if not settings.structured_logs:
        return
    payload = {
        "ts": time.time(),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=True))
