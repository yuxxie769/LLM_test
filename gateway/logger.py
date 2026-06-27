from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("ab") as f:
        f.write(orjson.dumps(record))
        f.write(b"\n")
