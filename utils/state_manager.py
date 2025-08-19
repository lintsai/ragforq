"""Simple state persistence for selected models and indexing status."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict

STATE_FILE = Path('config/runtime_state.json')

DEFAULT_STATE = {
    "language_model": None,
    "embedding_model": None,
    "last_index_full_ts": None,
    "last_index_incremental_ts": None,
    "last_index_doc_count": None,
    "updated_at": None
}

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return DEFAULT_STATE.copy()

def save_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    state = load_state()
    state.update(patch)
    state['updated_at'] = int(time.time())
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        pass
    return state

def record_full_index(doc_count: int):
    save_state({
        'last_index_full_ts': int(time.time()),
        'last_index_doc_count': doc_count
    })

def record_incremental_index(doc_count: int):
    save_state({
        'last_index_incremental_ts': int(time.time()),
        'last_index_doc_count': doc_count
    })
