#!/usr/bin/env python
"""
Backfill missing .model files for already trained traditional RAG vector DB folders.
Use this when older training runs created index files (index.faiss / index.pkl) but
no .model metadata file, preventing model selection in UI.

Usage (PowerShell):
  python scripts/backfill_model_files.py

It will:
  * Scan vector_db root for folders starting with 'ollama@'
  * For each folder missing .model but having index.faiss & index.pkl
    create a .model file with inferred fields.
  * If only one of the index files exists, still create .model (status partial_data=True).
  * Skip folders already having .model.

Limitations:
  Original model names with ':' were normalized to '_' in folder name; we cannot
  reconstruct them perfectly. Underscored names are stored as-is.
"""
from pathlib import Path
import json
import datetime
import sys
import logging

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from utils.vector_db_manager import vector_db_manager  # noqa

logger = logging.getLogger("backfill_model_files")

base_path = vector_db_manager.base_path

if not base_path.exists():
    logger.error(f"Vector DB path not found: {base_path}")
    sys.exit(1)

created = 0
skipped = 0
errors = 0

for folder in base_path.iterdir():
    if not folder.is_dir():
        continue
    name = folder.name
    if not name.startswith('ollama@'):
        continue
    model_file = folder / '.model'
    if model_file.exists():
        skipped += 1
        continue

    # Determine presence of vector data
    has_faiss = (folder / 'index.faiss').exists()
    has_pkl = (folder / 'index.pkl').exists()

    # Parse folder name using existing manager logic
    parsed = vector_db_manager.parse_folder_name(name)
    if not parsed:
        logger.warning(f"Cannot parse folder name, skipping: {name}")
        errors += 1
        continue

    model_name, embedding_name, version = parsed

    info = {
        "OLLAMA_MODEL": model_name,
        "OLLAMA_EMBEDDING_MODEL": embedding_name,
        "created_at": datetime.datetime.now().isoformat(),
    }
    if version and version != 'current':
        info['version'] = version
    if not (has_faiss and has_pkl):
        info['partial_data'] = True
        info['note'] = 'One or both index files missing at time of backfill.'

    try:
        with open(model_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        created += 1
        logger.info(f"Created .model for {name}")
    except Exception as e:
        errors += 1
        logger.error(f"Failed creating .model for {name}: {e}")

logger.info(f"Backfill complete: created={created} skipped(existing)={skipped} errors={errors}")
print(json.dumps({"created": created, "skipped": skipped, "errors": errors}, ensure_ascii=False))
