"""Shared atomic state and safe diagnostics for the web process and its worker."""
import json
import logging
from pathlib import Path
import traceback
import uuid


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    for name in ["httpx", "httpcore", "openai"]:
        logging.getLogger(name).setLevel(logging.WARNING)


def log_exception(logger, run_id, exc):
    # Provider exception messages and source lines can contain credentials or
    # prompts. Keep the exception type and stack locations, never those values.
    frames = " -> ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                         for f in traceback.extract_tb(exc.__traceback__))
    logger.error("run=%s exception=%s traceback=%s", run_id, type(exc).__name__, frames)
