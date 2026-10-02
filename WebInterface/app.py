"""MetaSymbO workbench; keep this process behind the supplied public proxy."""
import argparse
from contextlib import asynccontextmanager
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
import uuid
import zipfile

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import plotly
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .geometry import read_lattice
from .runtime import write_json, configure_logging, log_exception

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("METASYMBO_RESULTS", ROOT / "results")).resolve()
LOCAL = Path(os.environ.get("METASYMBO_WORKSPACE", HERE / ".local")).resolve()
CHECKPOINT = Path(os.environ.get("METASYMBO_CHECKPOINT", ROOT / "checkpoints/vae_cond_128_beta001_dis_same_100_frac")).resolve()
DATASET = Path(os.environ.get("METASYMBO_DATASET", Path.home() / "datasets/metamaterial/LatticeModulus")).resolve()
PUBLIC = os.environ.get("METASYMBO_PUBLIC", "0") == "1"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"} | set(os.environ.get("METASYMBO_ALLOWED_HOSTS", "").split(","))
MAX_PROMPT = 2000
MAX_ATTEMPTS = 2
RUN_TIMEOUT = 900
BUSY = "The MetaSymbO demo is currently busy. Please try again shortly."
LOGGER = logging.getLogger("metasymbo")
FILES = {}
CACHE = {}
LOCK = threading.Lock()
SUBMIT_LOCK = threading.Lock()
READ_LIMITS = {}
ACTIVE_PROCESS = None
SHUTDOWN = threading.Event()


def read_runs():
    return [json.loads(path.read_text()) for path in (LOCAL / "runs").glob("*/status.json")]


def public_status(item):
    return {key: value for key, value in item.items() if key != "owner"}


def owner(request):
    return request.state.owner if PUBLIC else "local"


def storage_available():
    # Only the disposable workspace counts. Research results are never deleted.
    files = [p for p in LOCAL.rglob("*") if p.is_file() and not p.is_symlink()]
    if len(files) >= 8000 or sum(p.stat().st_size for p in files) >= 1024 ** 3:
        raise HTTPException(503, "Demo storage is full. Please try again after maintenance.")


def reserve_quota(request, action):
    """Called under SUBMIT_LOCK. Persist quotas so restart cannot reset spending."""
    if not PUBLIC:
        return
    now = time.time()
    path = LOCAL / "limits.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    entries = [e for e in entries if e["time"] > now - 86400]
    address = request.client.host if request.client else "unknown"
    # Uvicorn trusts forwarding headers only from the loopback Caddy proxy.
    client = hashlib.sha256(address.encode()).hexdigest()
    hourly, daily, global_daily = {"generation": (2, 6, 24), "prediction": (6, 24, 120), "import": (10, 30, 200)}[action]
    matching = [e for e in entries if e["action"] == action]
    own = [e for e in matching if e["client"] == client]
    if len(matching) >= global_daily:
        raise HTTPException(429, "The demo's daily allowance is used. Please try again tomorrow.", headers={"Retry-After": "3600"})
    if len(own) >= daily or sum(e["time"] > now - 3600 for e in own) >= hourly:
        raise HTTPException(429, "Your request limit is reached. Please try again later.", headers={"Retry-After": "3600"})
    entries.append(dict(time=now, action=action, client=client))
    write_json(path, entries)


def file_allowed(path, directory):
    return not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()) and path.is_file()


def visible(path, request):
    if not PUBLIC or path.is_relative_to(RESULTS):
        return True
    metadata = path.with_suffix(".json")
    return file_allowed(metadata, LOCAL) and json.loads(metadata.read_text()).get("owner") == owner(request)


def index_files(workspace_only=False):
    global FILES
    files = {key: value for key, value in FILES.items() if value[1].startswith("results/")} if workspace_only else {}
    directories = [("workspace", LOCAL / "candidates")]
    if not workspace_only:
        directories.insert(0, ("results", RESULTS))
    for prefix, directory in directories:
        if directory.exists():
            for path in sorted(directory.rglob("*.npz")):
                if not file_allowed(path, directory):
                    continue
                relative = f"{prefix}/{path.relative_to(directory).as_posix()}"
                identifier = hashlib.sha256(relative.encode()).hexdigest()[:24]
                files[identifier] = (path, relative)
    FILES = files


def candidate(identifier, request):
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{24}", identifier):
        raise HTTPException(404, "Lattice not found.")
    if identifier not in FILES:
        # A worker commits a candidate before reporting completion; make it
        # available immediately, including while later attempts are running.
        index_files(workspace_only=True)
    if identifier not in FILES:
        raise HTTPException(404, "This lattice is no longer available.")
    path, relative = FILES[identifier]
    directory = RESULTS if relative.startswith("results/") else LOCAL / "candidates"
    if not file_allowed(path, directory) or not visible(path, request):
        raise HTTPException(404, "Lattice not found.")
    stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    if identifier not in CACHE or CACHE[identifier][0] != stamp:
        try:
            item = read_lattice(path)
        except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as exc:
            raise HTTPException(422, "This archive has unsupported or invalid numeric geometry.") from exc
        item.update(id=identifier, name=path.stem, source=relative,
                    group=relative.rsplit("/", 2)[-2], evidence="Predicted" if item["predictions"] else "Geometry only")
        if len(CACHE) >= 256:
            CACHE.pop(next(iter(CACHE)), None)
        CACHE[identifier] = (stamp, item)
    result = dict(CACHE[identifier][1])
    metadata = path.with_suffix(".json")
    provenance = json.loads(metadata.read_text()) if file_allowed(metadata, directory) else None
    # Serve only fields produced by the workbench, never arbitrary sidecar data.
    allowed = {"run_id", "kind", "parent_id", "prompt", "seed", "checkpoint", "property_units", "component_mapping", "created", "settings", "generation_settings", "supervisor_score", "suggested_prompt", "geometry_fingerprint", "note"}
    result["provenance"] = {k: v for k, v in provenance.items() if k in allowed} if provenance else None
    return result


@lru_cache(maxsize=1)
def model_capabilities():
    # Read an existing local credential without copying it into run metadata.
    for env_file in [ROOT / ".env", HERE / ".env.local"]:
        if PUBLIC and env_file.is_file() and env_file.stat().st_mode & 0o077:
            raise RuntimeError("Credential files must have owner-only permissions")
        if not os.environ.get("OPENAI_API_KEY") and env_file.is_file():
            for line in env_file.read_text().splitlines():
                name, separator, raw = line.strip().removeprefix("export ").partition("=")
                if separator and name.strip() == "OPENAI_API_KEY":
                    value = shlex.split(raw, comments=True)
                    if len(value) == 1:
                        os.environ["OPENAI_API_KEY"] = value[0]
    model_ready = all(importlib.util.find_spec(name) is not None for name in ["torch", "torch_geometric", "torch_scatter", "torch_cluster", "openai"])
    checkpoints = all((CHECKPOINT / name).is_file() for name in ["best_ae_model.pt", "best_predictor_model.pt"])
    dataset = (DATASET / "data/processed/data.pt").is_file()
    key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    return dict(prediction=model_ready and checkpoints, generation=model_ready and checkpoints and dataset and key,
                model_runtime=model_ready, checkpoints=checkpoints, dataset=dataset, api_key_configured=key,
                simulation=False)


def capabilities():
    return dict(model_capabilities(), count=len(FILES), public_demo=PUBLIC,
                max_prompt=MAX_PROMPT, max_attempts=MAX_ATTEMPTS,
                busy=any(item["state"] in {"queued", "running"} for item in read_runs()))


@asynccontextmanager
async def lifespan(app):
    SHUTDOWN.clear()
    LOCAL.mkdir(parents=True, exist_ok=True)
    # ponytail: one web process and one worker; flock refuses accidental second
    # instances. Move to a shared job queue if multi-process serving is needed.
    instance = (LOCAL / ".instance.lock").open("a")
    fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (LOCAL / "candidates").mkdir(parents=True, exist_ok=True)
    (LOCAL / "runs").mkdir(exist_ok=True)
    index_files()
    # A process may have died before committing its final status. Never claim
    # such a run completed, and never automatically resubmit paid API requests.
    for path in (LOCAL / "runs").glob("*/status.json"):
        status = json.loads(path.read_text())
        if status["state"] in {"queued", "running"}:
            status.update(state="interrupted", stage="Server restarted. Review saved candidates before starting another run.")
            write_json(path, status)
    caps = model_capabilities()
    LOGGER.info("startup public=%s prediction=%s generation=%s OPENAI_API_KEY %s", PUBLIC, caps["prediction"], caps["generation"], "configured" if caps["api_key_configured"] else "missing")
    try:
        yield
    finally:
        SHUTDOWN.set()
        process = ACTIVE_PROCESS
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        instance.close()


app = FastAPI(title="MetaSymbO", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def protect_requests(request: Request, call_next):
    if request.url.hostname not in ALLOWED_HOSTS:
        return JSONResponse({"detail": "Unknown demo address."}, status_code=403)
    if request.method not in {"GET", "HEAD"}:
        origin = request.headers.get("origin")
        if (origin and origin != f"{request.url.scheme}://{request.headers.get('host')}") or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Cross-site writes are not allowed."}, status_code=403)
    if PUBLIC and request.url.path.startswith("/api/") and request.url.path != "/api/health":
        minute = int(time.time() // 60)
        client = request.client.host if request.client else "unknown"
        if READ_LIMITS.get("minute") != minute:
            READ_LIMITS.clear()
            READ_LIMITS["minute"] = minute
        READ_LIMITS[client] = READ_LIMITS.get(client, 0) + 1
        READ_LIMITS["total"] = READ_LIMITS.get("total", 0) + 1
        if READ_LIMITS[client] > 120 or READ_LIMITS["total"] > 2400:
            return JSONResponse({"detail": "Too many requests. Please wait a minute."}, status_code=429, headers={"Retry-After": "60"})
    token = request.cookies.get("metasymbo_session", "")
    new_session = not re.fullmatch(r"[a-f0-9]{64}", token)
    if new_session:
        token = secrets.token_hex(32)
    request.state.owner = hashlib.sha256(token.encode()).hexdigest()
    try:
        response = await call_next(request)
    except Exception as exc:
        log_exception(LOGGER, "web", exc)
        response = JSONResponse({"detail": "The request could not complete. Please try again later."}, status_code=500)
    if PUBLIC and (new_session or request.url.path == "/api/status") and request.url.path.startswith("/api/") and request.url.path != "/api/health":
        response.set_cookie("metasymbo_session", token, max_age=30 * 86400, httponly=True, samesite="lax", secure=request.url.scheme == "https")
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD"}:
            return await self.app(scope, receive, send)
        limit = 4 * 1024 * 1024 if scope["path"] == "/api/import" else 16384
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > limit:
                return await JSONResponse({"detail": "Request too large."}, status_code=413)(scope, receive, send)
            messages.append(message)
            if not message.get("more_body"):
                break

        async def replay():
            return messages.pop(0) if messages else await receive()
        await self.app(scope, replay, send)


app.add_middleware(BodyLimitMiddleware)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return JSONResponse({"detail": "Invalid request. Check the prompt, attempt limit, and numeric values."}, status_code=422)


@app.get("/api/health")
def health():
    caps = model_capabilities()
    return dict(status="ok", web=True, generation_available=caps["generation"], prediction_available=caps["prediction"])


@app.get("/api/status")
def status():
    return capabilities()


@app.get("/api/candidates")
def list_candidates(request: Request, q: str = Query("", max_length=200), group: str = Query("", max_length=200), offset: int = 0, limit: int = 12):
    offset, limit = max(0, offset), min(48, max(1, limit))
    entries = [(identifier, path, relative) for identifier, (path, relative) in FILES.items() if visible(path, request)]
    groups = sorted({relative.rsplit("/", 2)[-2] for _, _, relative in entries})
    entries = [row for row in entries if q.casefold() in row[2].casefold()
               and (not group or row[2].rsplit("/", 2)[-2] == group)]
    entries.sort(key=lambda row: ("/lattices/" not in row[2], not row[2].startswith("workspace/"), row[2].count("/"), row[2]))
    items = []
    for identifier, path, relative in entries[offset:offset + limit]:
        try:
            items.append(candidate(identifier, request))
        except HTTPException as exc:
            items.append(dict(id=identifier, name=path.stem, source=relative, error=exc.detail))
    return dict(items=items, total=len(entries), groups=groups, offset=offset)


@app.get("/api/candidates/{identifier}")
def get_candidate(identifier: str, request: Request):
    return candidate(identifier, request)


@app.get("/api/candidates/{identifier}/download")
def download(identifier: str, request: Request):
    candidate(identifier, request)
    path, _ = FILES[identifier]
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.post("/api/import", status_code=201)
async def import_lattice(request: Request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 4 * 1024 * 1024:
            raise HTTPException(413, "Choose an NPZ file smaller than 4 MB.")
        chunks.append(chunk)
    content = b"".join(chunks)
    try:
        read_lattice(content)
    except Exception as exc:
        raise HTTPException(422, "The file is not a supported lattice NPZ. Check its numeric arrays and cell parameters.") from exc
    name = f"import-{uuid.uuid4().hex}"
    path = LOCAL / "candidates" / f"{name}.npz"
    with SUBMIT_LOCK:
        storage_available()
        reserve_quota(request, "import")
        write_json(path.with_suffix(".json"), {"kind": "import", "owner": owner(request), "created": time.time(), "note": "Imported geometry; original run history may be unavailable."})
        path.write_bytes(content)
        index_files(workspace_only=True)
    identifier = next(key for key, (file, _) in FILES.items() if file == path)
    return candidate(identifier, request)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    kind: str
    candidate_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{24}$")
    prompt: str = Field(default="", max_length=MAX_PROMPT)
    logic_mode: str = "union"
    attempts: int = Field(default=1, ge=1, le=MAX_ATTEMPTS)
    seed: int = Field(default=42, ge=0, le=2147483647)
    condition: list[float] | None = None

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, value):
        if value not in {"prediction", "generation"}:
            raise ValueError("Choose prediction or generation.")
        return value

    @field_validator("logic_mode")
    @classmethod
    def valid_logic(cls, value):
        if value not in {"union", "mix", "int", "neg"}:
            raise ValueError("Unknown symbolic operation.")
        return value

    @field_validator("condition")
    @classmethod
    def valid_condition(cls, value):
        if value is not None and (len(value) != 12 or not np.isfinite(value).all() or any(abs(v) > float(np.finfo(np.float32).max) for v in value)):
            raise ValueError("Conditioning requires exactly 12 finite values.")
        return value


def run_worker(directory):
    global ACTIVE_PROCESS
    with LOCK:
        path = directory / "status.json"
        status = json.loads(path.read_text())
        status.update(state="running", stage="Loading model", updated=time.time())
        write_json(path, status)
        started = time.monotonic()
        LOGGER.info("run=%s state=running kind=%s", status["id"], status["kind"])
        env = {**os.environ, "METASYMBO_CHECKPOINT": str(CHECKPOINT), "METASYMBO_DATASET": str(DATASET), "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2"}
        try:
            if SHUTDOWN.is_set():
                raise OSError("Server stopping")
            ACTIVE_PROCESS = subprocess.Popen([sys.executable, "-m", "WebInterface.worker", str(directory)],
                                              cwd=ROOT, env=env)
            if ACTIVE_PROCESS.wait(timeout=RUN_TIMEOUT):
                raise subprocess.CalledProcessError(ACTIVE_PROCESS.returncode, "model worker")
        except (subprocess.SubprocessError, OSError) as exc:
            log_exception(LOGGER, status["id"], exc)
            if ACTIVE_PROCESS and ACTIVE_PROCESS.poll() is None:
                ACTIVE_PROCESS.kill()
                ACTIVE_PROCESS.wait()
            status = json.loads(path.read_text())
            if status["state"] not in {"complete", "failed"}:
                status.update(state="interrupted" if SHUTDOWN.is_set() else "failed", stage="Run timed out" if isinstance(exc, subprocess.TimeoutExpired) else "The model process stopped before completion.", updated=time.time())
                write_json(path, status)
        finally:
            ACTIVE_PROCESS = None
            LOGGER.info("run=%s state=%s duration=%.1fs", status["id"], json.loads(path.read_text())["state"], time.monotonic() - started)
        index_files(workspace_only=True)


@app.post("/api/runs", status_code=202)
def start_run(body: RunRequest, request: Request):
    caps = capabilities()
    if not caps[body.kind]:
        message = "Prediction needs the model environment and checkpoints." if body.kind == "prediction" else "Generation needs the model environment, dataset, checkpoints, and an active API key."
        raise HTTPException(409, message)
    payload = dict(body.model_dump(), owner=owner(request))
    if body.kind == "prediction":
        item = candidate(body.candidate_id, request)
        if item["nodes"] > 100 or item["zero_length_struts"] or not item["struts"]:
            raise HTTPException(422, "This predictor requires at most 100 nodes and nonzero struts.")
        payload["source_path"] = str(FILES[body.candidate_id][0])
    elif not body.prompt.strip():
        raise HTTPException(422, "Describe a design goal first.")
    # One active run prevents accidental double-submission and unbounded local work.
    with SUBMIT_LOCK:
        if SHUTDOWN.is_set() or any(item["state"] in {"queued", "running"} for item in read_runs()):
            raise HTTPException(409, BUSY, headers={"Retry-After": "30"})
        storage_available()
        reserve_quota(request, body.kind)
        identifier = uuid.uuid4().hex
        directory = LOCAL / "runs" / identifier
        write_json(directory / "request.json", payload)
        write_json(directory / "status.json", dict(id=identifier, kind=body.kind, prompt=body.prompt,
                   owner=owner(request), candidate_id=body.candidate_id, state="queued", stage="Queued", created=time.time(), updated=time.time(), candidates=[]))
        LOGGER.info("run=%s state=queued kind=%s", identifier, body.kind)
        threading.Thread(target=run_worker, args=(directory,), daemon=True).start()
    return {"id": identifier}


@app.get("/api/runs")
def list_runs(request: Request):
    items = [public_status(item) for item in read_runs() if not PUBLIC or item.get("owner") == owner(request)]
    return sorted(items, key=lambda item: item["created"], reverse=True)[:100]


@app.get("/api/runs/{identifier}")
def get_run(identifier: str, request: Request):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise HTTPException(404, "Run not found.")
    path = LOCAL / "runs" / identifier / "status.json"
    if not file_allowed(path, LOCAL / "runs"):
        raise HTTPException(404, "Run not found.")
    item = json.loads(path.read_text())
    if PUBLIC and item.get("owner") != owner(request):
        raise HTTPException(404, "Run not found.")
    return public_status(item)


@app.get("/plotly.min.js")
def plotly_bundle():
    return FileResponse(Path(plotly.__file__).parent / "package_data/plotly.min.js", media_type="text/javascript")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(HERE / "static/index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn
    configure_logging()
    uvicorn.run(app, host="127.0.0.1", port=args.port, proxy_headers=True, forwarded_allow_ips="127.0.0.1", access_log=False, server_header=False, limit_concurrency=100)
