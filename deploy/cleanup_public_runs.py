"""Preview cleanup by default. Stop metasymbo.service before passing --apply."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import time

WORKSPACE = Path(__file__).resolve().parents[1] / "WebInterface/.public"


def expired_paths(workspace, days, now=None):
    cutoff = (time.time() if now is None else now) - days * 86400
    paths, active = [], set()
    for directory in (workspace / "runs").glob("*"):
        if directory.is_symlink() or not re.fullmatch(r"[a-f0-9]{32}", directory.name):
            continue
        status = directory / "status.json"
        if not status.is_file() or status.is_symlink():
            continue
        item = json.loads(status.read_text())
        if item.get("state") in {"queued", "running"}:
            active.add(directory.name)
        elif item.get("state") in {"complete", "failed", "interrupted"} and item.get("updated", time.time()) < cutoff:
            paths.append(directory)
    for metadata in (workspace / "candidates").glob("*.json"):
        if metadata.is_symlink() or not re.fullmatch(r"(?:import-[a-f0-9]{32}|(?:prediction|generation)-[a-f0-9]{32}-[0-9]+)\.json", metadata.name):
            continue
        item = json.loads(metadata.read_text())
        if item.get("run_id") in active or item.get("created", time.time()) >= cutoff:
            continue
        paths.append(metadata)
        candidate = metadata.with_suffix(".npz")
        if candidate.is_file() and not candidate.is_symlink():
            paths.append(candidate)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("Retention must be at least one day")
    if WORKSPACE.is_symlink() or not WORKSPACE.is_dir():
        parser.error("The expected public workspace is missing or is a symlink")
    with (WORKSPACE / ".instance.lock").open("a") as lock:
        if args.apply:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("Stop metasymbo.service before applying cleanup")
        paths = expired_paths(WORKSPACE, args.days)
        for path in paths:
            if not path.resolve().is_relative_to(WORKSPACE.resolve()):
                raise ValueError("Refusing to remove a path outside the public workspace")
            print(("Remove " if args.apply else "Would remove ") + str(path.relative_to(WORKSPACE)))
            if args.apply:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        print(f"{len(paths)} expired entries; research results and the private .local workspace are excluded.")


if __name__ == "__main__":
    main()
