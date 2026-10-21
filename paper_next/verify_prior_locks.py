"""Read-only verification that the new research did not change frozen studies."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCKS = [
    (ROOT / "results/protocol_lock.json", ROOT),
    (ROOT / "study_v2/results/protocol_lock.json", ROOT / "study_v2"),
    (ROOT / "study_v2/supplement_exact/lock.json", ROOT / "study_v2/supplement_exact"),
    (ROOT / "research_next/results/round1/code_lock.json", ROOT / "research_next"),
    (ROOT / "research_next/results/round1/model_lock.json", ROOT / "research_next"),
    (ROOT / "research_next/results/round1/test_lock.json", ROOT / "research_next"),
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    result = {"checked_unix": time.time(), "locks": [], "failures": []}
    cache = {}
    for lock, base in LOCKS:
        content = json.loads(lock.read_text())
        files = content["files"]
        result["locks"].append({"path": str(lock.relative_to(ROOT)),
                                 "sha256": sha256(lock), "entries": len(files)})
        for name, expected in files.items():
            path = (base / name).resolve()
            try:
                if path not in cache:
                    cache[path] = sha256(path)
                observed = cache[path]
            except OSError as error:
                observed = str(error)
            if observed != expected:
                result["failures"].append({"path": str(path), "expected": expected,
                                           "observed": observed, "lock": str(lock)})
    result["entries"] = sum(lock["entries"] for lock in result["locks"])
    result["unique_files"] = len(cache)
    result["passed"] = not result["failures"]
    output = ROOT / "paper_next/prior_locks_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("entries", "unique_files", "passed")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
