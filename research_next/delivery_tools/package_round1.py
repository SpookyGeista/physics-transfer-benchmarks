"""Package the completed local study and its legacy dependencies; never publish.

Delivery utility added after the experimental source lock. It cannot train,
simulate, select models, or change the predefined statistical analysis.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NEXT = ROOT / "research_next"
OUT = NEXT / "deliverables"
sys.path.insert(0, str(NEXT / "src"))
from round1_lock import verify_test_lock


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect():
    allowed = ["HANDOFF.md", "README.md", "STATUS.md", "pyproject.toml",
               "UEBERGABE_null-physics-transfer.md", "configs", "src", "scripts",
               "tests", "reports", "results", "videos", "study_v2", "research_next"]
    files = []
    for top in allowed:
        path = ROOT / top
        candidates = [path] if path.is_file() else path.rglob("*")
        for p in candidates:
            rel = p.relative_to(ROOT)
            if (not p.is_file() or p.is_symlink()
                    or any(s.startswith(".") or s in {"deliverables", "__pycache__"} for s in rel.parts)
                    or p.suffix in {".pyc", ".zip"} or p.name.endswith((".tar.gz", ".tar.zst"))):
                continue
            if (not p.resolve().is_relative_to(ROOT.resolve())
                    or any((ROOT / Path(*rel.parts[:i])).is_symlink()
                           for i in range(1, len(rel.parts) + 1))):
                raise RuntimeError(f"Unexpected symbolic or external input: {p}")
            files.append(p)
    return sorted(set(files))


def package(name, files, role, snapshot):
    manifest = {str(p.relative_to(ROOT)): snapshot[str(p.relative_to(ROOT))] for p in files}
    declaration = dict(role=role, extraction_directory="physics-transfer",
                       original_root=str(ROOT),
                       dependency_note="Installed Isaac/IsaacLab/PyTorch and MuJoCo runtimes are not bundled. Legacy provenance expects the original filesystem mapping.",
                       files=manifest)
    payload = (json.dumps(declaration, indent=2, sort_keys=True) + "\n").encode()
    path = OUT / name
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for p in files:
                    if sha(p) != manifest[str(p.relative_to(ROOT))]:
                        raise RuntimeError(f"Input changed while packaging: {p}")
                    info = archive.gettarinfo(str(p), arcname="physics-transfer/" + str(p.relative_to(ROOT)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with p.open("rb") as f:
                        archive.addfile(info, f)
                info = tarfile.TarInfo("physics-transfer/DELIVERY_MANIFEST.json")
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))
    # Stream every member back through SHA-256; do not merely check a tar header.
    observed = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            relative = str(Path(member.name).relative_to("physics-transfer"))
            if relative == "DELIVERY_MANIFEST.json":
                assert archive.extractfile(member).read() == payload
                continue
            h = hashlib.sha256()
            f = archive.extractfile(member)
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
            observed[relative] = h.hexdigest()
    assert observed == manifest, "Archive content does not match manifest"
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                bytes=path.stat().st_size, files=len(files), content_verified=True,
                role=role)


def main():
    test_lock = verify_test_lock()
    result = NEXT / "results/round1"
    decision = json.loads((result / "analysis/decision.json").read_text())
    assert decision["complete"] and decision["prediction_rows"] == decision["controller_rows"] == 7168
    evidence = json.loads((result / "numerical_validity.json").read_text())
    assert evidence["complete"] is True, "Finish numerical evaluation, whether passing or failing"
    assert (NEXT / "reports/conclusion.md").is_file(), "Finish the synthesis before delivery"
    assert (NEXT / "reports/round1_support.md").is_file()
    for world in ["maxwell_boucwen", "boucwen_maxwell"]:
        assert (NEXT / f"videosround1/round1_{world}_r1_clean_episode0.mp4").is_file()
    if (OUT.is_symlink() or not OUT.resolve().is_relative_to(ROOT.resolve())
            or NEXT.is_symlink()):
        raise RuntimeError("Delivery output must be a local nonsymlink directory")
    OUT.mkdir(exist_ok=True)
    for name in ["physics-transfer-round1-source.tar.gz",
                 "physics-transfer-round1-complete.tar.gz", "SHA256SUMS.json"]:
        if (OUT / name).exists():
            raise FileExistsError(f"Preserve existing delivery artifact: {OUT / name}")
    files = collect()
    snapshot = {str(p.relative_to(ROOT)): sha(p) for p in files}
    excluded_parts = {"results", "videos", "videosround1", "runs", "figures", "figuresround1"}
    source = [p for p in files if not set(p.relative_to(ROOT).parts) & excluded_parts
              and (p.suffix in {".py", ".md", ".json", ".toml", ".sh"}
                   or p.relative_to(ROOT) == Path("configs/requirements-observed.txt"))]
    artifacts = [package("physics-transfer-round1-source.tar.gz", source,
                         "Sources, configuration and reports; complete data package required for evaluation", snapshot),
                 package("physics-transfer-round1-complete.tar.gz", files,
                         "Full study including prior v1/v2 dependencies, models, raw measurements, figures and videos", snapshot)]
    for p in files:
        if sha(p) != snapshot[str(p.relative_to(ROOT))]:
            raise RuntimeError(f"Study changed during delivery; retain archives for review: {p}")
    assert verify_test_lock() == test_lock
    with (OUT / "SHA256SUMS.json").open("x") as f:
        json.dump(dict(test_lock_sha256=test_lock, archives=artifacts), f, indent=2)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
