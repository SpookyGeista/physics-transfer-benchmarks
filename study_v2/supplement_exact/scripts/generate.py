"""One prospectively locked supplement: exact measured local-profile replay.

The original frozen study is not modified. The sole stimulus-construction
repair uses the ACTUAL current reference state for cam cancellation, rather
than the desired state. Commands are generated once, saved, and replayed
unchanged by every predictor and the independent backend.
"""
import argparse
import json
from pathlib import Path
import sys
import time

STUDY_ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from decision_study.common import (
    CFG, DT, MASS, cam, profiles, features, history_init, history_update, dump, digest,
)
from decision_study.locking import verify_lock


def verify_supplement():
    original_hash = verify_lock()
    lock_path = SUPPLEMENT_ROOT / "lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("Supplement must be locked before any generation: " + str(lock_path))
    lock = json.loads(lock_path.read_text())
    resolved = set()
    for name, expected in lock["files"].items():
        path = Path(name)
        if not path.is_absolute():
            path = STUDY_ROOT / path
        path = path.resolve()
        assert digest(path) == expected, f"Changed supplement file: {path}"
        resolved.add(path)
    for required in (Path(__file__).resolve(), SUPPLEMENT_ROOT / "config.json", SUPPLEMENT_ROOT / "protocol.md"):
        assert required.resolve() in resolved, f"Supplement lock omits required file: {required}"
    return original_hash, digest(lock_path)


def main():
    # Both locks are checked BEFORE importing or starting the simulator.
    original_hash, supplement_hash = verify_supplement()
    supplement = json.loads((SUPPLEMENT_ROOT / "config.json").read_text())
    amplitudes = supplement["cam_amplitudes"]
    assert amplitudes == [-.005, -.002, .002, .005], "Unexpected supplemental arrangements"
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", choices=CFG["worlds"], default=CFG["worlds"])
    parser.add_argument("--seeds", type=int, nargs="+", choices=CFG["seeds"], default=CFG["seeds"])
    parser.add_argument("--out", type=Path, default=SUPPLEMENT_ROOT / "results/data")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    import numpy as np
    import torch
    from decision_study.reference import Reference
    from decision_study.isaac_rig import IsaacRig

    torch.set_num_threads(1)
    count, steps = CFG["profiles_train"], CFG["steps"]
    rig = IsaacRig(count * len(amplitudes), args.device)
    output = args.out
    for world in args.worlds:
        for seed in args.seeds:
            folder = output / world / f"r{seed}"
            target = folder / "replay.npz"
            if target.exists() or target.with_suffix(".json").exists():
                raise FileExistsError(f"Refusing to replace an existing supplemental replay: {target}")
            started = time.perf_counter()
            pr, source_rows = profiles("train", seed, count=count, steps=steps)
            idx = np.tile(np.arange(count), len(amplitudes))
            amp = np.repeat(amplitudes, count)
            rig.reset()
            reference = Reference(world, len(idx), pr["delta"][idx, 0])
            h = None
            record = {key: [] for key in (
                "features", "normal", "friction", "state", "next_state", "delta",
                "compression_rate", "rate", "u", "d", "force",
            )}
            for t in range(steps):
                state = rig.state()
                x, velocity = state.T
                gap, slope = cam(x, amp)
                # The ONE repaired construction step: cancel cam geometry at
                # the actual reference state, not at a desired float64 state.
                d = pr["delta"][idx, t] - gap
                rate_command = pr["rate"][idx, t] - slope * velocity
                delta = d + gap
                compression_rate = rate_command + slope * velocity
                normal, friction = reference.force(delta, compression_rate, velocity)
                if h is None:
                    h = history_init(delta, velocity, normal)
                u = MASS * (pr["v"][idx, t + 1] - velocity) / DT + slope * normal - friction
                force = u - slope * normal + friction
                next_state = rig.advance(force)
                values = {
                    "features": features(delta, compression_rate, velocity, h, normal),
                    "normal": normal, "friction": friction, "state": state,
                    "next_state": next_state, "delta": delta,
                    "compression_rate": compression_rate, "rate": rate_command,
                    "u": u, "d": d, "force": force,
                }
                for key, value in values.items():
                    record[key].append(np.array(value, copy=True))
                h = history_update(h, delta, velocity, normal)
            arrays = {key: np.stack(values, axis=1) for key, values in record.items()}
            rows = [{**source_rows[j], "mode": "AB", "amplitude": float(amp[k]),
                     "id": f"{source_rows[j]['id']}-cam{amp[k]:+.3f}"}
                    for k, j in enumerate(idx)]
            calibration = STUDY_ROOT / "results/data" / world / f"r{seed}" / "train.npz"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            dump(target.with_suffix(".json"), {
                "world": world, "replicate": seed, "split": "replay", "rows": rows,
                "dt": DT, "steps": steps, "seconds": time.perf_counter() - started,
                "backend": "Isaac Lab / native CPU PhysX", "oracle_stimulus": True,
                "oracle_geometry_construction": "actual measured reference x,v; one-time saved commands",
                "local_profiles_reused": True, "novelty_axis": "mechanical arrangement only",
                "calibration_source": str(calibration), "calibration_sha256": digest(calibration),
                "supplement": "one prospective exact-local construction repair",
                "original_lock_sha256": original_hash, "supplement_lock_sha256": supplement_hash,
                "generator_sha256": digest(__file__),
            })
            print("SAVED_SUPPLEMENT", world, seed, len(rows), steps,
                  f"{time.perf_counter() - started:.2f}s", flush=True)
    print("SUPPLEMENT_GENERATION_COMPLETE", flush=True)
    app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
