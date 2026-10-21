"""Fresh EXPLORATORY isolated profiles; no material or learner imports.

Every motion segment is a sin² pulse and every boundary has zero derivative.
The normal channel and tangential velocity use separate random streams.
"""
import numpy as np

DT = 0.0025
DURATION = 6.0
DEFAULT_COUNTS = {"train": 24, "val": 8}
SOURCE_BASES = {"train": 7_100_000, "val": 7_200_000}


def pulse(t, start, duration, amplitude):
    """C1 compact pulse and its exact derivative (SI seconds)."""
    value = np.zeros_like(t, dtype=np.float64)
    rate = np.zeros_like(t, dtype=np.float64)
    mask = (t > start) & (t < start + duration)
    phase = np.pi * (t[mask] - start) / duration
    value[mask] = amplitude * np.sin(phase) ** 2
    rate[mask] = amplitude * np.pi / duration * np.sin(2 * phase)
    return value, rate


def _normal_segments(index, rng):
    if index < 2:
        sign = 1 if index == 0 else -1
        # Opposite preloads, identical three-second hold, mirrored reversal.
        return [(0., 1., sign * .008), (4., 1., -sign * .008),
                (5., 1., sign * .004)], [(1., 4.)], "mirrored_preload_pair"
    sign = int(rng.choice([-1, 1]))
    preload_time = float(rng.uniform(.7, 1.3))
    hold = float(rng.uniform(.5, 3.0))
    amplitude = float(rng.uniform(.004, .008))
    segments = [(0., preload_time, sign * amplitude)]
    start = preload_time + hold
    count = int(rng.integers(2, 5))
    raw = rng.uniform(.8, 1.2, count)
    durations = raw / raw.sum() * (DURATION - start)
    for j, length in enumerate(durations):
        # Alternating full/minor loops; no material-dependent timescale.
        scale = float(rng.uniform(.35, 1.0))
        segments.append((float(start), float(length),
                         float(sign * (-1) ** (j + 1) * amplitude * scale)))
        start += length
    return segments, [(preload_time, preload_time + hold)], "random_preload_hold_minor_loops"


def _velocity_segments(rng):
    holds = rng.uniform(.5, 2.0, 2)
    raw = rng.uniform(.8, 1.2, 3)
    durations = raw / raw.sum() * (DURATION - holds.sum())
    sign = int(rng.choice([-1, 1]))
    segments, intervals = [], []
    start = 0.
    for j, length in enumerate(durations):
        amplitude = float(rng.uniform(.08, .28)) * sign * (-1) ** j
        segments.append((float(start), float(length), amplitude))
        start += length
        if j < 2:
            intervals.append((float(start), float(start + holds[j])))
            start += holds[j]
    return segments, intervals


def profiles(split, seed=0, count=None, steps=2400, source_base=None):
    """Return old-schema delta/rate/v/x arrays, plus auditable stimulus rows.

    Arrays contain steps+1 samples; a reduced steps value truncates the same
    six-second design for smoke tests, never accelerates its time constants.
    Source bases and replicate indices are explicit development provenance.
    """
    if split not in DEFAULT_COUNTS:
        raise ValueError("Only exploratory train/val splits are defined")
    count = DEFAULT_COUNTS[split] if count is None else count
    if not 1 <= count <= 999 or not 1 <= steps <= 2400 or seed < 0:
        raise ValueError("Require 1<=count<=999, 1<=steps<=2400, seed>=0")
    base = SOURCE_BASES[split] if source_base is None else int(source_base)
    if base < 0:
        raise ValueError("source_base must be nonnegative")
    t = np.arange(steps + 1, dtype=np.float64) * DT
    arrays = {key: [] for key in ("delta", "rate", "v", "x")}
    rows = []
    for index in range(count):
        source_seed = base + seed * 1000 + index
        normal_rng = np.random.default_rng(np.random.SeedSequence([source_seed, 0]))
        # Equal velocity for the preload pair exposes its different load history.
        velocity_seed = base + seed * 1000 + (0 if split == "train" and index < 2 else index)
        velocity_rng = np.random.default_rng(np.random.SeedSequence([velocity_seed, 1]))
        # The diagnostic pair belongs to train only; every validation waveform
        # is fresh, including validation indices 0 and 1.
        normal_segments, normal_holds, kind = _normal_segments(index if split == "train" else index + 2, normal_rng)
        velocity_segments, velocity_holds = _velocity_segments(velocity_rng)
        delta, rate, velocity = np.full_like(t, .02), np.zeros_like(t), np.zeros_like(t)
        for start, duration, amplitude in normal_segments:
            bump, derivative = pulse(t, start, duration, amplitude)
            delta += bump
            rate += derivative
        for start, duration, amplitude in velocity_segments:
            velocity += pulse(t, start, duration, amplitude)[0]
        x = np.zeros_like(t)
        x[1:] = DT * np.cumsum(velocity[1:])
        for key, value in (("delta", delta), ("rate", rate), ("v", velocity), ("x", x)):
            arrays[key].append(value)
        rows.append({"id": f"development-{split}-r{seed}-p{index}",
                     "profile": index, "replicate": seed, "stimulus_seed": source_seed,
                     "normal_rng_stream": 0, "velocity_rng_stream": 1,
                     "velocity_seed": velocity_seed, "source_base": base,
                     "kind": kind, "duration_seconds": DURATION,
                     "normal_segments_start_duration_amplitude": normal_segments,
                     "normal_hold_intervals": normal_holds,
                     "velocity_segments_start_duration_amplitude": velocity_segments,
                     "velocity_zero_hold_intervals": velocity_holds,
                     "split_role": "exploratory_development_only"})
    result = {key: np.asarray(value) for key, value in arrays.items()}
    assert np.all((result["delta"] >= .012 - 1e-14) & (result["delta"] <= .028 + 1e-14))
    assert np.all(np.isfinite(np.concatenate(list(result.values()), axis=1)))
    return result, rows
