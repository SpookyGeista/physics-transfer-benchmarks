#!/usr/bin/env python3
"""Read existing timing metadata; never run a model or read prediction scores.

Default v1 deliberately leaves Main2 inference pending. After its completion use
--include-main2-complete --out-prefix paper_study/reports/resource_accounting_v2.
Both output files must be new. CPU process time is unavailable, not wall time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_study"
SCALARS = {"status", "seconds", "wall_seconds", "started_unix", "ended_unix",
           "pid", "batches", "episodes", "rows", "fit_batches", "expected_rows",
           "contexts", "native_branches", "triplets", "policy_rows", "branch_rows",
           "fit_count", "optimizer_starts", "successful_starts", "optimizer_successes",
           "budget_cases", "optimizer_budget_cases", "budget_exhausted_episodes",
           "optimizer_success_episodes", "completed_batches", "metric_rows",
           "peak_rss_MiB", "error", "failed_batch", "attempt_index"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = PAPER / path
        payload = path.read_bytes()
        self.hashes[str(path.relative_to(ROOT))] = sha(payload)
        return json.loads(payload)

    def source(self, path):
        path = ROOT / path
        self.hashes[str(path.relative_to(ROOT))] = sha(path.read_bytes())

    def verify(self):
        for path, digest in self.hashes.items():
            if sha((ROOT / path).read_bytes()) != digest:
                raise RuntimeError(f"Input changed during resource snapshot: {path}")


def metadata(j):
    return {k: v for k, v in j.items() if k in SCALARS}


def interval_summary(records):
    intervals = [(r["started_unix"], r["started_unix"] + r["seconds"]) for r in records]
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    return dict(sum_worker_wall_seconds=sum(r["seconds"] for r in records),
                reconstructed_elapsed_span_seconds=max(b for _, b in intervals)-min(a for a, _ in intervals),
                maximum_overlapping_recorded_workers=peak,
                scope="Timestamp envelope from started_unix + measured seconds; includes launch gaps, excludes external checks/aggregation. Not CPU time.")


def collect(include_main2=False):
    inp = Inputs()
    entries = []

    def process(name, relative, scope, parallel, allow_failed=False):
        raw = inp.read(relative)
        if raw.get("status") != "complete" and not allow_failed:
            raise ValueError(f"Incomplete process: {relative}")
        entries.append(dict(name=name, source="paper_study/"+relative,
                            wall_seconds=raw.get("seconds", raw.get("wall_seconds")),
                            cpu_process_seconds=None, scope=scope, parallelism=parallel,
                            metadata=metadata(raw)))
        return raw

    process("Main1 native", "results/native_main1/process.json",
            "12,288 native episodes, 256 sequential batches; includes Isaac app launch within recorded scope.",
            "One native process, 48 independent bodies per batch; CPU PhysX.")
    process("Main1 inference/evaluation", "results/observer_main1/process.json",
            "294,912 episode-method rows, 8 banks, 3 forecast methods; filtering, forecasts, scoring and serialization. Existing fitted models/priors excluded.",
            "One serial Python evaluator, vectorized particles; no process pool in the evaluator.")
    process("Main3 native", "results/decision_test_native/process.json",
            "32 triplets, 1,152 contexts, 3,456 physical branches; includes Isaac app launch.",
            "One native process; 108 independent branch bodies per triplet.")

    bankrows, all_attempts = [], []
    for bank in range(1, 9):
        folder = PAPER / f"results/decision_test_evaluation/banks/r{bank}"
        pointer = inp.read(folder / "process.json")
        if pointer["status"] != "complete":
            raise ValueError(f"Main3 bank {bank} incomplete")
        attempts = sorted((folder / "attempts").glob("attempt_*/process.json"))
        if len(attempts) != pointer["attempt_index"]:
            raise ValueError("Missing Main3 attempt timing history")
        for path in attempts:
            row = metadata(inp.read(path))
            row["bank"] = bank
            row["source"] = str(path.relative_to(ROOT))
            all_attempts.append(row)
        selected = Path(pointer["selected_attempt"])
        if not selected.is_absolute():
            selected = ROOT / selected
        row = metadata(inp.read(selected / "process.json"))
        plan = inp.read(selected / "planning_complete.json")
        if row["status"] != "complete" or plan["plans"] != 32:
            raise ValueError("Incomplete Main3 planning timing")
        row.update(bank=bank, planning_wall_seconds=plan["seconds"],
                   remaining_wall_seconds=row["seconds"]-plan["seconds"],
                   plans=plan["plans"])
        bankrows.append(row)
    main3summary = interval_summary(all_attempts)
    aggregate = inp.read("results/decision_test_evaluation/process.json")
    if aggregate["status"] != "complete" or aggregate["policy_rows"] != 294912:
        raise ValueError("Main3 aggregate incomplete")

    for p in (256, 1024, 4096):
        process(f"PI particle development P{p}", f"results/costed_pi_convergence_v1/p{p}/process.json",
                "432 physical episodes, 12 cells, 1,296 forecast rows (prior/posterior/privileged); noisy bank 1. Separate sequential resolution run.",
                "One CPU Python process; vectorized particle dynamics.")
    process("PI bounded MLE1 development", "results/costed_pi_mle_v1/process.json",
            "432 optimizations/forecasts, incremental over cached P4096 causal starting hypotheses; 60 nfev / 256 calls / 3 s guard.",
            "One CPU Python process, episodes processed serially.")
    mlemetas = [inp.read(p) for p in sorted((PAPER / "results/costed_pi_mle_v1/predictions").rglob("*.json"))]
    if len(mlemetas) != 12:
        raise ValueError("Expected all 12 MLE timing sidecars")
    mlecost = dict(cached_particle_filter_plus_forecast_seconds=sum(j["source_particle_inference_plus_forecast_seconds"] for j in mlemetas),
                   incremental_fit_seconds=sum(j["incremental_fit_seconds"] for j in mlemetas),
                   incremental_forecast_seconds=sum(j["incremental_forecast_seconds"] for j in mlemetas),
                   scope="P4096 source timing includes particle future forecasts, so it is an upper bound for causal start construction, not filter-only timing. Prior-bank construction and original calibration remain outside this timing. Do not count cached initialization as free or count the same cached run twice.")
    process("PI MLE two-start resolution", "results/costed_pi_mle_resolution_v1/process.json",
            "72 episodes, 144 starts; 120 nfev / 512 calls / 6 s per start; cached P4096 initialization excluded from this incremental timer.",
            "One CPU Python process; two sequential starts per episode.")
    process("PI comparative timestep diagnostic", "results/pi_comparative_timestep_v1/process.json",
            "72 episodes × 3 grids, 432 MLE starts, 864 method rows; includes fresh P1024 filtering and MLE. Public canonical prior construction/lock excluded; ideal sensors, noisy bank 1. Wall guards 6/12/24 s by grid.",
            "One CPU Python process, sequential cells/grids.")
    process("Scalar decision resolution", "results/decision_resolution_v1/process.json",
            "Four P/K settings on 288 development contexts; numerical results complete, original process FAILED only at optional Markdown/tabulate rendering. Separate completion audit retained.",
            "One CPU Python process.", allow_failed=True)
    dataset = inp.read("multistate_learning/results/dataset_manifest.json")
    fits = sorted((PAPER / "multistate_learning/results/models").glob("*/r*/*/diagnostics.json"))
    if len(fits) != 24:
        raise ValueError("Expected all 24 local fits")
    fittimes = [inp.read(p)["wall_seconds"] for p in fits]
    fitresources = dict(fit_count=24, dataset_wall_seconds=dataset["wall_seconds"],
                        sum_individual_fit_wall_seconds=sum(fittimes),
                        max_individual_fit_wall_seconds=max(fittimes),
                        median_individual_fit_wall_seconds=statistics.median(fittimes),
                        cpu_process_seconds=None, configured_workers=2,
                        total_original_fit_phase_wall_seconds=None,
                        scope="Individual optimizer wall timers overlap across two fit workers; sum is not CPU time. Original full fitting wall measurement (2.03 s) appears in RESULTS.md but has no standalone process-clock JSON; not reconstructed as a precise total here.")
    process("Local PI learning reproduction", "results/local_learning_reproduction_v1/process.json",
            "Full isolated repeat: six tests, lock, data generation, 24 fits, summarization and verification; identical coefficients; not 24 fresh independent scientific replicates.",
            "One orchestration process; fit stage uses 2 CPU workers.")
    process("Main2 native only", "results/native_main2/process.json",
            "3,456 native episodes in 96 batches; completed physical generation only. No Main2 prediction score accessed.",
            "One native CPU-PhysX process, 36 bodies per batch.")

    main2 = dict(status="pending_for_this_v1_snapshot", inference_included=False,
                 reason="Main2 inference was not complete when this accounting scope was commissioned. Default intentionally omits it, even on a later rerun. Use the explicit completion option and a fresh output prefix.")
    if include_main2:
        main2banks = []
        for bank in (1, 2, 3):
            relative = f"results/observer_main2/bank{bank}/process.json"
            j = process(f"Main2 inference bank {bank}", relative,
                        "Completed noisy-bank model evaluation; process metadata only, no scores read. Includes P1024 start computation and both MLE optimizations.",
                        "One worker per bank; up to three independent concurrent workers.")
            main2banks.append(metadata(j))
        main2 = dict(status="complete_process_metadata_only", inference_included=True,
                     banks=main2banks, sum_worker_wall_seconds=sum(j["seconds"] for j in main2banks),
                     scope="Sum is worker wall time, not CPU time or total project elapsed time. Main2 score/statistics audit is outside this report.")

    c1 = inp.read("configs/main1_native.json")
    c2 = inp.read("costed_benchmark/main2_native.json")
    c3 = inp.read("decision_test/config.json")
    physical = dict(main1=dict(dt_s=c1["dt"], probes=c1["probes"], future_s=c1["future_seconds"]),
                    main2=dict(dt_s=c2["dt"], fixture_s=c2["preparation_seconds"], probes=c2["probes"], future_s=c2["future_seconds"]),
                    main3=dict(dt_s=c3["dt"], common_prefix_s=c3["common_seconds"],
                               optional_continuation_s=c3["continuation_seconds"],
                               future_s=c3["future_seconds"], sensor_period_s=c3["dt"]*c3["observation_steps"]),
                    scope="Simulated durations fixed by the stored clock/configuration, not elapsed computer time. Hidden 4 s fixture preparation is a separate material stage, not a 4 s online robot operation. No time-speedup or realtime claim is formed by dividing vectorized episode totals by wall time.")
    for path in ("paper_study/scripts/report_resources.py", "paper_study/scripts/evaluate_native_v2.py",
                 "paper_study/scripts/generate_native_v2.py", "paper_study/decision_test/evaluate.py",
                 "paper_study/decision_test/generate.py", "paper_study/multistate_learning/fit.py",
                 "paper_study/costed_benchmark/convergence/mle_protocol.md",
                 "paper_study/costed_benchmark/convergence/mle_resolution_protocol.md"):
        inp.source(path)
    inp.verify()
    return dict(version=2 if include_main2 else 1, created_utc=datetime.now(timezone.utc).isoformat(),
                measured_quantity="Elapsed wall seconds from perf_counter/monotonic; CPU process time not recorded.",
                rows=entries, main3_banks=bankrows, main3_all_attempts=all_attempts,
                main3_summary=main3summary, main3_aggregation_wall_seconds=None,
                mle_initialization=mlecost, local_fits=fitresources, main2=main2,
                physical_clocks=physical, source_sha256=inp.hashes,
                limitations=["Not a complete all-project CPU/energy account: prior generation, historical scalar learning, some imports/JIT/checks, aggregation, audits, figures and videos are not all timed in these process records.",
                             "Wall time includes host contention, Python/JIT/IO within each timer; recorded scopes differ. No hardware-normalized method-speed ranking.",
                             "No NPZ, score CSV, summary loss table or model execution was used by this script."])


def render(j):
    lines = ["# Ressourcenbilanz aus vorhandenen Prozessartefakten", "", f"Snapshot v{j['version']}; {j['created_utc']}.", "",
             "**Alle hier als Laufzeit bezeichneten Werte sind Wandzeiten. CPU-Prozesszeiten wurden nicht gespeichert.** Physische Probenzeiten und Rechnerzeiten sind getrennte Größen. Diese Bilanz begründet keine Echtzeitfähigkeit.", "",
             "| Phase | Wandzeit [s] | Umfang / Messgrenze | Parallelität |", "|---|---:|---|---|"]
    for row in j["rows"]:
        wall = "nicht gespeichert" if row["wall_seconds"] is None else f"{row['wall_seconds']:.3f}"
        lines.append(f"| {row['name']} | {wall} | {row['scope']} | {row['parallelism']} |")
    lines += ["", "## Main3: alle acht Bankworker", "", "| Bank | Gesamte Wandzeit [s] | Bis Planung komplett [s] | Rest [s] |", "|---|---:|---:|---:|"]
    for b in j["main3_banks"]:
        lines.append(f"| {b['bank']} | {b['seconds']:.3f} | {b['planning_wall_seconds']:.3f} | {b['remaining_wall_seconds']:.3f} |")
    s = j["main3_summary"]
    lines += ["", f"Summe der Worker-Wandzeiten: {s['sum_worker_wall_seconds']:.3f} s. Aus Startzeitstempeln und Dauern rekonstruierte Spanne: {s['reconstructed_elapsed_span_seconds']:.3f} s, maximal {s['maximum_overlapping_recorded_workers']} überlappende Worker. Die vorhandenen Starts bilden zwei Gruppen mit je vier Banken. Die Spanne enthält die Pause zwischen Gruppen; Aggregation und externe Prüfungen sind darin nicht vollständig erfasst. Keine dieser Größen ist CPU-Zeit.",
              "", "Die Planungszeit ist ein Präfix derselben gesamten Bank-Wandzeit; sie darf nicht noch einmal addiert werden. Pro Bank wurden 32 Pläne für 1.152 Kontexte sowie 36.864 Policy- und 3.456 Branchzeilen berechnet. Alle gespeicherten Versuchszeiten, einschließlich etwaiger Fehlversuche, stehen im JSON.",
              "", "## Physische Uhren", "", "Main1: 0/50/150/300 ms Probe; anschließend 250 ms Zukunft. Main2: 150 oder 300 ms Probe, anschließend 400 ms Zukunft. Main3: gemeinsamer 50-ms-Präfix, danach 0 oder 100 ms Zusatzpräparation, anschließend 250 ms Zukunft. Basisschritt 2,5 ms, reguläre Sensorperiode 25 ms. Dies sind die gespeicherten Simulationsuhren. Die separate versteckte Materialpräparation dauert 4 s; ihre ideale Fixture ist kein zeitlich so ausgeführter Roboterreset.",
              "", "## MLE-Initialisierung und lokale Fits", ""]
    m = j["mle_initialization"]
    lines += [f"Der Basis-MLE-Lauf verwendet bereits berechnete P4096-Hypothesen. Die zugehörigen zwölf Inferenz-plus-Partikel-Forecast-Timer summieren sich auf {m['cached_particle_filter_plus_forecast_seconds']:.3f} s. Die zusätzlichen MLE-Fit-/Forecast-Timer betragen {m['incremental_fit_seconds']:.3f}/{m['incremental_forecast_seconds']:.3f} s. Die P4096-Zeit ist kein reiner Filtertimer: sie enthält auch eine für die Startauswahl unnötige Partikel-Zukunft. Sie darf weder als kostenlos verschwinden noch beim Wiederverwenden doppelt angerechnet werden. Für den 72-Fall-Zweistartlauf existiert kein separat gemessener P4096-Startaufbau nur dieses Subsets. Die vergleichende Zeitschrittdiagnose enthält ihren neuen P1024-Inferenzaufwand bereits.", ""]
    f = j["local_fits"]
    lines += [f"Ursprüngliche PI-Datengenerierung: {f['dataset_wall_seconds']:.3f} s. Die 24 einzelnen Fit-Timer summieren sich auf {f['sum_individual_fit_wall_seconds']:.3f} s (Maximum {f['max_individual_fit_wall_seconds']:.3f} s, Median {f['median_individual_fit_wall_seconds']:.3f} s); zwei Fitprozesse wurden verwendet. Diese überlappenden Einzelzeiten sind keine Gesamtwand- oder CPU-Zeit. Der frühere Fitbericht nennt 2,03 s Gesamtwandzeit, aber dafür liegt kein separater Prozess-JSON-Timer vor; hier wird diese Zahl nicht als neu rekonstruierte genaue Gesamtmessung ausgegeben. Der vollständige isolierte 24-Fit-Reproduktionslauf ist oben mit seinem eigenen Gesamtprozess-Timer enthalten.",
              "", "## Main2 und Wiederholung", "", f"Status dieses Snapshots: `{j['main2']['status']}`. Die native Erzeugung ist separat erfasst. Main2-Scores wurden nicht gelesen.", "",
              "Nach vollständigen Main2-Bankprozessen in **frische** Dateien schreiben; vorhandene Ausgaben werden abgelehnt:", "", "```bash", "cd /home/loya2/physics-transfer", "/home/loya2/miniconda3/envs/unitree_sim_env/bin/python paper_study/scripts/report_resources.py --include-main2-complete --out-prefix paper_study/reports/resource_accounting_v2", "```", "",
              "## Grenzen und Provenienz", "", *[f"- {x}" for x in j["limitations"]], "", f"Das zugehörige JSON enthält {len(j['source_sha256'])} SHA-256-Eingabebindungen, genaue Zeiten, Status und Messgrenzen. Alle verwendeten Eingabedateien wurden vor dem Schreiben erneut auf unveränderte Hashes geprüft. Keine Quelle oder bestehende Ausgabe wurde verändert.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=PAPER / "reports/resource_accounting")
    parser.add_argument("--include-main2-complete", action="store_true")
    args = parser.parse_args()
    prefix = args.out_prefix.resolve()
    outputs = [Path(str(prefix)+suffix) for suffix in (".json", ".md")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Use a fresh output prefix; resource snapshots are never overwritten")
    result = collect(args.include_main2_complete)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    with outputs[1].open("x") as f:
        f.write(render(result))
    print(json.dumps(dict(outputs=[str(p) for p in outputs], version=result["version"], main2=result["main2"]["status"])))


if __name__ == "__main__":
    main()
