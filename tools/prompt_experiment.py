"""Bounded GPU prompt search with controlled Git, CPU and held-out gates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
from collections.abc import Callable

from src import prompt
from src.json_io import loads
from src.schema import FunctionCall, FunctionDef
from tools.prompt_candidates import (
    Candidate, save_candidate, score_records, select_candidate, write_json,
)

DATA_REF = "f1b487e247e86aee9ea8503e3ce0902fc303e447"
TARGET_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def choose_update(
    local: str, remote: str, ancestor: Callable[[tuple[str, str]], bool],
) -> str:
    """Use the newer comparable ref; never guess between divergent sources."""
    if local == remote or ancestor((remote, local)):
        return local
    if ancestor((local, remote)):
        return remote
    raise ValueError("local/remote feat/implementation diverged; sync stopped")


def load_cached(path: Path, fingerprint: str) -> dict[str, Any] | None:
    """Reuse only an artifact for precisely the requested source/settings."""
    if not path.exists():
        return None
    result: dict[str, Any] = loads(path.read_text(encoding="utf-8"))
    return result if result.get("fingerprint") == fingerprint else None


def estimate_cpu(seconds: float) -> dict[str, float]:
    """Keep actual elapsed time and apply the owner's one-per-run offset."""
    return {"raw_seconds": seconds, "estimated_seconds": float(
        Decimal(str(seconds)) + Decimal(8))}


def git(*args: str) -> str:
    """Read or update the authorized experimental worktree."""
    return subprocess.check_output(["git", *args], text=True).strip()


def synchronize(log: Path) -> bool:
    """Merge upstream only at idle boundaries; fail closed on dirty state."""
    if git("branch", "--show-current") != "exp/local-prompt-improvement":
        raise ValueError("use the exp/local-prompt-improvement worktree")
    if git("status", "--porcelain"):
        raise ValueError("uncommitted changes prevent synchronization")
    with log.open("a", encoding="utf-8") as stream:
        subprocess.run(["git", "fetch", "origin"], check=True,
                       stdout=stream, stderr=stream, timeout=120)
        target = choose_update(
            git("rev-parse", "feat/implementation"),
            git("rev-parse", "origin/feat/implementation"),
            lambda pair: subprocess.run(
                ["git", "merge-base", "--is-ancestor", *pair],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0,
        )
        if subprocess.run(["git", "merge-base", "--is-ancestor", target,
                           "HEAD"]).returncode == 0:
            return False
        merged = subprocess.run(["git", "merge", "--no-edit", target],
                                stdout=stream, stderr=stream)
        if merged.returncode:
            subprocess.run(["git", "merge", "--abort"], check=False,
                           stdout=stream, stderr=stream)
            raise ValueError("merge conflicted; original state restored")
        env = os.environ.copy()
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        env.update(UV_OFFLINE="1")
        subprocess.run(["make", "install", "lint", "test"],
                       env=env, check=True,
                       stdout=stream, stderr=stream, timeout=600)
    return True


def fingerprint(candidate: Path, fixture: Path) -> str:
    """Bind evidence to code, data, prompts and fixed target settings."""
    pieces = [git("rev-parse", "HEAD"), candidate.read_bytes().hex(),
              fixture.read_bytes().hex(), TARGET_REVISION,
              "target=float32;greedy;cap=256;gpu;" + sys.executable]
    return hashlib.sha256(json.dumps(pieces).encode()).hexdigest()


def run_logged(
    command: list[str], log: Path, env: dict[str, str], timeout: int = 1200,
) -> int:
    """Launch a fixed command and terminate its whole group on timeout."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def run_trial(
    candidate: Path, fixture: Path, report: Path, env: dict[str, str],
) -> dict[str, Any]:
    """Run the fixed pipeline; reject incomplete cached reports."""
    stamp = fingerprint(candidate, fixture)
    cached = load_cached(report, stamp)
    if cached is not None and cached.get("finished"):
        return cached
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "tools.prompt_trial", "--candidate",
               str(candidate), "--fixture", str(fixture), "--report",
               str(report), "--fingerprint", stamp]
    code = run_logged(command, report.with_suffix(".log"), env, timeout=900)
    data = loads(fixture.read_text())
    result = load_cached(report, stamp)
    if code or result is None or not result.get("finished"):
        result = {"fingerprint": stamp, "finished": False,
                  "error": f"worker failed with exit {code}",
                  "records": [] if result is None else result["records"]}
        result["score"] = score_records(data["cases"], result["records"])
        write_json(report, result)
    return result


def copy_calibrated(root: Path) -> list[Path]:
    """Read fixed fixture blobs without merging their source branch."""
    paths = []
    for suite in ("original", "transfer", "boolean"):
        path = root / f"calibrated-{suite}.json"
        path.write_bytes(subprocess.check_output([
            "git", "show", f"{DATA_REF}:tests/fixtures/calibrated/{suite}/"
            "measurement_fixture.json",
        ]))
        paths.append(path)
    return paths


def cpu_check(
    root: Path, candidate: Path, fixture: Path, env: dict[str, str],
    gpu_report: dict[str, Any], cache: Path,
) -> dict[str, Any]:
    """Measure the native CLI under the prescribed CPU scope."""
    data = loads(fixture.read_text())
    functions = root / "cpu-functions.json"
    inputs = root / "cpu-inputs.json"
    output = root / "cpu-calls.json"
    write_json(functions, data["functions"])
    write_json(inputs, [{"prompt": c["prompt"]} for c in data["cases"]])
    lab = root / "lab"
    lab.mkdir(exist_ok=True)
    if not (lab / "hf").exists():
        (lab / "hf").symlink_to(cache, target_is_directory=True)
    script = root / "cpu-measure.sh"
    shutil.copyfile("research/evo-x3/measure.sh", script)
    cpu_env = dict(env, CMM_LAB=str(lab), CMM_OMP_THREADS="6",
                   CMM_MODULE="tools.prompt_trial", CUDA_VISIBLE_DEVICES="",
                   HIP_VISIBLE_DEVICES="")
    cpu_env.pop("UV_PROJECT_ENVIRONMENT", None)
    code = run_logged([
        "bash", str(script), str(Path.cwd()), "selected-cpu", "--candidate",
        str(candidate), "--cli", "--functions_definition", str(functions),
        "--input", str(inputs), "--output", str(output),
    ], root / "cpu.log", cpu_env, timeout=600)
    raw = (root / "measurements/selected-cpu.txt").read_text()
    match = re.search(r"^wall_s=([0-9.]+)", raw, re.MULTILINE)
    if match is None:
        raise ValueError("CPU harness did not produce wall_s")
    records = []
    calls = loads(output.read_text()) if output.exists() else []
    for case in data["cases"]:
        matching = [call for call in calls if call["prompt"] == case["prompt"]]
        if len(matching) == 1:
            records.append({"id": case["id"], "call": matching[0]})
    definitions = {f["name"]: FunctionDef.model_validate(f)
                   for f in data["functions"]}
    score = score_records(data["cases"], records, definitions)
    normalized = [FunctionCall.model_validate(r["call"]).model_dump()
                  for r in records]
    gpu_calls = [r["call"] for r in gpu_report["records"] if "call" in r]
    identical = (json.dumps(normalized, sort_keys=True)
                 == json.dumps([FunctionCall.model_validate(c).model_dump()
                                for c in gpu_calls], sort_keys=True))
    result: dict[str, Any] = {
              **estimate_cpu(float(match[1])), "exit_code": code,
              "score": score, "gpu_calls_identical": identical,
              "profile": raw, "code_sha": git("rev-parse", "HEAD")}
    result["passed"] = (
        code == 0 and identical
        and score["correct_ids"] == gpu_report["score"]["correct_ids"]
        and score["valid"] == score["total"]
        and result["estimated_seconds"] < 300
    )
    write_json(root / "cpu-result.json", result)
    return result


def editor_context(
    base: dict[str, Any], fixture: Path, candidate: Path, role: int,
) -> dict[str, Any]:
    """Expose only development failures and a specific generic editing goal."""
    data = loads(fixture.read_text())
    observed = {r["id"]: r for r in base["records"]}
    return {
        "current_prompt": loads(candidate.read_text()),
        "development_functions": data["functions"],
        "failures": [dict(case, observed={k: v for k, v in observed.get(
                         case["id"], {}).items() if k in {"call", "error"}})
                     for case in data["cases"]
                     if case["id"] not in base["score"]["correct_ids"]],
        "goal": (
            "Clarify input argument roles; do not execute functions or return "
            "their results. Keep instructions concise and general."
            if role == 0 else
            "Improve exact extraction of signs, literal replacement symbols, "
            "and meaningful whitespace. Generalize beyond these examples."
        ),
    }


def main() -> int:
    """Run a finite, resumable search; publish no results or code remotely."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--rounds", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    root, cache = args.output.resolve(), args.cache.resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HOME=str(cache), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    try:
        synchronize(root / "sync.log")
        preflight = subprocess.check_output([
            sys.executable, "-c", "import json,torch; "
            "assert torch.cuda.is_available(), 'GPU required'; "
            "print(json.dumps({'torch':torch.__version__,"
            "'gpu':torch.cuda.get_device_name(0),"
            "'free_total_bytes':torch.cuda.mem_get_info(0)}))",
        ], env=env, text=True)
        manifest = {"code_sha": git("rev-parse", "HEAD"),
                    "target_revision": TARGET_REVISION,
                    "optimizer": args.optimizer, "optimizer_dtype": "bfloat16",
                    "target_dtype": "float32", "target_decoding": "greedy",
                    "value_token_limit": 256, "data_ref": DATA_REF,
                    "workers": args.workers, "rounds": args.rounds,
                    "gpu_preflight": loads(preflight)}
        write_json(root / "manifest.json", manifest)
        store = root / "candidates"
        baseline_id = save_candidate(store, Candidate(
            name_instructions=prompt.NAME_INSTRUCTIONS,
            parameter_instructions=prompt.PARAMETER_INSTRUCTIONS))
        baseline_path = store / f"{baseline_id}.json"
        fixture = Path("tests/fixtures/prompt_development.json").resolve()
        epoch = root / manifest["code_sha"]
        epoch.mkdir(exist_ok=True)
        base = run_trial(baseline_path, fixture, epoch / "baseline.json", env)
        if not base.get("finished"):
            raise ValueError("baseline failed; see baseline artifact/log")
        trials: list[tuple[str, dict[str, Any]]] = []
        best_path, best = baseline_path, base
        previous = root / "best.json"
        if previous.exists():
            old_id = loads(previous.read_text())["candidate_id"]
            if not re.fullmatch(r"[a-f0-9]{64}", old_id):
                raise ValueError("invalid previous candidate ID")
            old_path = store / f"{old_id}.json"
            old_trial = run_trial(old_path, fixture,
                                  epoch / "previous.json", env)
            trials.append((old_id, old_trial["score"]))
            if select_candidate(base["score"], trials) is not None:
                best_path, best = old_path, old_trial
        for round_index in range(args.rounds):
            if len(best["score"]["correct_ids"]) == best["score"]["total"]:
                break
            if synchronize(root / "sync.log"):
                # Re-execution reloads modules after merging and invalidates
                # every cached comparison through the new code fingerprint.
                os.execv(sys.executable, [
                    sys.executable, "-m", "tools.prompt_experiment",
                    *sys.argv[1:],
                ])
            tasks = []
            for worker in range(args.workers):
                folder = epoch / f"round-{round_index}-worker-{worker}"
                folder.mkdir(exist_ok=True)
                context = folder / "context.json"
                write_json(context, editor_context(best, fixture, best_path,
                                                   worker))
                tasks.append((folder, context))

            def edit(task: tuple[Path, Path]) -> dict[str, Any]:
                """One bounded editor process; no edits to shared source."""
                folder, context = task
                output = folder / "editor.json"
                stamp = fingerprint(best_path, fixture) + args.optimizer
                old = load_cached(output, stamp)
                if old is not None and old.get("finished"):
                    return old
                code = run_logged([
                    sys.executable, "-m", "tools.prompt_editor", "--context",
                    str(context), "--store", str(store), "--fixture",
                    str(fixture), "--output", str(output), "--optimizer",
                    args.optimizer, "--fingerprint", stamp,
                ], folder / "editor.log", env)
                if not output.exists():
                    return {"error": f"editor failed: {code}",
                            "exit_code": code}
                edited_result: dict[str, Any] = loads(output.read_text())
                return edited_result

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                edits = list(pool.map(edit, tasks))
            for index, edited in enumerate(edits):
                if "candidate_id" not in edited or "trial" not in edited:
                    continue
                candidate_id = edited["candidate_id"]
                trials.append((candidate_id, edited["trial"]["score"]))
                chosen = select_candidate(base["score"], trials)
                if chosen is not None:
                    best_path = store / f"{chosen}.json"
                    if chosen == candidate_id:
                        best = edited["trial"]
                    write_json(root / "best.json", {"candidate_id": chosen})
                print(json.dumps({"round": round_index, "worker": index,
                                  "candidate_id": candidate_id,
                                  "correct": len(edited["trial"]["score"][
                                      "correct_ids"])}), flush=True)
            write_json(epoch / "search.json", {
                "baseline": base["score"], "trials": trials})
        if synchronize(root / "sync.log"):
            os.execv(sys.executable, [
                sys.executable, "-m", "tools.prompt_experiment", *sys.argv[1:],
            ])
        chosen = select_candidate(base["score"], trials)
        summary: dict[str, Any] = {
            "manifest": manifest, "baseline": base["score"],
            "selected_candidate": chosen, "status": "no_improvement"}
        if chosen is not None:
            selected = store / f"{chosen}.json"
            best = run_trial(selected, fixture, epoch / "selected.json", env)
            cpu = cpu_check(epoch, selected, fixture, env, best, cache)
            summary["cpu"] = cpu
            summary["status"] = "cpu_failed"
            if cpu["passed"]:
                # No editor remains alive, and no held-out information has
                # influenced candidate selection. One comparison, no retuning.
                summary["calibrated"] = {}
                for heldout in copy_calibrated(epoch):
                    summary["calibrated"][heldout.stem] = {
                        "baseline": run_trial(
                            baseline_path, heldout,
                            epoch / f"{heldout.stem}-baseline.json",
                            env)["score"],
                        "candidate": run_trial(
                            selected, heldout,
                            epoch / f"{heldout.stem}-candidate.json",
                            env)["score"],
                    }
                summary["status"] = "completed"
        write_json(epoch / "summary.json", summary)
        print(json.dumps(summary, default=str), flush=True)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        write_json(root / "failure.json", {
            "error": str(error), "time_ns": time.time_ns()})
        print(f"prompt experiment stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
