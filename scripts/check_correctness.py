"""
Check a benchmark run against every correctness criterion of judging-criteria.pdf.

    python benchmark.py --cpus 1 --repeats 3 --output run.json --save-arrays run.npz
    python scripts/check_correctness.py run.json run.npz [original.npz]

`original.npz` is the same --save-arrays output from the unmodified forked code (last organiser commit 02ec741).
benchmark.py is not modified; this script only reads its JSON report and saved arrays.

Criteria (judging-criteria.pdf, "Correctness"):
  C1  complete setup, all 101 frequencies, all 193 design variables, double precision; physical results and
      gradients preserved (compared with the original code's arrays when given)
  C2  sensitivities positive and |ln(sensitivity) - ln(reference)| <= 1e-5 at every frequency
  C3  carrier, signal, noise, sensitivity, powers, loss and all gradients finite; gradient not identically zero
  C4  gradients pass the supplied sanity check (benchmark.py's finite-difference check; the run printed PASS)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# "Preserve the physical results and gradients": the binding numeric tolerance is C2 (1e-5 on ln S). The bounds
# below for the other arrays are our own, well above round-off at cond(A) ~ 1e30 (sparse vs dense LU: ~1e-7 on the
# signal solution) and far below anything physically visible.
PRESERVE_TOLERANCE = {"carrier": 1e-9, "signal": 1e-6, "noise": 1e-6, "sensitivity": 1e-6, "powers": 1e-9, "gradient": 1e-4}

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main(report_path, arrays_path, original_path=None):
    report = json.loads(Path(report_path).read_text())
    a = np.load(arrays_path)
    reference = json.loads((ROOT / "examples/data/uifo_800_3000_test_data.json").read_text())
    expected = np.asarray(reference["sensitivities"], dtype=np.float64)
    v = report["validation"]
    print(f"run: {report_path}  (xla_flags: {report['environment'].get('xla_flags')})")
    src = report["source"]
    print(f"code: commit {src.get('commit')}, uncommitted changes to tracked files: {src.get('modified')}")
    check("run is of a committed state (benchmark.py's git record)", src.get("commit") is not None and src.get("modified") is False)

    print("C1 complete setup, 101 frequencies, 193 design variables, double precision")
    check("101 frequencies", a["frequencies"].shape == (101,) and report["inputs"]["frequency_count"] == 101,
          f"{a['frequencies'].shape}")
    check("reference frequencies used", np.array_equal(a["frequencies"], np.asarray(reference["frequencies"], dtype=np.float64)))
    check("193 design variables, full gradient", a["parameters"].shape == (193,) and a["gradient"].shape == (193,)
          and report["inputs"]["gradient_parameter_count"] == 193, f"parameters {a['parameters'].shape}, gradient {a['gradient'].shape}")
    check("unchanged public inputs", report["inputs"]["setup_sha256"] and report["source"].get("simulator_files_sha256") is not None)
    shapes = {k: a[k].shape for k in ("carrier", "signal", "noise", "sensitivity", "powers")}
    check("output shapes", shapes["signal"][-1] == 101 and shapes["noise"] == (101,) and shapes["sensitivity"] == (101,),
          str(shapes))
    dtypes = {k: str(a[k].dtype) for k in ("carrier", "signal", "noise", "sensitivity", "powers", "gradient", "parameters")}
    double = dtypes["carrier"] == dtypes["signal"] == "complex128" and all(
        dtypes[k] == "float64" for k in ("noise", "sensitivity", "powers", "gradient", "parameters"))
    check("double precision (float64 / complex128)", double, str(dtypes))
    if original_path:
        o = np.load(original_path)
        for key, tolerance in PRESERVE_TOLERANCE.items():
            if o[key].shape != a[key].shape:
                check(f"{key} preserved vs original code", False, f"shape {a[key].shape} vs {o[key].shape}")
                continue
            d = rel(a[key], o[key])
            check(f"{key} preserved vs original code", d <= tolerance, f"max rel diff {d:.2e} (bound {tolerance:g})")
        g, go = a["gradient"], o["gradient"]
        cosine = float(g @ go / (np.linalg.norm(g) * np.linalg.norm(go)))
        check("gradient direction vs original code", cosine > 1 - 1e-9, f"cosine {cosine:.12f}")
    else:
        print("  (no original-code arrays given: preservation not compared)")

    print("C2 positive sensitivities matching the public reference at every frequency")
    s = a["sensitivity"]
    check("sensitivities positive", bool(np.all(s > 0)), f"min {s.min():.3e}")
    log_error = np.abs(np.log(s) - np.log(expected))
    check("|ln S - ln ref| <= 1e-5 at every frequency", bool(np.all(log_error <= 1e-5)),
          f"max {log_error.max():.3e} at frequency index {int(log_error.argmax())}, {int(np.sum(log_error > 1e-5))} violations")

    print("C3 finite outputs, non-zero gradient")
    for key in ("carrier", "signal", "noise", "sensitivity", "powers", "gradient"):
        check(f"{key} finite", bool(np.isfinite(a[key]).all()))
    loss = float(np.mean(np.log10(s / expected)))
    check("loss finite and consistent", np.isfinite(v["loss"]) and abs(v["loss"] - loss) <= 1e-12 * max(1.0, abs(loss)),
          f"report {v['loss']:.12e}, recomputed {loss:.12e}")
    check("gradient not identically zero", bool(np.any(a["gradient"] != 0)), f"{int(np.count_nonzero(a['gradient']))} non-zero entries")

    print("C4 supplied gradient sanity check")
    gc = v["gradient_check"]
    check("finite-difference check passed (benchmark.py)", gc["passed"] and v["upstream_sensitivity_check_passed"]
          and v["all_outputs_and_gradients_finite"],
          f"{gc['parameter']}: autodiff {gc['autodiff']:.6e}, central differences "
          + ", ".join(f"{fd['central_difference']:.6e} (h={fd['step']:g})" for fd in gc["finite_differences"]))

    print("RESULT:", "ALL CRITERIA PASS" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    raise SystemExit(main(*sys.argv[1:]))
