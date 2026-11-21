"""
Run the benchmark's pipeline on many different inputs (setups, sweeps, parameter values) and time it.

    python scripts/scenarios.py list
    python scripts/scenarios.py time [--rounds 3] [names...]     # fresh pinned process per run, interleaved
    python scripts/scenarios.py check [names...]                 # host path vs the original dense JAX code path

Every run mirrors benchmark.py (unchanged): one CPU pinned before imports, x64, no compilation cache; a jitted
forward with AD residuals (jax.vjp), a jitted sensitivity read-out with its pullback, and a separately jitted reverse
pass, each lowered and compiled on its own; then a host transfer. The reported total includes `import jax` and
`import differometor`, the build (`run_build_step`), all lowering/compilation and all execution.

`check` applies the judging criteria to every scenario: finite outputs, non-zero gradient, positive sensitivities
within 1e-5 (ln) of a high-accuracy reference ("truth": every solve with 6 steps of mixed-precision refinement,
validated against dense LAPACK solves refined in np.longdouble), gradient within 1e-4 of it; it also compares
carrier, signal, noise, sensitivity (or powers) and the gradient with the original code path in
the same tree (host path, WS2 adjoint and WS3 assembly switched off: dense XLA LU and JAX autodiff, i.e. the
reference numerics), in one process per scenario.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
OPTIMIZED = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length", "phase"]
NO_REFERENCE = {"uifo800-400f": "the original dense code path needs > 48 GB at 400 frequencies"}
CHUNKS = 4   # such references are computed in CHUNKS equal frequency chunks


# ---------------------------------------------------------------------------------------
# Scenarios: name -> (description, builder). A builder returns (setup, changing_pairs, values, pairs,
# initial optimized values or None, reference sensitivity or None).
# ---------------------------------------------------------------------------------------

def _uifo_fixture(name, freqs=None, perturb=None, seed=0, fmin=None, fmax=None, edits=None):
    import numpy as np
    import differometor as df
    from differometor.setups import constrain_inter_grid_cell_spaces

    data = ROOT / "examples" / "data"
    fixture = json.loads((data / f"uifo_{name}.json").read_text())
    for kind, changes in (edits or {}).items():          # {"edges"/"nodes": {component: {property: value}}}
        for component, properties in changes.items():
            fixture[kind][component]["properties"].update(properties)
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    components = fixture["nodes"] | fixture["edges"]
    values = []
    for pair in pairs:
        member, prop = pair[0] if isinstance(pair[0], list) else pair
        values.append(components[member]["properties"][prop])
    values = np.asarray(values, dtype=np.float64)
    reference = None
    if name == "800_3000" and freqs is None and fmin is None:
        test = json.loads((data / "uifo_800_3000_test_data.json").read_text())
        frequencies, reference = np.asarray(test["frequencies"]), np.asarray(test["sensitivities"])
    else:
        lo, hi = (fmin, fmax) if fmin else map(float, name.split("_"))
        frequencies = np.logspace(np.log10(lo), np.log10(hi), freqs or 101)
    if edits:
        reference = None                                     # modified setup: the public reference does not apply
    if perturb:
        rng = np.random.default_rng(seed)
        values = values * (1 + perturb * rng.uniform(-1, 1, values.shape))
        reference = None
    return setup, [("f", "frequency")], frequencies, pairs, values, reference


def _predefined(function, freqs=101, lo=10, hi=5000, mode="space_modulation"):
    import numpy as np

    setup, pairs = function(mode=mode)[:2]
    pairs = [p for p in pairs if p[1] in OPTIMIZED]
    return setup, [("f", "frequency")], np.logspace(np.log10(lo), np.log10(hi), freqs), pairs, None, None


def _aligo_test():
    import numpy as np
    from differometor.setups import aligo

    data = json.loads((ROOT / "examples" / "data" / "aligo_test_data.json").read_text())
    setup, pairs = aligo()[:2]
    return setup, [("f", "frequency")], np.asarray(data["frequencies"]), [p for p in pairs if p[1] in OPTIMIZED], None, None


def _random_uifo(size, seed, mode="space_modulation", freqs=101):
    import numpy as np
    from differometor.setups import uifo

    out = uifo(size=size, random=True, random_seed=seed, mode=mode)
    setup, pairs = out[0], out[1]
    return setup, [("f", "frequency")], np.logspace(np.log10(20), np.log10(5000), freqs), [p for p in pairs if p[1] in OPTIMIZED], None, None


def _cavity():
    import numpy as np
    import differometor as df

    s = df.Setup()
    s.add("laser", "l0", power=1)
    s.add("mirror", "m0", reflectivity=0.99, loss=0)
    s.add("mirror", "m1", reflectivity=0.991, loss=0)
    s.space("l0", "m0", length=1)
    s.space("m0", "m1", length=1)
    s.add("detector", "refl", target="m0", port="left", direction="out")
    s.add("detector", "circ", target="m1", port="left", direction="in")
    s.add("detector", "trns", target="m1", port="right", direction="out")
    pairs = [("m0", "reflectivity"), ("m1", "reflectivity"), ("m1", "tuning")]
    return s, [("m0", "tuning")], np.linspace(-180, 180, 400), pairs, None, None


def _GLASS(n, alpha=None):
    """Edits of the edge-cases bundle: refractive index n on center12_ml12 / center12_mt12, optional center12 alpha."""
    edits = {"edges": {space: {"refractive_index": n} for space in ("center12_ml12", "center12_mt12")}}
    if alpha is not None:
        edits["nodes"] = {"center12": {"alpha": alpha}}
    return edits


def _scenarios():
    from differometor.setups import aligo, voyager

    return {
        "uifo800-benchmark": ("benchmark workload: UIFO 800-3000 Hz, 101 public frequencies, 193 variables", lambda: _uifo_fixture("800_3000")),
        "uifo800-perturbed-1pct": ("same, design variables perturbed by +-1% (seed 1)", lambda: _uifo_fixture("800_3000", perturb=0.01, seed=1)),
        "uifo800-perturbed-10pct": ("same, design variables perturbed by +-10% (seed 2)", lambda: _uifo_fixture("800_3000", perturb=0.10, seed=2)),
        "uifo800-20f": ("UIFO 800-3000, 20 frequencies", lambda: _uifo_fixture("800_3000", freqs=20, fmin=800, fmax=3000)),
        "uifo800-400f": ("UIFO 800-3000, 400 frequencies", lambda: _uifo_fixture("800_3000", freqs=400, fmin=800, fmax=3000)),
        "uifo800-wideband": ("UIFO 800-3000 setup swept over 10 Hz-10 kHz, 101 frequencies", lambda: _uifo_fixture("800_3000", freqs=101, fmin=10, fmax=1e4)),
        "uifo20-5000": ("second trained UIFO (20-5000 Hz fixture), 101 frequencies", lambda: _uifo_fixture("20_5000")),
        "uifo-random3": ("random 3x3 UIFO (seed 42), 101 frequencies", lambda: _random_uifo(3, 42)),
        "uifo-random3-ampl": ("random 3x3 UIFO, amplitude modulation", lambda: _random_uifo(3, 42, "amplitude_modulation")),
        "uifo-random3-freqmod": ("random 3x3 UIFO, frequency modulation", lambda: _random_uifo(3, 42, "frequency_modulation")),
        "voyager": ("Voyager, 101 frequencies 10 Hz-5 kHz", lambda: _predefined(voyager)),
        "voyager-ampl": ("Voyager, amplitude modulation (sqrt-linked amplitudes)", lambda: _predefined(voyager, mode="amplitude_modulation")),
        "voyager-freqmod": ("Voyager, frequency modulation", lambda: _predefined(voyager, mode="frequency_modulation")),
        "aligo-ampl": ("aLIGO, amplitude modulation (sqrt-linked amplitude)", lambda: _predefined(aligo, mode="amplitude_modulation")),
        "aligo": ("aLIGO, public test frequencies", _aligo_test),
        "cavity-tuning": ("Fabry-Perot cavity, carrier sweep over 400 tunings", _cavity),
        # edge cases (edge-cases bundle): glass on the right/bottom spaces of the 45-deg beamsplitter center12
        "edge-tir45": ("UIFO 800-3000, n=1.45 at center12: total internal reflection (arcsin arg +1.025)", lambda: _uifo_fixture("800_3000", edits=_GLASS(1.45))),
        "edge-tir135": ("same, center12 alpha=135 deg", lambda: _uifo_fixture("800_3000", edits=_GLASS(1.45, 135.0))),
        "edge-tir225": ("same, center12 alpha=225 deg: opposite TIR branch (arcsin arg -1.025 - 0j)", lambda: _uifo_fixture("800_3000", edits=_GLASS(1.45, 225.0))),
        "edge-glass-n12": ("UIFO 800-3000, n=1.2 at center12: refraction without TIR", lambda: _uifo_fixture("800_3000", edits=_GLASS(1.2))),
    }


# ---------------------------------------------------------------------------------------
# One run (in this process)
# ---------------------------------------------------------------------------------------

def _readout(jnp, df, outputs, ports, reference):
    carrier, signal, noise = outputs
    if noise.size == 0:                                      # no noise detectors: total detected power
        powers = df.power_detector(carrier)
        return jnp.sum(powers), (powers, powers)
    signals = df.signal_detector(carrier, signal)[ports]
    detected = signals.squeeze() if ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    return jnp.mean(jnp.log10(sensitivity / reference)), (sensitivity, df.power_detector(carrier))


def run_one(name, reference_path=False, chunk=None, accurate=False):
    start = time.perf_counter()
    import numpy as np
    import jax
    import jax.numpy as jnp
    import differometor as df

    t_import = time.perf_counter() - start
    if reference_path:                                       # original numerics: no host path, no WS2/WS3
        from differometor import adjoint, assembly, host
        host.ENABLED = adjoint.ENABLED = assembly.ENABLED = False
    if accurate:                                             # high-accuracy reference: every solve mixed-precision refined
        from differometor import adjoint, host
        host.REFINE.update(signal="extended", weights="extended", adjoint="extended")
        adjoint.EXTENDED_REFINEMENT_STEPS = 6
    jax.config.update("jax_enable_compilation_cache", False)
    jax.devices("cpu")
    stages = {}

    def timed(key, function):
        t = time.perf_counter()
        result = function()
        jax.block_until_ready(result)
        stages[key] = time.perf_counter() - t
        return result

    def build():
        setup, changing, values, pairs, initial, reference = _scenarios()[name][1]()
        if chunk is not None:                                # (k, m): k-th of m equal parts of the sweep
            values = np.array_split(np.asarray(values), chunk[1], axis=-1)[chunk[0]]
            reference = None if reference is None else np.array_split(np.asarray(reference), chunk[1])[chunk[0]]
        arrays, ports, *_ = df.run_build_step(setup, changing, jnp.asarray(values, dtype=jnp.float64), pairs)
        arrays = dict(arrays)
        if initial is None:
            indices = np.asarray(arrays["optimized_value_indices"])
            initial = np.zeros(int(indices.max()) + 1)
            initial[indices] = np.real(np.asarray(arrays["parameters"])[0, np.asarray(arrays["optimized_parameter_indices"])])
        params = jnp.asarray(initial, dtype=jnp.float64)
        arrays["optimized_parameters"] = params
        count = np.asarray(arrays["signal_changing_values"]).shape[1] * np.asarray(arrays["carrier_changing_values"]).shape[1]
        ref = jnp.asarray(reference if reference is not None else np.ones(count), dtype=jnp.float64)
        return arrays, jnp.asarray(ports), params, ref, jnp.asarray(1.0, dtype=jnp.float64)

    arrays, ports, params, ref, one = timed("build", build)

    def forward(p, a):
        return jax.vjp(lambda q: df.simulate(**dict(a, optimized_parameters=q)), p)

    def sensitivity(outputs, detector_ports, reference):
        return jax.vjp(lambda v: _readout(jnp, df, v, detector_ports, reference), outputs, has_aux=True)

    def backward(sim_pullback, read_pullback, cotangent):
        return sim_pullback(read_pullback(cotangent)[0])[0]

    def compile_phase(key, function, *values):
        lowered = timed(key + ".lower", lambda: jax.jit(function).lower(*values))
        return timed(key + ".compile", lowered.compile)

    f = compile_phase("forward", forward, params, arrays)
    outputs, sim_pb = timed("forward.run", lambda: f(params, arrays))
    s = compile_phase("sensitivity", sensitivity, outputs, ports, ref)
    loss, read_pb, (sens, powers) = timed("sensitivity.run", lambda: s(outputs, ports, ref))
    g = compile_phase("gradient", backward, sim_pb, read_pb, one)
    grad = timed("gradient.run", lambda: g(sim_pb, read_pb, one))
    host = timed("host_transfer", lambda: jax.device_get((outputs, sens, powers, loss, grad)))
    total = t_import + sum(stages.values())
    (carrier, signal, noise), sens, powers, loss, grad = host
    finite = all(np.isfinite(np.asarray(x)).all() for x in (carrier, signal, noise, sens, powers, loss, grad))
    return {
        "name": name, "total": total, "imports": t_import, "stages": stages,
        "host_path": "host_plan" in arrays, "frequencies": int(np.asarray(noise).shape[-1]) if np.asarray(noise).size else 0,
        "unknowns": int(np.asarray(signal).shape[0]), "variables": int(np.asarray(params).shape[0]),
        "finite": bool(finite), "gradient_nonzero": bool(np.any(np.asarray(grad) != 0)), "loss": float(loss),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "arrays": {k: np.asarray(v) for k, v in zip(("carrier", "signal", "noise", "readout", "powers", "gradient"),
                                                   (carrier, signal, noise, sens, powers, grad))},
    }


# ---------------------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------------------

def _child(name, mode, out, chunk=None):
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    cpu = sorted(os.sched_getaffinity(0))[:1]
    code = (f"import os,sys; os.sched_setaffinity(0, {cpu}); sys.path.insert(0, {str(ROOT)!r}); "
            f"sys.path.insert(0, {str(ROOT / 'scripts')!r}); import scenarios; scenarios._emit({name!r}, {mode!r}, {out!r}, {chunk!r})")
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if proc.returncode:
        return {"name": name, "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}"}
    import numpy as np
    meta = json.loads(Path(out + ".json").read_text())
    meta["arrays"] = dict(np.load(out + ".npz"))
    return meta


def _emit(name, mode, out, chunk=None):
    import numpy as np

    result = run_one(name, reference_path=(mode == "reference"), chunk=chunk, accurate=(mode == "accurate"))
    np.savez(out + ".npz", **result.pop("arrays"))
    Path(out + ".json").write_text(json.dumps(result))


def _rel(a, b):
    import numpy as np

    if b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def main(argv):
    import tempfile

    command = argv[0] if argv else "list"
    rounds = 3
    if "--rounds" in argv:
        rounds = int(argv[argv.index("--rounds") + 1])
        argv = argv[: argv.index("--rounds")] + argv[argv.index("--rounds") + 2:]
    names = argv[1:] or list(_scenarios())
    tmp = tempfile.mkdtemp(prefix="scenarios-")
    if command == "list":
        for name, (description, _) in _scenarios().items():
            print(f"{name:26s} {description}")
    elif command == "time":
        results = {name: [] for name in names}
        for r in range(rounds):
            for name in names:
                res = _child(name, "time", f"{tmp}/{name}-{r}")
                results[name].append(res)
                if "error" in res:
                    print(f"round {r + 1} {name:26s} ERROR {res['error']}", flush=True)
                    continue
                st = res["stages"]
                compile_s = sum(v for k, v in st.items() if k.endswith((".lower", ".compile")))
                run_s = st["forward.run"] + st["sensitivity.run"] + st["gradient.run"]
                print(f"round {r + 1} {name:26s} total {res['total']:.3f} s | imports {res['imports']:.3f} build {st['build']:.3f} "
                      f"compile {compile_s:.3f} run {run_s:.3f} (fwd {st['forward.run']:.3f} grad {st['gradient.run']:.3f}) | "
                      f"host={res['host_path']} N={res['unknowns']} F={res['frequencies']} vars={res['variables']} "
                      f"finite={res['finite']} grad!=0={res['gradient_nonzero']}", flush=True)
        print("\nsummary (seconds): name, median total, min, max | median build, compile, run")
        medians = []
        for name, runs in results.items():
            ok = [r for r in runs if "error" not in r]
            if not ok:
                print(f"{name:26s} ERROR")
                continue
            tot = [r["total"] for r in ok]
            med = lambda key: statistics.median(  # noqa: E731
                sum(v for k, v in r["stages"].items() if key(k)) for r in ok)
            medians.append(statistics.median(tot))
            print(f"{name:26s} {statistics.median(tot):6.3f} {min(tot):6.3f} {max(tot):6.3f} | "
                  f"{med(lambda k: k == 'build'):.3f} {med(lambda k: k.endswith(('.lower', '.compile'))):.3f} "
                  f"{med(lambda k: k.endswith('.run')):.3f}")
        if len(medians) > 1:
            print(f"across scenarios: mean {statistics.mean(medians):.3f} s, std {statistics.stdev(medians):.3f} s, "
                  f"min {min(medians):.3f}, max {max(medians):.3f}")
    elif command == "check":
        import numpy as np

        failed = []
        for name in names:
            new = _child(name, "time", f"{tmp}/{name}-new")
            if name in NO_REFERENCE:     # original path in equal frequency chunks (memory); frequencies are independent
                parts = [_child(name, "reference", f"{tmp}/{name}-ref{k}", (k, CHUNKS)) for k in range(CHUNKS)]
                errors = [p["error"] for p in parts if "error" in p]
                ref = {"error": errors[0]} if errors else {
                    "total": sum(p["total"] for p in parts),
                    "arrays": {
                        "carrier": parts[0]["arrays"]["carrier"], "powers": parts[0]["arrays"]["powers"],
                        "signal": np.concatenate([p["arrays"]["signal"] for p in parts], axis=-1),
                        "noise": np.concatenate([p["arrays"]["noise"] for p in parts], axis=-1),
                        "readout": np.concatenate([p["arrays"]["readout"] for p in parts], axis=-1),
                        # loss = mean over all frequencies = mean of the equal-size chunk means
                        "gradient": np.mean([p["arrays"]["gradient"] for p in parts], axis=0),
                    }}
            else:
                ref = _child(name, "reference", f"{tmp}/{name}-ref")
            acc = _child(name, "accurate", f"{tmp}/{name}-acc")
            if "error" in new or "error" in ref or "error" in acc:
                print(f"{name:26s} ERROR new={new.get('error')} original={ref.get('error')} accurate={acc.get('error')}", flush=True)
                failed.append(name)
                continue
            a, r, t = new["arrays"], ref["arrays"], acc["arrays"]
            diffs = {k: _rel(a[k], r[k]) for k in a}
            # judging-criteria.pdf, applied to every scenario. "Truth" = high-accuracy reference (all solves with
            # 6 steps of mixed-precision refinement); the original code path is compared as well.
            criteria = {
                "finite": all(np.isfinite(v).all() for v in a.values()) and np.isfinite(new["loss"]),
                "gradient!=0": bool(np.any(a["gradient"] != 0)),
            }
            info = ""
            if a["noise"].size:                                    # sensitivity read-out
                criteria["S>0"] = bool(np.all(a["readout"] > 0))
                dln = float(np.max(np.abs(np.log(a["readout"]) - np.log(t["readout"]))))
                criteria[f"|dlnS| vs truth {dln:.1e}<=1e-5"] = dln <= 1e-5
                dln_orig = float(np.max(np.abs(np.log(r["readout"]) - np.log(t["readout"]))))
                dln_vs_orig = float(np.max(np.abs(np.log(a["readout"]) - np.log(r["readout"]))))
                info = f" | original vs truth {dln_orig:.1e}, ours vs original {dln_vs_orig:.1e}"
            else:                                                  # power read-out (no noise detectors)
                criteria[f"powers vs original {diffs['powers']:.1e}"] = diffs["powers"] <= 1e-9
            g_truth = _rel(a["gradient"], t["gradient"])
            criteria[f"gradient vs truth {g_truth:.1e}<=1e-4"] = g_truth <= 1e-4
            info += f" | gradient: original vs truth {_rel(r['gradient'], t['gradient']):.1e}"
            ok = all(criteria.values())
            if not ok:
                failed.append(name)
            print(f"{name:24s} {'PASS' if ok else 'FAIL'} host={new['host_path']} | "
                  + " ".join(f"{k}:{'ok' if v else 'NO'}" for k, v in criteria.items()) + info
                  + " | ours vs original " + " ".join(f"{k} {v:.1e}" for k, v in diffs.items())
                  + f" | time {new['total']:.2f} s vs original path {ref['total']:.2f} s", flush=True)
        print("RESULT:", "ALL SCENARIOS PASS" if not failed else f"FAILED: {failed}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
