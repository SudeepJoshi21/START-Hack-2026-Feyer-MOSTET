"""
Check whether Differometor's XLA compile options are on or off, and that XLA really applies them.

    python scripts/check_xla_flags.py            # with the default (flags on)
    DIFFEROMETOR_XLA_OPT0=0 python scripts/check_xla_flags.py   # flags off

`differometor/__init__.py` appends the options to the XLA_FLAGS environment variable on import (opt-out:
DIFFEROMETOR_XLA_OPT0=0). benchmark.py is not modified; it imports jax *before* differometor and records
os.environ["XLA_FLAGS"] in its JSON report ("environment" -> "xla_flags").

Every check runs in a fresh subprocess with benchmark.py's import order (jax, jax.numpy, numpy, differometor) and
its environment (JAX_PLATFORMS=cpu, JAX_ENABLE_X64, one pinned CPU):
  1. XLA_FLAGS after `import differometor`, exactly what benchmark.py records.
  2. Proof that XLA reads XLA_FLAGS only when the CPU backend starts, i.e. *after* differometor was imported: an
     unknown flag appended at that point must abort the process ("Unknown flag in XLA_FLAGS"). If it did not, flags
     set at import time would be ignored.
  3. Effect: XLA compile time of a small program shaped like the benchmark's read-out, and its output bits,
     with the options on and off.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = textwrap.dedent(
    f"""
    import os, sys
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    sys.path.insert(0, {str(ROOT)!r})
    import jax, jax.numpy as jnp, numpy as np   # benchmark.py order: jax first ...
    import differometor                         # ... then differometor
    """
)


def run(code: str, flag: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, DIFFEROMETOR_XLA_OPT0=flag)
    env.pop("XLA_FLAGS", None)
    return subprocess.run([sys.executable, "-c", PRELUDE + textwrap.dedent(code)], env=env, capture_output=True, text=True)


SHOW = 'print(os.environ.get("XLA_FLAGS"))'

PROBE = """
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_this_flag_does_not_exist=1").strip()
jax.devices("cpu")
print("backend started without reading XLA_FLAGS")
"""

EFFECT = """
import time
x = jnp.asarray(np.linspace(1.0, 2.0, 101) * np.exp(1j * np.linspace(0, 3, 101)))
noise = jnp.asarray(np.linspace(0.5, 1.5, 101))
def readout(signal, noise):
    sensitivity = noise / jnp.abs(signal)
    return jnp.mean(jnp.log10(sensitivity)), sensitivity
lowered = jax.jit(jax.value_and_grad(lambda s, n: readout(s, n)[0], argnums=1)).lower(x, noise)
start = time.perf_counter(); compiled = lowered.compile(); seconds = time.perf_counter() - start
value, grad = compiled(x, noise)
print(f"{seconds:.4f} {float(value).hex()} {np.asarray(grad).tobytes().hex()[:32]}")
"""


def main():
    print(f"DIFFEROMETOR_XLA_OPT0 in this shell: {os.environ.get('DIFFEROMETOR_XLA_OPT0', '(unset -> 1, flags on)')}")
    for flag in ("1", "0"):
        label = "flags ON " if flag == "1" else "flags OFF"
        print(f"\n[{label}] (DIFFEROMETOR_XLA_OPT0={flag})")
        print(f"  1. XLA_FLAGS after import differometor: {run(SHOW, flag).stdout.strip()}")
        probe = run(PROBE, flag)
        applied = probe.returncode != 0 and "Unknown flag in XLA_FLAGS" in probe.stderr
        print(f"  2. XLA parses XLA_FLAGS after differometor's import: {'YES' if applied else 'NO  <-- flags would be ignored'}")
        times, results = [], set()
        for _ in range(3):
            out = run(EFFECT, flag)
            if out.returncode:
                print(out.stderr[-2000:])
                raise SystemExit(1)
            seconds, *bits = out.stdout.split()
            times.append(float(seconds))
            results.add(tuple(bits))
        print(f"  3. compile of a read-out-like program: {sorted(times)[1] * 1e3:.1f} ms (median of 3 fresh processes); "
              f"output bits: {results.pop()}")


if __name__ == "__main__":
    main()
