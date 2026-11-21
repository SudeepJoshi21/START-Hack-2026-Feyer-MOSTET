import os as _os

# [host-native-xla-opt0] XLA CPU backend optimisation level 0: the remaining XLA programs (benchmark readout, two
# callbacks) compile ~2x faster (official benchmark 2.01 -> 1.63 s). The physics runs in NumPy/SciPy, so run time is
# unaffected. Must be set before JAX creates its CPU backend; appended to any user flags; recorded by benchmark.py.
# NEEDS ORGANISER APPROVAL before being part of a submission. Opt out: DIFFEROMETOR_XLA_OPT0=0.
if _os.environ.get("DIFFEROMETOR_XLA_OPT0", "1") != "0" and "xla_backend_optimization_level" not in _os.environ.get("XLA_FLAGS", ""):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "") + " --xla_backend_optimization_level=0").strip()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector


def __getattr__(name):
    # `visualize_setup` pulls in plotly; import it only when it is first used.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
