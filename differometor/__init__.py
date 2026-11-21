import os as _os

# [host-native-xla-opt0] XLA CPU compile options. Backend optimisation level 0: the remaining XLA programs (benchmark readout, two
# callbacks) compile ~2x faster (official benchmark 2.01 -> 1.63 s). The physics runs in NumPy/SciPy, so run time is
# unaffected. Must be set before JAX creates its CPU backend; appended to any user flags; recorded by benchmark.py.
# NEEDS ORGANISER APPROVAL before being part of a submission. Opt out: DIFFEROMETOR_XLA_OPT0=0.
#
# [the-prime-model-new] two more compile-speed options for the same programs (outputs bitwise identical, checked with
# benchmark.py --save-arrays): the legacy LLVM-IR fusion emitters instead of the MLIR-based ones, and no splitting of
# the LLVM module for parallel code generation (pure overhead on one CPU). XLA compile 0.35 -> 0.15 s.
_XLA_COMPILE_FLAGS = {
    "xla_backend_optimization_level": "0",
    "xla_cpu_use_fusion_emitters": "false",
    "xla_cpu_parallel_codegen_split_count": "1",
}
if _os.environ.get("DIFFEROMETOR_XLA_OPT0", "1") != "0":
    _flags = _os.environ.get("XLA_FLAGS", "")
    for _name, _value in _XLA_COMPILE_FLAGS.items():
        if _name not in _flags:  # user-provided values win
            _flags += f" --{_name}={_value}"
    _os.environ["XLA_FLAGS"] = _flags.strip()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector

if _os.environ.get("DIFFEROMETOR_HOST_DETECTORS", "1") != "0":
    # [the-prime-model-new] detector read-outs as host callbacks with hand-written VJPs (see host.py)
    from differometor.host import power_detector, signal_detector  # noqa: F811


def __getattr__(name):
    # `visualize_setup` pulls in plotly; import it only when it is first used.
    if name == "visualize_setup":
        from differometor.plot import visualize_setup

        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
