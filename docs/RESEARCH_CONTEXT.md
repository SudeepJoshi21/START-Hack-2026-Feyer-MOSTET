# Research brief: faster differentiable frequency-domain interferometer simulation

**Your task:** find recent papers (2023 onward; newest first) and practical techniques that could make the pipeline
below faster. Give priority to methods that can be implemented with **NumPy 2.5 / SciPy 1.18 / JAX 0.9 on one CPU
core**, with no new dependencies, and that keep results **identical up to float64 round-off**. For each finding,
report:
- the citation and link;
- what it does;
- which stage below it targets;
- an expected gain, and whether that gain is measured in the paper or your estimate;
- whether it is implementable under the constraints;
- risks to accuracy (the systems are extremely ill-conditioned).

---

## 1. Project in one paragraph

This is the START Hack 2026 "Faster Physics" challenge on **Differometor**, a JAX simulator of
gravitational-wave interferometers (laser-interferometer optics in the frequency domain, similar to Finesse). The
score is the **complete first-run wall time** of `benchmark.py --cpus 1`. That time covers Python imports, building
the optical setup, all JAX tracing/lowering/XLA compilation, a forward simulation at **101 frequencies**, a
sensitivity read-out, and a **reverse-mode gradient over 193 parameters**, all on one pinned CPU core, in float64.
Results must match the public reference within |Δ ln sensitivity| ≤ 1e-5. The original code needed **62 s**
(AMD EPYC 7763); our current branch needs **~1.16 s** (median, shared node).

## 2. The mathematics (what is actually computed)

For each frequency f (101 of them):

1. **Carrier:** solve one small dense complex system (286 × 286) once, since it is frequency-independent here.
2. **Signal system:** A(f) x(f) = b(f).
   - A is complex, **751 × 751, 2,597 non-zeros** (0.46 %), with the **same sparsity pattern at every frequency**.
   - About 1,006 entries vary with f; the rest are constant.
   - A is extremely badly scaled: **cond(A) ≈ 1e30**, though not singular. Dense LU and sparse LU agree to about 4e-8.
   - The entries come from ~680 "component" functions (mirrors, spaces/propagation phases e^{-i2πfL/c},
     optomechanical susceptibilities, beamsplitters) evaluated on a parameter vector.
3. **Quantum noise:** noise² ∝ Re(wᴴ Q w) with Aᴴ w = c.
   - c is a detector read-out vector.
   - Q is a sparse, frequency-independent noise matrix (272 non-zeros).
   - The same factorisation of A is reused for the Aᴴ solve.
4. **Read-out:** sensitivity = noise / |signal at detector|, and loss = mean log10(sensitivity / reference).
5. **Gradient:** a hand-written adjoint. It reuses the 101 factorisations for the Aᵀλ and Aλ solves, takes Ā on
   the non-zeros only, and propagates back through the component functions with forward-mode dual numbers.

## 3. Current implementation ("host-native", branch `the-prime-model-new`)

- The whole simulation runs in NumPy/SciPy inside **one `jax.custom_vjp`** with two `jax.pure_callback`s (forward,
  backward), so XLA only compiles tiny programs.
- **Sparse LU:** SciPy's SuperLU (`_superlu.gstrf`).
  - The COLAMD column order is computed once and reused through a symmetric permutation with `ColPerm=NATURAL`.
  - `PanelSize=4`, `Relax=4`.
  - Partial pivoting, threshold 1.0.
  - One step of **iterative refinement** on the noise-weight and adjoint solves; it is required for accuracy on
    another setup, Voyager.
- **Component derivatives:** dual numbers (forward mode), seeded only on the inputs each function reads.
- **XLA compile options** (bitwise-neutral): `--xla_backend_optimization_level=0`,
  `--xla_cpu_use_fusion_emitters=false`, `--xla_cpu_parallel_codegen_split_count=1`. These may need organiser
  approval.
- **Ruled out:** Numba, Cython and other compiled extensions. The jury installs only a pinned `requirements.txt`
  (jax/jaxlib 0.9.0, numpy 2.5.5, scipy 1.18.1, matplotlib, optax, …). Also ruled out: the persistent compilation
  cache, and any change to `benchmark.py` or the fixtures.

## 4. Where the remaining ~1.1 s goes (one core, EPYC 7763)

| Stage | Time | Ours to change? |
|---|---:|---|
| `import jax` (+ numpy) | ~0.44 s | No; the benchmark imports JAX before our package |
| Build optical setup + prepare arrays | ~0.12 s | Partly (~0.02 s of it is the benchmark's own `jnp.asarray` compile) |
| JAX trace/lower of the 3 programs | ~0.07 s | Little |
| XLA compile of the 3 programs | ~0.15 s | Little; mostly the benchmark's own read-out code |
| Forward: 101 sparse LU factorisations | ~0.07 s | **Yes** |
| Forward: solves + refinement + assembly | ~0.06 s | **Yes** |
| Backward: adjoint solves + refinement | ~0.04 s | **Yes** |
| Backward: component Jacobians (dual numbers) | ~0.035 s | **Yes** |

## 5. Research questions, in order of value

1. **Many sparse LU factorisations with one fixed pattern** (101 complex matrices, n = 751, nnz ≈ 2.6k, fill ≈ 9.8k).
   - Look for: refactorisation with a reused symbolic analysis and pivot sequence (KLU-style refactor, SuperLU
     `SamePattern_SameRowPerm`, GLU/NICSLU from circuit simulation); batched or vectorised sparse LU across matrices;
     static-pivoting LU with refinement (as in SuperLU_DIST GESP) and its stability at cond ~1e30.
   - What helps: anything that can be written in vectorised NumPy over the frequency axis, such as level-scheduled
     elimination run once for all 101 matrices.
2. **Solving a parametric family A(f) = A₀ + A₁(f) quickly over many f.**
   - Background: only ~40 % of the entries vary with f, and the frequency enters mostly through phase factors
     e^{-i2πfL/c}.
   - Look for: low-rank update / Sherman–Morrison–Woodbury approaches, parametric model order reduction (rational
     Krylov, interpolatory MOR, the Loewner framework), recycling Krylov methods (GCRO-DR) with one LU as
     preconditioner, and multi-shift solvers. The key question is accuracy at cond 1e30 with a 1e-5 tolerance on
     log-sensitivity.
3. **Adjoint and sensitivity analysis for frequency-domain linear optics and photonics.** Look for: inverse design
   and adjoint methods in photonics (FDFD adjoint, Ceviche-style), frequency-domain interferometer simulators
   (Finesse 3, Optickle, recent differentiable GW-detector design papers, "Differometor"), and AI-driven
   interferometer discovery (Krenn et al.).
4. **Ill-conditioned complex systems:** scaling/equilibration (MC64-type matchings, Ruiz scaling) to raise accuracy
   so that refinement steps can be dropped, and mixed-precision iterative refinement theory (Carson & Higham)
   applied to complex float64.
5. **Fast forward-mode or Taylor-mode differentiation in pure NumPy** of ~680 small elementwise complex functions
   evaluated at 101 points. Alternatives to dual numbers: complex-step-like tricks for complex outputs, symbolic
   pre-differentiation, and how JAX/Enzyme handle it.
6. **JAX/XLA start-up cost:** recent work on reducing `import jax` time, tracing/lowering overhead, and XLA CPU
   compile latency for small programs. This includes XLA CPU's thunk runtime, the fusion emitters, and the
   LLVM ORC JIT settings exposed as `XLA_FLAGS`.

## 6. Already tried (no need to re-propose)

- **Measured and rejected:**
  - block triangular form (BTF): 612-core block plus 139 1×1 blocks, no gain;
  - other fill-reducing orderings (MMD, RCM);
  - turning off SuperLU equilibration: slower;
  - pivot threshold 0.1: −9 ms, rejected for robustness;
  - fully dense LU in XLA: 36 ms per frequency;
  - a static-structure JAX assembly: no gain;
  - other XLA flags (`xla_llvm_disable_expensive_passes`, legacy runtime, xnnpack/verification/scheduler options):
    within noise.
- **Adopted:**
  - noise via Re(wᴴQw), reusing the LU factors for the Aᴴ solve;
  - host-side SuperLU behind `pure_callback`;
  - hand-written adjoint;
  - grouped component evaluation;
  - the reused column ordering;
  - batched refinement residuals;
  - direction-leading dual-number layout: Jacobians 64 → 35 ms, bitwise identical.

## 7. Useful anchors

- Repository (fork of the public Differometor): https://github.com/SudeepJoshi21/START-Hack-2026-Feyer-MOSTET.
  Upstream: https://github.com/artificial-scientist-lab/Differometor.
- Key files: `differometor/host.py` (host forward/backward, dual numbers), `differometor/adjoint.py` (SuperLU
  wrapper, refinement), `differometor/simulate.py`, `differometor/components.py`, `benchmark.py`.
- Results log: `docs/FINDINGS.md`. Workload: setup `examples/data/uifo_800_3000.json`, 101 frequencies.

**Output format wanted:** a ranked table (rank, paper/technique, stage, expected gain, implementable yes/no/how,
accuracy risk), followed by one short paragraph per top-5 item explaining how it would map onto `host.py` /
`adjoint.py`.
