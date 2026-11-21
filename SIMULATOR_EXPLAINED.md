# Understanding Differometor: physics, computation, and bottlenecks

Think of Differometor as **SPICE AC analysis for an optical circuit—with movable mirrors and quantum noise**.

This explanation is based on the simulator source and the supplied CPU benchmark. No new performance measurements were made. Timings are the repository’s published baseline; proposed internal optimizations remain hypotheses until profiled and validated.

## 1. Light as a phasor

In circuit analysis, a sinusoidal voltage can be represented using a complex amplitude:

$$
v(t)=\operatorname{Re}\{V e^{i\omega t}\}.
$$

Do the same for the electric field of laser light:

$$
E(t)=\operatorname{Re}\{a e^{i\omega_0t}\}.
$$

The complex number $a$ contains amplitude and phase. With the simulator’s normalization, optical power is $P=|a|^2$.

**Fields interfere; powers do not simply add.** For two beams arriving at a detector:

$$
P=|a_1+a_2|^2
=|a_1|^2+|a_2|^2+2\operatorname{Re}(a_1a_2^*).
$$

The final term makes an interferometer sensitive to phase. Changing an optical path changes phase, which changes measured power.

For a simple reflected beam, a mirror displacement $x$ changes the round-trip phase by

$$
\Delta\phi=\frac{4\pi x}{\lambda}.
$$

This is the bridge to LIGO: a gravitational wave changes effective optical paths, and interference converts the resulting phase modulation into a measurable signal. The full response also depends on geometry, cavities, and frequency.

## 2. Build an optical circuit

| Electronics idea | Optical equivalent |
|---|---|
| AC source | Laser |
| Transmission-line phase delay | Light propagation through a space |
| Coupler or scattering network | Mirror or beamsplitter |
| Resonator | Optical cavity |
| Operating-point calculation | Carrier-field solution |
| Small-signal AC analysis | Signal-sideband solution |
| Input-referred noise | Strain sensitivity |
| Parameter sensitivity | Gradient with respect to a design variable |

A mirror mixes incoming fields through reflection and transmission. A space multiplies a field by a propagation phase. Connecting these relations produces simultaneous linear equations:

$$
A(\theta)a=b(\theta),
$$

where $\theta$ contains the physical design parameters.

Feedback appears naturally: light can travel around a cavity many times. Solving the equations accounts for repeated reflections without explicitly following every round trip.

For example, a simplified cavity has a circulating field

$$
a_{\mathrm{circ}}=\frac{t_1a_{\mathrm{in}}}{1-r_1r_2e^{i\phi}},
$$

where $r_1,r_2$ are amplitude reflectivities and $\phi$ is round-trip phase. Near resonance, the denominator becomes small and the field builds up, much like a resonant electrical circuit.

## 3. What the simulator calculates

The sequence in [simulate.py](differometor/simulate.py) is:

```text
Optical layout + physical parameters
                  ↓
         Steady carrier fields
                  ↓
     Small-signal response at each frequency
                  ↓
     Quantum noise at the detector
                  ↓
       Sensitivity and scalar objective
                  ↓
      Derivatives of that objective
```

1. **Carrier:** Solve the steady optical operating point: field amplitudes and phases throughout the network, and therefore circulating powers.
2. **Signal:** Apply a small sinusoidal disturbance at frequency $f$. A modulated optical carrier develops sidebands at $f_0+f$ and $f_0-f$. The simulator solves their propagation together with mechanical coupling.
3. **Quantum noise:** Propagate vacuum fluctuations, including contributions associated with optical loss and squeezed inputs, to the readout.
4. **Sensitivity:** Divide output noise by signal response.
5. **Gradients:** Differentiate the final objective with respect to the design variables.

This small-signal approach is described in the [Finesse documentation](https://finesse.ifosim.org/docs/develop/usage/plane-waves/input_fields.html).

**The 800–3000 Hz sweep is the disturbance frequency, not the optical frequency.** It asks how the detector responds to oscillations in that band.

There is no time-step loop resolving every oscillation of the laser and no 3D electromagnetic mesh here. The model uses plane-wave optical fields and small disturbances about an operating point. Time-step size and mesh resolution are therefore not the main performance controls for this simulator.

## 4. Why mirror masses matter

Light carries momentum. Fluctuating optical power produces fluctuating radiation pressure, which moves mirrors. Motion changes optical phase, which changes fields and radiation pressure again:

$$
\text{light}\rightarrow\text{force}\rightarrow\text{mirror motion}
\rightarrow\text{optical phase}\rightarrow\text{light}.
$$

This is **optomechanical feedback**.

The mechanical model includes free-mass susceptibility:

$$
x(f)=\chi(f)F(f),\qquad
\chi(f)=-\frac{1}{m(2\pi f)^2}.
$$

Heavier mirrors move less under the same force, and free-mass response falls with frequency squared. See `susceptibility()` in [components.py](differometor/components.py); the matrix implementation applies an additional sign convention.

Increasing laser power is not an unlimited improvement: it can improve measurement precision while increasing radiation-pressure back-action.

## 5. Sensitivity and the optimization objective

Let the readout response to strain be $H(f)$, and let the output noise amplitude spectral density be $n_{\mathrm{out}}(f)$. The equivalent input strain noise is

$$
h_n(f)=\frac{n_{\mathrm{out}}(f)}{|H(f)|}.
$$

Its usual units are $1/\sqrt{\mathrm{Hz}}$. Lower means better: less strain is needed to compete with noise in a given measurement bandwidth.

This is analogous to dividing an amplifier’s output voltage noise by its gain to obtain input-referred noise.

The [starter script](start_here.py) evaluates

$$
J(\theta)=\frac{1}{101}\sum_f
\log_{10}\left(\frac{h_n(f;\theta)}{h_{\mathrm{reference}}(f)}\right).
$$

A negative value means improvement in the geometric-average sensitivity ratio over the sampled frequencies. It does not guarantee improvement at every frequency.

The 193 gradient values are

$$
\frac{\partial J}{\partial\theta_1},\ldots,
\frac{\partial J}{\partial\theta_{193}}.
$$

They tell an optimizer how small design changes affect the objective. JAX uses automatic differentiation through the calculation; it does not need 193 independent finite-difference simulations.

Raw gradient magnitudes are not directly comparable across metres, kilograms, degrees, and dimensionless reflectivities. Parameter scaling matters when interpreting them.

## 6. Measured bottlenecks

The supplied [CPU baseline report](benchmarks/cpu-baseline.json) measures one logical CPU:

| First evaluation phase | Time | Approximate share |
|---|---:|---:|
| Tracing and compilation | 22.918 s | 52% |
| Forward simulation, including saved differentiation intermediates | 13.288 s | 30% |
| Reverse-mode gradient | 4.336 s | 10% |
| Imports, initialization, input preparation | 3.661 s | 8% |
| Readout calculation and result transfer | About 0.0014 s | Negligible |

Total: **44.204 seconds**. Once compiled, the reported median evaluation takes **17.679 seconds**.

- For the scored first evaluation, **compilation is the largest measured cost**.
- For repeated evaluations, **the forward simulation dominates**.

The final noise-to-signal division is not where the time goes.

Within the forward simulation, strong suspects are large dense linear solves, matrix construction, and intermediate arrays retained for differentiation. The report does not separate these internals, so their exact ranking requires profiling.

The saved output shapes reveal the scale:

| System | Unknowns | Frequency points |
|---|---:|---:|
| Carrier | 286 | 1 |
| Signal, including optical sidebands and additional coupled variables | 751 | 101 |

Dense factorization typically costs $O(N^3)$; storing a matrix costs $O(N^2)$. One batch of 101 complex128 matrices of size $751\times751$ occupies approximately **0.85 GiB**. Multiple matrices, factorizations, temporary arrays, and differentiation intermediates explain why memory matters. The README reports roughly **14.4 GiB peak memory**.

The “3×3 setup” refers to the optical layout, not a $3\times3$ numerical matrix.

## 7. A mathematical opportunity in the noise calculation

The code effectively calculates, for each readout and frequency,

$$
w=A^{-H}c,\qquad
z=A^{-1}Qw,\qquad
S=\operatorname{Re}(c^Hz),
$$

where:

- $A$ is the signal-system matrix;
- $Q$ is the noise-source covariance matrix;
- $c$ describes the detector readout;
- $H$ as a superscript means conjugate transpose, and $A^{-H}=(A^H)^{-1}$.

These operations appear in the quantum-noise section of [simulate.py](differometor/simulate.py), around lines 375–410. The inverse notation describes the algebra; the implementation uses linear solves rather than explicitly constructing inverses.

Because $A^Hw=c$,

$$
c^HA^{-1}=w^H.
$$

Therefore,

$$
\boxed{S=\operatorname{Re}(w^HQw)}.
$$

**The final scalar noise can mathematically be obtained without the second noise-propagation solve.** The subsequent spectral-density normalization and square root remain necessary.

This is a concrete optimization hypothesis grounded in the source. It preserves the mathematical quantity, although floating-point results and gradients still require validation. It has not been implemented or timed here.

Other useful investigations:

| Candidate | Why it could help |
|---|---|
| Reuse matrix factorizations | Signal and noise calculations use the same matrix or its conjugate transpose |
| Group updates by component type | Can simplify the compiled graph and avoid unnecessary branch work |
| Exploit sparse or block structure | Optical connectivity is local, but the current solver uses dense matrices |
| Process frequencies in chunks | Can reduce temporary memory, potentially trading throughput for memory |
| Reduce unnecessary differentiation intermediates | May reduce memory pressure and compilation complexity |

The component dispatcher uses `vmap(lax.switch(...))` in `update()`. Batched conditional execution can become selection operations, so it should not be assumed to execute only the useful component formula cheaply. This is a plausible compilation hotspot, consistent with [JAX’s documented batching behavior](https://docs.jax.dev/en/latest/_autosummary/jax.lax.switch.html), but the compiled computation needs inspection.

## 8. Parameters that affect the physics

| Physical parameter | What it changes |
|---|---|
| Mirror reflectivity | Cavity buildup, coupling, resonance linewidth |
| Mirror tuning | Operating phase and detuning from resonance |
| Optical path length | Propagation delay, resonance spacing, frequency response |
| Laser power | Carrier amplitude, circulating power, measurement noise and back-action |
| Mirror mass | Mechanical response and optomechanical coupling |
| Squeezing strength, `db` | Noise distribution between optical quadratures |
| Squeezing angle, `angle` | Which combination of amplitude and phase fluctuations is reduced |
| Optical loss | Stored power and vacuum-noise coupling; limits squeezing benefits |
| Readout phase | Which optical quadrature is measured |
| Disturbance frequency | Optical and mechanical transfer response |

The benchmark varies the first seven property categories. Other properties still influence the model while remaining fixed.

Changing a mass or tuning usually leaves matrix dimensions unchanged, so it does not automatically change the operation count. Parameters can, however, push the system near sharp resonances or poorly conditioned regimes, making numerical accuracy more delicate.

## 9. Parameters that affect computational cost

| Computational control | Effect |
|---|---|
| Number of unknowns | Dense factorization grows approximately cubically |
| Number of frequencies | More frequency points require more systems to solve |
| Number of readouts or noise projections | Potentially more right-hand sides |
| Matrix structure | Dense versus sparse representation, block structure, reusable factors |
| Precision | This task requires float64 and complex128 |
| Differentiation strategy | Retained intermediates, extra solves, gradient graph size |
| Compilation complexity | Branching, array updates, shapes, specialization |
| Memory and hardware | Batching has both execution and memory costs |

For this challenge, the setup, all 101 frequencies, all 193 derivatives, and double precision must remain. Useful speedups come from computing the same physical model more efficiently—especially simplifying the noise algebra and compiled computation.

## Suggested reading order in the repository

1. [start_here.py](start_here.py): the full workflow in a short script.
2. [examples/cavity.py](examples/cavity.py): a simpler optical resonator.
3. [differometor/components.py](differometor/components.py): physical component equations.
4. [differometor/simulate.py](differometor/simulate.py): carrier, signal, and noise solves.
5. [benchmark.py](benchmark.py): timing and correctness checks.

---

This file uses `$...$` for inline mathematics and `$$...$$` for display equations. Open it in a Markdown viewer that supports LaTeX mathematics, such as GitHub’s rendered Markdown view or a math-enabled editor preview. A plain text editor will show the equation source.
