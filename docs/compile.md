Which functions make the program big

These are the forward program's operations, grouped by the function in differometor they came from:

┌───────────────────────────────────────────────────────────────┬─────────────┬──────────┐
│                           Function                            │     Ops     │  Share   │
├───────────────────────────────────────────────────────────────┼─────────────┼──────────┤
│ incidence_angle_right (Snell's law: cos(arcsin(n₁/n₂·sin α))) │         889 │    26.8% │
├───────────────────────────────────────────────────────────────┼─────────────┼──────────┤
│ incidence_angle_left (cos α)                                  │         413 │    12.5% │
├───────────────────────────────────────────────────────────────┼─────────────┼──────────┤
│ standardize_output (list → array helper)                      │         203 │     6.1% │
├───────────────────────────────────────────────────────────────┼─────────────┼──────────┤
│ space_modulation                                              │         190 │     5.7% │
├───────────────────────────────────────────────────────────────┼─────────────┼──────────┤
│ simulate (glue code)                                          │         173 │     5.2% │
├───────────────────────────────────────────────────────────────┼─────────────┼──────────┤
│ assembly.component_outputs                                    │         137 │     4.1% │
├───────────────────────────────────────────────────────────────┼─────────────┼──────────┤
│ phase_shift, expand_parameters, surface, carrier solve        │ 80–100 each │ ~3% each │
└───────────────────────────────────────────────────────────────┴─────────────┴──────────┘


help more than any trig rewrite:

1. If alpha and the refractive indices are fixed (not being optimized), compute this value once with plain numpy/math outside jit. It then becomes a constant and costs nothing at trace or compile time.
2. If the call sites sit inside Python loops over components, vectorizing them or using jax.lax.scan shrinks the graph much more.
