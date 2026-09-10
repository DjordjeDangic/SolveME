# Dense-q electron-phonon interpolation

The dense-q extension reconstructs the complete coarse electron-phonon grid from irreducible q points, Fourier transforms the Cartesian displacement-space matrices to a real-space representation, evaluates them on a denser regular q mesh, and only then projects them onto phonon eigenmodes.

## Public solver

Use the opt-in solver from `dense_mesolver.py`:

```python
from dense_mesolver import mesolver

me = mesolver()
me.load(
    dyn_filename="sscha_dense.dyn",
    nqirr=16,
    elphdyn_filename="H3S.dyn",
    wee_filename="Wee.nc",
)

me.calculate_a2f(
    interpolation_mesh=(16, 16, 16),
    interpolation_block_size=64,
    nom=2000,
    a2f_smearing=0.5,
)
```

Calling `calculate_a2f()` without `interpolation_mesh` delegates to the historical SolveME implementation.

## Current scope

The interpolated path currently supports harmonic isotropic calculations with `mode_mixing="no"`. It intentionally rejects anharmonic lineshapes, adaptive a2F smearing, multiband interpolation, and simultaneous `on_full_grid=True` until the harmonic implementation has been validated numerically.

The target mesh is a complete regular grid. `interpolation_shift=(0,0,0)` gives a Gamma-centered mesh. `interpolation_block_size` controls peak memory use; the full fine-grid electron-phonon tensor is never required in memory.

## Validation

To compare the reconstructed full coarse grid with the historical irreducible-grid alpha2F calculation:

```python
report = me.validate_coarse_grid_equivalence(
    nom=1000,
    a2f_smearing=0.5,
)
print(report["reference_lambda"])
print(report["dense_lambda"])
print(report["lambda_relative_error"])
print(report["a2f_close"])
```

This is the main acceptance test before using denser meshes for production superconductivity calculations.

## q-mesh convergence

```python
results = me.converge_q_meshes(
    [(4,4,4), (6,6,6), (8,8,8), (12,12,12)],
    calculate_kwargs={"nom": 2000, "a2f_smearing": 0.5},
)

for row in results:
    print(row["mesh"], row["lambda"], row["omega_log_meV"])
```

To include the Migdal-Eliashberg solution at each mesh, pass `solve_kwargs`. The returned records then also contain `tc` and a low-temperature gap estimate.

```python
results = me.converge_q_meshes(
    [(4,4,4), (6,6,6), (8,8,8)],
    calculate_kwargs={"nom": 2000, "a2f_smearing": 0.5},
    solve_kwargs={
        "T_start": 20.0,
        "T_end": 250.0,
        "ntemp": 101,
        "mu_star_approx": True,
        "mu_star": 0.15,
    },
)
```

Convergence should be assessed using the shape of alpha2F together with lambda, omega_log, Tc, and the superconducting gap, rather than using a single scalar alone.

## Physical convention check

The Fourier gauge convention is documented in `docs/elph_matrix_conventions.md`. Before production use, verify its atomic-position phase against the exact Quantum ESPRESSO/EPW deformation-potential convention used to generate the input files. The phase handling is isolated in `to_periodic_gauge()` and `from_periodic_gauge()` so the sign convention can be corrected without changing the interpolation machinery.
