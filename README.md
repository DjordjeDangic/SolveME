# SolveME
Simple solver for Migdal-Eliashberg equations with Coulombic interaction explicitly included

## Dense-q electron-phonon interpolation

The `feature/elph-symmetry-grid` implementation adds symmetry reconstruction and Fourier interpolation of Cartesian electron-phonon matrices to denser regular q meshes. The opt-in public solver is `dense_mesolver.mesolver`; calling it without `interpolation_mesh` preserves the historical calculation path.

See `docs/dense_q_interpolation.md` for usage, validation, and q-mesh convergence examples, and `docs/elph_matrix_conventions.md` for the symmetry/Fourier phase conventions.
