# Electron-phonon matrix and q-grid conventions

This document fixes the conventions used by the symmetry-reconstruction layer added before Fourier interpolation.

## Reciprocal-space coordinates

All q-points handled by `elph_grid.py` and `elph_symmetry.py` are fractional reciprocal-lattice coordinates. Equivalent points differing by an integer reciprocal-lattice vector are identified, and canonical storage is in `[0, 1)`.

A regular mesh is described by `(N1, N2, N3)` plus an optional shift measured in units of one mesh spacing. A point therefore has coordinates

`q = ((i+s1)/N1, (j+s2)/N2, (k+s3)/N3)`.

The integer mesh coordinate `(i,j,k)` is the preferred key for regular-grid lookup, avoiding repeated floating-point searches.

## Electron-phonon matrix axes

The first axis of an electron-phonon array is q. The last two axes are the square Cartesian atomic-displacement matrix indices used by the existing SolveME deformation-potential workflow. The displacement basis is atom-major with the Cartesian components of each atom grouped together. Intermediate axes may contain smearing or electronic-band indices and are transformed independently by the same displacement-space symmetry matrices.

No Fourier-space atomic-position gauge is defined in this milestone. Gauge/phase removal and restoration must be established explicitly before implementing q-to-R Fourier interpolation.

## Symmetry convention

The symmetry reconstruction follows the convention already used in `mesolver.get_elph_on_full_grid`. For a space-group operation `S`, the reciprocal point is generated using the transpose of the reciprocal-space rotation stored by CellConstructor. The Cartesian displacement representation is `Gamma(S,q)` from `CellConstructor.ThermalConductivity.construct_symmetry_matrix`.

The electron-phonon matrix transforms as

`D(Sq) = Gamma(S,q) D(q) Gamma(S,q)^dagger`.

If the target point is reached through time reversal, the transformed matrix is complex conjugated. Reciprocal-lattice shifts used to bring a transformed point onto the stored q-grid are retained in `QPointMapping`; this is needed later when the Fourier gauge is implemented.

## Validation expectations

The symmetry layer must satisfy identity mapping, q-star coverage, time reversal, deterministic selection when several operations reach the same q-point, and equality of matrices obtained through different valid symmetry paths. The last condition is checked by `validate_symmetry_collisions` and is intended primarily for tests and debugging.
