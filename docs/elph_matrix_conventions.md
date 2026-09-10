# Electron-phonon matrix and q-grid conventions

This document fixes the conventions used by the symmetry-reconstruction and Fourier-interpolation layers.

## Reciprocal-space coordinates

All q-points handled by `elph_grid.py`, `elph_symmetry.py`, and `elph_interpolation.py` are fractional reciprocal-lattice coordinates. Equivalent points differing by an integer reciprocal-lattice vector are identified, and canonical storage is in `[0, 1)`.

A regular mesh is described by `(N1, N2, N3)` plus an optional shift measured in units of one mesh spacing. A point therefore has coordinates

`q = ((i+s1)/N1, (j+s2)/N2, (k+s3)/N3)`.

The integer mesh coordinate `(i,j,k)` is the preferred key for regular-grid lookup, avoiding repeated floating-point searches.

## Electron-phonon matrix axes

The first axis of an electron-phonon array is q. The last two axes are the square Cartesian atomic-displacement matrix indices used by the existing SolveME deformation-potential workflow. The displacement basis is atom-major with the Cartesian components of each atom grouped together. Intermediate axes may contain smearing or electronic-band indices and are transformed independently by the same displacement-space symmetry matrices.

## Symmetry convention

The symmetry reconstruction follows the convention already used in `mesolver.get_elph_on_full_grid`. For a space-group operation `S`, the reciprocal point is generated using the transpose of the reciprocal-space rotation stored by CellConstructor. The Cartesian displacement representation is `Gamma(S,q)` from `CellConstructor.ThermalConductivity.construct_symmetry_matrix`.

The electron-phonon matrix transforms as

`D(Sq) = Gamma(S,q) D(q) Gamma(S,q)^dagger`.

If the target point is reached through time reversal, the transformed matrix is complex conjugated. Reciprocal-lattice shifts used to bring a transformed point onto the stored q-grid are retained in `QPointMapping`.

## Fourier interpolation gauge

Fourier interpolation is carried out in a periodic Cartesian displacement gauge. Let `tau_kappa` be the fractional direct-lattice coordinate of atom `kappa`. For matrix elements connecting Cartesian displacements on atoms `kappa` and `kappa'`, `elph_interpolation.py` defines

`D_periodic(kappa alpha, kappa' beta; q) = exp[-2*pi*i*q.(tau_kappa - tau_kappa')] D_input(kappa alpha, kappa' beta; q)`.

The inverse phase is restored after evaluating the Fourier series at a target q-point. `atom_positions` must therefore be supplied in fractional direct coordinates. Passing `atom_positions=None` intentionally disables the gauge conversion for data that are already periodic or for scalar/test quantities.

This convention is explicit and unit-tested, but the physical interface to a particular external electron-phonon file format must still be checked against that producer's phase convention before using interpolated values for production calculations. In particular, if Quantum ESPRESSO/EPW input matrices are later shown to already include/remove the basis-position phase differently, only the two gauge-conversion functions should need to change; the FFT and symmetry layers remain independent.

## Fourier transform convention

For a complete regular coarse mesh, the real-space representation is

`D(R) = (1/Nq) sum_q D_periodic(q) exp[-2*pi*i*q.R]`

and reconstruction/interpolation is

`D_periodic(q) = sum_R D(R) exp[+2*pi*i*q.R]`.

`R` is an integer Born-von-Karman lattice vector represented in signed FFT ordering. Shifted q meshes include the corresponding analytic phase factor so that a q->R->q round trip is exact on the original mesh.

The implementation first reorders q-points by integer mesh coordinates, performs a three-dimensional FFT on the q axes only, and keeps all intermediate smearing/band axes and the final displacement-matrix axes untouched.

## Validation expectations

The symmetry layer must satisfy identity mapping, q-star coverage, time reversal, deterministic selection when several operations reach the same q-point, and equality of matrices obtained through different valid symmetry paths. The last condition is checked by `validate_symmetry_collisions` and is intended primarily for tests and debugging.

The Fourier layer must satisfy: gauge conversion followed by inverse conversion recovers the original matrices; q->R->q reproduces every coarse-grid matrix; shifted meshes round-trip exactly; and interpolation onto a commensurate denser mesh reproduces the original values at every coarse q-point contained in the fine mesh.
