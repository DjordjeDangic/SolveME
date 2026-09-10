"""Public SolveME solver with symmetry/Fourier-interpolated electron-phonon grids.

This module keeps the existing :class:`mesolver.mesolver` implementation
unchanged and layers the dense-q harmonic path on top of it.  Calling
``calculate_a2f`` without ``interpolation_mesh`` delegates exactly to the
legacy solver.
"""

import numpy as np

import cellconstructor as CC
from cellconstructor.Units import RY_TO_MEV

from mesolver import mesolver as _BaseMesolver
from elph_dense import iter_dense_elph_mesh
from elph_interpolation import elph_to_real_space
from elph_symmetry import expand_irreducible_elph


class mesolver(_BaseMesolver):
    """SolveME solver with an opt-in dense-q Fourier-interpolation path."""

    def calculate_a2f(
        self,
        anharmonic=False,
        comm_dyn_filename=None,
        comm_nqirr=1,
        third_order_filename=None,
        nom=2000,
        scattering_mesh=(10, 10, 10),
        mode_mixing="no",
        on_full_grid=False,
        temperature=0.0,
        a2f_smearing=0.5,
        automatic_a2f_smearing=False,
        interpolation_mesh=None,
        interpolation_shift=(0.0, 0.0, 0.0),
        interpolation_block_size=64,
        validate_symmetry=False,
        phonon_evaluator=None,
    ):
        """Calculate alpha2F, optionally on a Fourier-interpolated q mesh.

        Parameters added by the dense-q extension
        -----------------------------------------
        interpolation_mesh : tuple(int, int, int) or None
            Target regular q mesh.  ``None`` preserves legacy SolveME behavior.
        interpolation_shift : tuple(float, float, float)
            Target-mesh shift in units of one mesh spacing.
        interpolation_block_size : int
            Number of target q points processed at once.
        validate_symmetry : bool
            Check all symmetry paths while reconstructing the coarse full grid.
        phonon_evaluator : callable, optional
            Compatibility adapter passed to ``elph_dense``.

        Notes
        -----
        Dense interpolation currently supports the harmonic isotropic path.
        Anharmonic lineshapes, mode mixing and multiband interpolation remain
        on the established irreducible-grid implementation.
        """
        if interpolation_mesh is None:
            return super().calculate_a2f(
                anharmonic=anharmonic,
                comm_dyn_filename=comm_dyn_filename,
                comm_nqirr=comm_nqirr,
                third_order_filename=third_order_filename,
                nom=nom,
                scattering_mesh=scattering_mesh,
                mode_mixing=mode_mixing,
                on_full_grid=on_full_grid,
                temperature=temperature,
                a2f_smearing=a2f_smearing,
                automatic_a2f_smearing=automatic_a2f_smearing,
            )

        if self.multiband:
            raise NotImplementedError(
                "Fourier-interpolated calculate_a2f currently supports isotropic calculations only"
            )
        if anharmonic:
            raise NotImplementedError(
                "Fourier-interpolated calculate_a2f currently supports harmonic phonons only"
            )
        if mode_mixing != "no":
            raise NotImplementedError(
                "Dense-q interpolation projects onto harmonic modes at each target q; "
                "anharmonic mode_mixing is not supported in this path"
            )
        if on_full_grid:
            raise ValueError(
                "interpolation_mesh already requests a full dense grid; do not also set on_full_grid=True"
            )
        if automatic_a2f_smearing:
            raise NotImplementedError(
                "Adaptive a2F smearing is not yet defined on arbitrary interpolated q points"
            )

        return self._calculate_a2f_dense_isotropic(
            mesh=interpolation_mesh,
            shift=interpolation_shift,
            block_size=interpolation_block_size,
            nom=nom,
            scattering_mesh=scattering_mesh,
            a2f_smearing=a2f_smearing,
            validate_symmetry=validate_symmetry,
            phonon_evaluator=phonon_evaluator,
        )

    def _build_coarse_tc(self, scattering_mesh, a2f_smearing):
        """Build the same harmonic TC object used by the legacy a2F path."""
        _, fc3 = self._build_fc3(False, None, 1, None)
        tc = CC.ThermalConductivity.ThermalConductivity(
            self.dyn,
            fc3,
            kpoint_grid=self.elph_supercell,
            scattering_grid=scattering_mesh,
            smearing_scale=None,
            smearing_type="constant",
            cp_mode="quantum",
            off_diag=False,
            phase_conv="step",
        )
        tc.setup_harmonic_properties(a2f_smearing)
        return tc

    def _fractional_atom_positions(self):
        """Return direct fractional atomic positions for the interpolation gauge."""
        cell = np.asarray(self.dyn.structure.unit_cell, dtype=float)
        coords = np.asarray(self.dyn.structure.coords, dtype=float)
        return np.dot(coords, np.linalg.inv(cell))

    def _mass_scaled_phonon_evaluator(self, phonons, qpoint):
        """Evaluate target-q phonons and reproduce SolveME's mass scaling."""
        if hasattr(phonons, "DyagDinQ"):
            freq, eig = phonons.DyagDinQ(qpoint)
        elif hasattr(phonons, "DiagonalizeQPoint"):
            freq, eig = phonons.DiagonalizeQPoint(qpoint)
        else:
            raise AttributeError(
                "CellConstructor phonon object has no recognized q-point diagonalizer"
            )
        eig = np.asarray(eig, dtype=complex).copy()
        if eig.shape[0] != 3 * self.dyn.structure.N_atoms:
            eig = eig.T
        for iat, at in enumerate(self.dyn.structure.atoms):
            eig[3 * iat : 3 * (iat + 1), :] /= np.sqrt(self.dyn.structure.masses[at])
        return np.asarray(freq, dtype=float), eig

    def _prepare_dense_real_space(self, tc, validate_symmetry=False):
        """Expand the irreducible coarse e-ph grid and Fourier transform it."""
        coarse = expand_irreducible_elph(
            tc,
            self.elph_qpts,
            self.ep_deformation_potentials,
            validate_collisions=validate_symmetry,
        )
        coarse.mesh = tuple(int(x) for x in self.elph_supercell)
        coarse.shift = (0.0, 0.0, 0.0)
        # Re-run dataclass validation now that mesh metadata are known.
        coarse.__post_init__()
        return elph_to_real_space(
            coarse,
            atom_positions=self._fractional_atom_positions(),
        )

    def _calculate_a2f_dense_isotropic(
        self,
        mesh,
        shift,
        block_size,
        nom,
        scattering_mesh,
        a2f_smearing,
        validate_symmetry,
        phonon_evaluator,
    ):
        tc = self._build_coarse_tc(scattering_mesh, a2f_smearing)
        real_space = self._prepare_dense_real_space(tc, validate_symmetry)
        mesh = tuple(int(x) for x in mesh)
        nq = int(np.prod(mesh))
        nph = 3 * self.dyn.structure.N_atoms

        evaluator = phonon_evaluator or self._mass_scaled_phonon_evaluator

        # Establish a safe spectral window from the target mesh without storing
        # any dense e-ph matrices.  The second streaming pass performs the
        # actual accumulation.
        max_freq = 0.0
        for block in iter_dense_elph_mesh(
            real_space,
            self.dyn,
            mesh,
            shift=shift,
            block_size=block_size,
            phonon_evaluator=evaluator,
        ):
            max_freq = max(max_freq, float(np.max(block.frequencies)))

        self.a2f_omega = (np.arange(nom, dtype=float) + 1.0) / float(nom) * (2.1 * max_freq)
        sigma = self.a2f_omega[1] * a2f_smearing
        self.a2f = np.zeros((self.ep_nsmear, nom), dtype=float)
        self.pdos = np.zeros(nom, dtype=float)
        self.lambdas = np.zeros((self.ep_nsmear, nq, nph), dtype=float)
        self.dense_q_weights = np.full(nq, 1.0 / max(nq - 1, 1), dtype=float)
        self.interpolation_mesh = mesh
        self.interpolation_shift = tuple(float(x) for x in shift)

        non_gamma = 0
        for block in iter_dense_elph_mesh(
            real_space,
            self.dyn,
            mesh,
            shift=shift,
            block_size=block_size,
            phonon_evaluator=evaluator,
        ):
            for iloc, qpoint in enumerate(block.qpoints):
                iq = block.start + iloc
                if np.linalg.norm(qpoint - np.rint(qpoint)) < 1.0e-10:
                    self.dense_q_weights[iq] = 0.0
                    continue
                non_gamma += 1
                freq = block.frequencies[iloc]
                positive = freq > 1.0e-12
                if not np.any(positive):
                    continue

                gauss = np.exp(
                    -0.5 * (self.a2f_omega[None, :] - freq[:, None]) ** 2 / sigma**2
                ) / (np.sqrt(2.0 * np.pi) * sigma)

                # mode_matrices: (nsmear, nmode, nmode) for the isotropic input.
                mode_diag = np.diagonal(
                    block.mode_matrices[iloc], axis1=-2, axis2=-1
                ).real
                for ism in range(self.ep_nsmear):
                    dos = float(self.elph_dos[ism])
                    mat = mode_diag[ism]
                    self.a2f[ism] += np.sum(
                        mat[:, None] * gauss / self.a2f_omega[None, :], axis=0
                    ) / (4.0 * dos)
                    safe_lambda = np.zeros_like(freq, dtype=float)
                    safe_lambda[positive] = mat[positive] / (
                        2.0 * freq[positive] ** 2 * dos
                    )
                    self.lambdas[ism, iq] = safe_lambda
                self.pdos += np.sum(gauss, axis=0)

        if non_gamma == 0:
            raise RuntimeError("target interpolation mesh contains no non-Gamma q points")

        # Equal-weight full-grid average, matching the legacy convention that
        # removes Gamma from the normalization.
        self.a2f /= float(non_gamma)
        self.pdos /= float(non_gamma)
        self.dense_q_weights /= np.sum(self.dense_q_weights)

        self.a2f_omega *= RY_TO_MEV
        self.pdos /= RY_TO_MEV
        return None

    def validate_coarse_grid_equivalence(
        self,
        nom=1000,
        scattering_mesh=(10, 10, 10),
        a2f_smearing=0.5,
        rtol=5.0e-3,
        atol=1.0e-8,
    ):
        """Compare legacy irreducible-grid and reconstructed coarse-full alpha2F."""
        super().calculate_a2f(
            nom=nom,
            scattering_mesh=scattering_mesh,
            a2f_smearing=a2f_smearing,
        )
        reference_omega = self.a2f_omega.copy()
        reference_a2f = self.a2f.copy()
        reference_lambda = np.array([self.get_lambda(i) for i in range(self.ep_nsmear)])

        self.calculate_a2f(
            nom=nom,
            scattering_mesh=scattering_mesh,
            a2f_smearing=a2f_smearing,
            interpolation_mesh=tuple(int(x) for x in self.elph_supercell),
            validate_symmetry=True,
        )
        # Frequency windows can differ slightly because the target-q phonons
        # are evaluated independently.  Compare alpha2F on the common interval.
        common_max = min(reference_omega[-1], self.a2f_omega[-1])
        common = np.linspace(max(reference_omega[0], self.a2f_omega[0]), common_max, nom)
        ref_interp = np.array([np.interp(common, reference_omega, a) for a in reference_a2f])
        dense_interp = np.array([np.interp(common, self.a2f_omega, a) for a in self.a2f])
        dense_lambda = np.array([self.get_lambda(i) for i in range(self.ep_nsmear)])

        return {
            "a2f_close": bool(np.allclose(ref_interp, dense_interp, rtol=rtol, atol=atol)),
            "reference_lambda": reference_lambda,
            "dense_lambda": dense_lambda,
            "lambda_relative_error": np.abs(dense_lambda - reference_lambda)
            / np.maximum(np.abs(reference_lambda), atol),
            "common_omega": common,
            "reference_a2f": ref_interp,
            "dense_a2f": dense_interp,
        }

    def converge_q_meshes(
        self,
        meshes,
        *,
        smear_id=0,
        solve_kwargs=None,
        calculate_kwargs=None,
    ):
        """Run dense-q convergence for lambda, omega_log and optionally Tc/gap."""
        calculate_kwargs = {} if calculate_kwargs is None else dict(calculate_kwargs)
        rows = []
        for mesh in meshes:
            self.calculate_a2f(interpolation_mesh=tuple(mesh), **calculate_kwargs)
            row = {
                "mesh": tuple(int(x) for x in mesh),
                "nq": int(np.prod(mesh)),
                "lambda": float(self.get_lambda(smear_id=smear_id)),
                "omega_log_meV": float(self.get_omega_log(units="meV", smear_id=smear_id)),
                "a2f_omega": self.a2f_omega.copy(),
                "a2f": self.get_tot_a2f(smear_id=smear_id).copy(),
            }
            if solve_kwargs is not None:
                kwargs = dict(solve_kwargs)
                kwargs.setdefault("smear_id", smear_id)
                self.solve(**kwargs)
                row["tc"] = getattr(self, "tc", None)
                if self.delta and self.indices:
                    row["gap_low_T_meV"] = float(
                        np.asarray(self.delta[0])[..., self.indices[0]].reshape(-1)[0]
                    )
            rows.append(row)
        return rows
