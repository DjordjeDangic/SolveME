import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ThermalConductivity
from cellconstructor.Units import *
from scipy.optimize import brentq
from parsing import read_elph, read_wee
import numpy as np
import os
import time


EV_TO_MEV = 1000.0000000000000000000
RY_TO_MEV = RY_TO_EV * EV_TO_MEV

import numpy as np

def pade_analytic_continuation_fast(z_i, u_i, z_o, eps= 1.0e-14):
    """
    Optimized multipoint Pade approximant (Vidberg–Serene)
    """
    z_i = np.asarray(z_i, dtype=np.complex128)
    u_i = np.asarray(u_i, dtype=np.complex128)
    z_o = np.asarray(z_o, dtype=np.complex128)

    np_i = z_i.size
    np_o = z_o.size

    # Preallocate triangular table
    g = np.zeros((np_i, np_i), dtype=np.complex128)

    # First row
    g[0] = u_i

    # Build recursion table (critical part)
    for ip in range(1, np_i):
        gip_prev_prev = g[ip - 1, ip - 1]   # scalar
        zi_prev = z_i[ip - 1]
        g_prev = g[ip - 1]                  # whole row ref
        gip = g[ip]                         # target row ref

        for jp in range(ip, np_i):
            denom = (z_i[jp] - zi_prev) * g_prev[jp]
            gip[jp] = (gip_prev_prev - g_prev[jp]) / denom + eps

    # Extract diagonal
    dia = np.diag(g)

    # Allocate output
    u_o = np.zeros(np_o, dtype=np.complex128)

    # Evaluate continued fraction
    for k in range(np_o):
        D = 1.0 + 0j
        zk = z_o[k]
        for j in range(np_i - 1, 0, -1):
            D = 1.0 + dia[j] * (zk - z_i[j - 1]) / D
        u_o[k] = dia[0] / D

    return u_o

def load_mesolver(filename = 'sc.pkl'):

    import pickle

    infile = open(filename, 'rb')
    sc = pickle.load(infile)
    infile.close()
    return sc

def get_starting_number_electrons(wee_energy, wee_dos, efermi, T):

    occs = Fermi_Dirac(wee_energy, efermi, T)
    if wee_dos.ndim == 1:
        y = occs * wee_dos
        return np.trapz(y, wee_energy)

    # 2-D case
    y = occs[None, :] * wee_dos
    return np.sum(np.trapz(y, wee_energy, axis=1))

def get_number_electrons(efermi, wee_energy, wee_dos, T, w, chi, theta):
    """
    Optimized electron number evaluation for isotropic or multiband case.
    """

    # Compute Fermi-Dirac once
    occs = Fermi_Dirac(wee_energy, efermi, T)

    # Shared variables
    wee_diff = wee_energy - efermi          # shape: (Ne)
    w2 = w * w                              # shape: (Nw,)

    # ---- Isotropic: wee_dos.shape == (Ne,) --------------------------------
    if wee_dos.ndim == 1:
        # Broadcast shapes:
        # wee_diff[:,None]  -> (Ne, Nw)
        # chi[None,:]       -> (Ne, Nw)
        # w2[None,:]        -> (Ne, Nw)
        denom_free = w2[None, :] + wee_diff[:, None]**2

        # sum over Matsubara frequencies
        sumand = np.sum(
            (wee_diff[:, None] + chi[None, :]) / theta
            - wee_diff[:, None] / denom_free,
            axis=1
        )

        integrand = occs * wee_dos - 2.0 * T * sumand * wee_dos
        return np.trapz(integrand, wee_energy)

    # ---- Multiband: wee_dos.shape == (Nb, Ne) ------------------------------
    # Broadcasting dimensions:
    # wee_diff[None,:,None]  -> (1, Ne, 1)
    # chi[:,None,:]          -> (Nb,  1, Nw)
    # w2[None,None,:]        -> (1,   1, Nw)
    denom_free = w2[None, None, :] + wee_diff[None, :, None]**2

    sumand = np.sum(
        (wee_diff[None, :, None] + chi[:, None, :]) / theta
        - wee_diff[None, :, None] / denom_free,
        axis=2
    )

    # integrand shape: (Nb, Ne)
    integrand = occs * wee_dos - 2.0 * T * sumand * wee_dos

    # integrate over energies for each band, then sum over bands
    return np.sum(np.trapz(integrand, wee_energy, axis=1), axis=0)

def fermi_root(efermi, wee_energy, wee_dos, T, w, chi, theta, Ne):
    return get_number_electrons(efermi, wee_energy, wee_dos, T, w, chi, theta) - Ne

def Fermi_Dirac(x, efermi, temperature):

    if(temperature > 0.0):
        return 1.0/(np.exp((x - efermi)/temperature) + 1.0)
    else:
        return np.where(x <= efermi, 1.0, 0.0)

def gaussian(x, x0, smearing):

    x1 = x[np.newaxis, np.newaxis, :]
    x2 = x0[:, np.newaxis]
    x3 = smearing[:, np.newaxis]

    return np.exp(-0.5*(x1-x2)**2/x3**2)/np.sqrt(2.0*np.pi)/x3

def match_qpoints(qpts, tc):

    found = [False for x in range(len(qpts))]
    found[0] = True
    tc_qpt_id = np.zeros(len(qpts), dtype=int)  # Index of the q point for which electron-phonon was calculated
    tc_qpt_id1 = np.zeros(len(qpts), dtype=int) # Index of the q point for which lineshape was directly calculated
    for iqpt in range(len(qpts)):
        for jqpt in range(len(tc.qpoints)):
            diff = qpts[iqpt] - tc.qpoints[jqpt]
            diff -= np.rint(diff)
            if(np.linalg.norm(diff) < 1.0e-4):
                tc_qpt_id1[iqpt] = jqpt
                found_star = False
                for istar in range(len(tc.qstar)):
                    for kqpt in range(len(tc.qstar[istar])):
                        if(jqpt == tc.qstar[istar][kqpt]):
                            tc_qpt_id[iqpt] = tc.qstar[istar][0]
                            found_star = True
                            break
                    if(found_star):
                        break
                found[iqpt] = True
                break
        if(not found[iqpt] or not found_star):
            print('Could not find this point in calculated lineshapes!')
            raise RuntimeError(qpts[iqpt])

    return tc_qpt_id, tc_qpt_id1

def get_elph_on_full_grid(tc, elph, tc_qpt_id, star_id):

    rotations, translations = tc.get_sg_in_cartesian()
    elph_full = np.zeros((tc.nkpt, tc.nband, tc.nband), dtype=complex)
    for iqpt in range(len(tc_qpt_id)):
        ikpt = tc_qpt_id[iqpt]
        qpt1 = tc.qpoints[ikpt]
        elph_full[ikpt] = elph[iqpt].copy()
        for istar in range(len(tc_qpt_id)):
            if(tc.qstar[istar][0] == star_id[iqpt]):
                break
            for jqpt in range(len(tc.qstar[istar])):
                jkpt = tc.qstar[istar][jqpt]
                qpt2 = tc.qpoints[jkpt]
                if(ikpt != jkpt):
                    found = False
                    if(np.linalg.norm(qpt1 + qpt2 - np.rint(qpt1 + qpt2)) < 1.0e-6):
                        elph_full[jkpt] = elph[iqpt].conj().copy()
                        found = True
                    else:
                        for irot in range(len(tc.rotations)):
                            qpt21 = np.dot(tc.rotations[irot].T, qpt1)
                            kpt21 = np.dot(qpt21, tc.reciprocal_lattice)
                            diffq = qpt21 - qpt2
                            addq = qpt21 + qpt2
                            if(np.linalg.norm(diffq - np.rint(diffq)) < 1.0e-6):
                                gamma = CC.ThermalConductivity.construct_symmetry_matrix(rotations[irot], translations[irot], kpt21, tc.dyn.structure.coords, tc.atom_map[irot], tc.unitcell)
                                elph_full[jkpt] = np.einsum('ij,jk,kl->il', gamma, elph[iqpt], gamma.conj().T)
                                found = True
                                break
                            elif(np.linalg.norm(addq - np.rint(addq)) < 1.0e-6):
                                gamma = CC.ThermalConductivity.construct_symmetry_matrix(rotations[irot], translations[irot], kpt21, tc.dyn.structure.coords, tc.atom_map[irot], tc.unitcell)
                                elph_full[jkpt] = np.einsum('ij,jk,kl->il', gamma, elph[iqpt], gamma.conj().T).conj()
                                found = True
                                break
                    if(not found):
                        raise RuntimeError('Could not find mapping between q points in the construction of a full ELPH grid!')
    return elph_full


class mesolver:

    """

    Solver for the isotropic Migdal-Eliashberg equations 

    """

    def __init__(self, a2f = None, wee = None, multiband = False):

        self.a2f = a2f
        self.wee = wee
        self.pdos = None
        self.lambdas = None

        self.multiband = multiband
        self.nmultiband = 0 
        self.keep_bands = []
        self.dos_thr = 1.0e-7
        self.av_mapping = None

        self.thr = 1.0e-4
        self.mix = 1.0
        self.max_iter = 1000
        self.starting_Coulomb_weight = 20.0
        self.shift_guess = 50.0
        self.aa = False
        self.anderson = 4
        self.beta = 0.3

        self.wee_freq = None
        self.wee_energy = None
        self.wee_dos = None
        self.wee = None

    def load(
        self,
        dyn_filename=None,
        nqirr=1,
        elphdyn_filename=None,
        wee_filename=None,
        nband_el=None,
        av_mapping=None,
    ):
        """
        Load dynamical matrices, electron-phonon interactions, and Coulomb kernels.
        """


        if self.multiband and nband_el is None:
            raise ValueError("Calculation set to multiband, but total number of electronic bands was not provided!")
        elif(not self.multiband and nband_el):
            print(' Warning! Requested isotropic calculation, but reading in band resolved properties.')
            print(' We will calculate isotropic values from band resolved ones ...')

        self.av_mapping = av_mapping
        dyn = None
        if dyn_filename is not None:
            dyn = CC.Phonons.Phonons(dyn_filename, nqirr)
            self.dyn = dyn

        if elphdyn_filename is not None:
            self._load_elph_data(elphdyn_filename, dyn, nband_el)

        if wee_filename is not None:
            self._load_coulomb_data(wee_filename)

        if dyn_filename is None and elphdyn_filename is None and wee_filename is None:
            print("You need provide a path to a dynamical matrix, deformation potential, or Coulomb interaction file!")


    def _load_elph_data(self, elphdyn_filename, dyn, nband_el):
        """Load deformation potentials and DOS from EPW-like files."""

        iqr = 1
        while os.path.isfile(elphdyn_filename + str(iqr)):
            iqr += 1
        elph_nqirr = iqr - 1
        if(elph_nqirr == 0):
            raise RuntimeError('Could not find any of the electron-phonon matrix element files !')

        qpts, smearings, dos, elph, weights, qstar = read_elph(elphdyn_filename, elph_nqirr, dyn.structure.N_atoms, nband_el)

        if(not self.multiband and nband_el is not None):
            self.elph_qpts = qpts
            self.elph_smearings = smearings
            self.elph_dos = np.sum(np.array(dos), axis = 0)
            self.ep_deformation_potentials = np.sum(np.array(elph), axis = (1,2))
            self.ep_nqirr = elph.shape[0]
            self.ep_nsmear = elph.shape[1]
            self.elph_weights = weights
            self.elph_qstar = qstar
        else:
            self.elph_qpts = qpts
            self.elph_smearings = smearings
            self.elph_dos = np.array(dos)
            self.ep_deformation_potentials = elph
            self.ep_nqirr = elph.shape[0]
            self.ep_nsmear = elph.shape[1]
            self.elph_weights = weights
            self.elph_qstar = qstar

        # Handle multiband
        if self.multiband:
            self.keep_bands = [ib for ib in range(nband_el) if np.any(self.elph_dos[ib] >= self.dos_thr)]
            if(self.av_mapping is None):
                self.nmultiband = len(self.keep_bands)
            else:
                if(len(self.av_mapping) != len(self.keep_bands)):
                    raise RuntimeError(' The provided averaging map is not consistent with number of bands crossing the Fermi level! ')
                else:
                    self.nmultiband = len(np.unique(self.av_mapping))

            if self.nmultiband == 0:
                raise RuntimeError("None of the DOS values at the Fermi level exceed threshold. Is it a metal?")

            print("In the multiband calculation, keeping bands:", self.keep_bands)

        # Load supercell
        elph_dyn = CC.Phonons.Phonons(elphdyn_filename, elph_nqirr)
        self.elph_supercell = elph_dyn.GetSupercell()

    def _load_coulomb_data(self, wee_filename):

        """Load Coulomb interaction W(ε,ε') and the electronic DOS."""
        
        from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline

        freq, wee_energy, dos_energy, dos, re_wee, im_wee = read_wee(wee_filename)

        is_multi_file = re_wee.shape[-1] > 1

        if self.multiband and not is_multi_file:
            raise ValueError("Requested multiband calculation, but provided isotropic Coulomb data!")

        correct_mapping = False
        if(re_wee.shape[-1] == self.nmultiband):
            correct_mapping = True
        elif(re_wee.shape[-1] != self.nmultiband and self.av_mapping is not None and re_wee.shape[-1] == len(self.av_mapping)):
            correct_mapping = True
        if self.multiband and not correct_mapping:
            raise ValueError("Mismatch between number of bands in ELPH and Coulomb data.")

        self.wee_freq = freq * EV_TO_MEV

        mask = (dos_energy >= wee_energy.min()) & (dos_energy <= wee_energy.max())
        using_dos_energy = dos_energy[mask]
        using_dos = dos[mask]

        self.wee_energy = using_dos_energy * EV_TO_MEV

        if self.multiband:
            if(self.av_mapping is None):
                self.wee_dos = using_dos.T / EV_TO_MEV
            else:
                self.wee_dos = np.zeros((self.nmultiband, len(self.wee_energy)), dtype=float)
                for i in range(len(self.av_mapping)):
                    ib = self.av_mapping[i]
                    self.wee_dos[ib] += using_dos[:,i]
        else:
            if(not is_multi_file):
                self.wee_dos = using_dos[:, 0] / EV_TO_MEV
            else:
                self.wee_dos = np.sum(using_dos, axis = 1) #/ EV_TO_MEV

        if self.multiband:
            if(self.av_mapping is None):
                self.wee = self._interpolate_multiband_coulomb(wee_energy, using_dos_energy, re_wee) * EV_TO_MEV
            else:
                wee = self._interpolate_multiband_coulomb(wee_energy, using_dos_energy, re_wee)
                wee_shape = (self.nmultiband, self.nmultiband, len(self.wee_energy), len(self.wee_energy))
                self.wee = np.zeros(wee_shape, dtype=float)
                for i in range(wee.shape[0]):
                    ib = self.av_mapping[i]
                    for j in range(wee.shape[1]):
                        jb = self.av_mapping[j]
                        self.wee[ib,jb] += wee[i,j] * np.outer(using_dos[:,i], using_dos[:,j])
                for ib in range(self.nmultiband):
                    for jb in range(self.nmultiband):
                        self.wee[ib,jb] = np.divide(self.wee[ib,jb], np.outer(self.wee_dos[ib], self.wee_dos[jb]), out = np.zeros_like(self.wee[ib,jb]), where = np.outer(self.wee_dos[ib], self.wee_dos[jb]) > 1.0e-6) * EV_TO_MEV
                self.wee_dos /= EV_TO_MEV
        else:
            if(not is_multi_file):
                interp = RectBivariateSpline(wee_energy, wee_energy, re_wee[0, :, :, 0, 0], kx=3, ky=3)
                self.wee = interp(using_dos_energy, using_dos_energy) * EV_TO_MEV
            else:
                wee = self._interpolate_multiband_coulomb(wee_energy, using_dos_energy, re_wee) #* EV_TO_MEV
                self.wee = np.zeros_like(wee[0, 0])
                for ib in range(wee.shape[0]):
                    for jb in range(wee.shape[1]):
                        self.wee += wee[ib,jb] * np.outer(using_dos[:,ib], using_dos[:,jb])
                self.wee = np.divide(self.wee, np.outer(self.wee_dos, self.wee_dos), out = np.zeros_like(self.wee), where = np.outer(self.wee_dos, self.wee_dos) > 1.0e-6) * EV_TO_MEV
                self.wee_dos /= EV_TO_MEV

    def _interpolate_multiband_coulomb(self, wee_energy, using_dos_energy, re_wee):

        """Interpolate multiband Coulomb matrix W_ij(ε, ε') on trimmed DOS grid."""

        from scipy.interpolate import RectBivariateSpline

        if(self.multiband):
            nb = self.nmultiband
        else:
            nb = re_wee.shape[-1]
        out = np.zeros((nb, nb, len(using_dos_energy), len(using_dos_energy)))

        for i in range(nb):
            for j in range(nb):
                interp = RectBivariateSpline(wee_energy, wee_energy, re_wee[0, :, :, i, j], kx=3, ky=3)
                out[i, j] = interp(using_dos_energy, using_dos_energy)

        return out

    def calculate_a2f(self, anharmonic = False, comm_dyn_filename = None, comm_nqirr = 1, third_order_filename = None, nom = 2000, scattering_mesh = [10,10,10], mode_mixing = 'no', on_full_grid = False, temperature = 0.0, a2f_smearing = 0.5, automatic_a2f_smearing = False):
        
        if(self.multiband):
            if(on_full_grid):
                print(' Warning! Requested on full grid, but also multiband calculation. This would be too memory intensive, so we will just calculate electron-phonon on irreducible grid.')
            self.calculate_a2f_multiband(anharmonic, comm_dyn_filename, comm_nqirr, third_order_filename, nom, scattering_mesh, mode_mixing, temperature, a2f_smearing, automatic_a2f_smearing)
        else:
            self.calculate_a2f_isotropic(anharmonic, comm_dyn_filename, comm_nqirr, third_order_filename, nom, scattering_mesh, mode_mixing, on_full_grid, temperature, a2f_smearing, automatic_a2f_smearing)

    def calculate_a2f_isotropic(
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
    ):
        """
        Compute isotropic Eliashberg α²F(ω).
        """

        # 1) Build FC3 tensor and comm_dyn (harmonic or anharmonic)
        comm_dyn, fc3 = self._build_fc3(anharmonic=anharmonic, comm_dyn_filename=comm_dyn_filename, comm_nqirr=comm_nqirr, third_order_filename=third_order_filename)

        # 2) Build TC engine
        tc = CC.ThermalConductivity.ThermalConductivity(self.dyn, fc3, kpoint_grid=self.elph_supercell, scattering_grid=scattering_mesh, smearing_scale=1.0, smearing_type="adaptive", cp_mode="quantum", off_diag=False, phase_conv="smooth")
        tc.setup_harmonic_properties()

        # Match q-points
        qpt_id, qpt_id1 = match_qpoints(self.elph_qpts, tc)

        # 3) a2F frequency grid
        freq_max = np.max(tc.freqs) * 2.1
        self.a2f_omega = (np.arange(nom, dtype=float) + 1.0) / float(nom) * freq_max
        smearing = self.a2f_omega[1] * a2f_smearing

        # 4) Allocate arrays for a2F, projected DOS, lambdas
        self._allocate_a2f_arrays(tc, nom=nom, mode_mixing=mode_mixing, on_full_grid=on_full_grid, anharmonic=anharmonic)

        # 5) Compute projected DOS (lineshapes or Gaussian)
        self.pq_dos = self._compute_projected_dos(tc, qpt_id, qpt_id1, anharmonic=anharmonic, temperature=temperature, smearing=smearing, mode_mixing=mode_mixing, on_full_grid=on_full_grid, automatic_a2f_smearing=automatic_a2f_smearing)

        # 6) If needed, build deformation potentials on full grid
        elph_full = None
        if on_full_grid and mode_mixing != "no":
            # Uses your existing get_elph_on_full_grid implementation
            elph_full = get_elph_on_full_grid(tc, self.ep_deformation_potentials, qpt_id1, qpt_id)
        elif on_full_grid and mode_mixing == "no":
            raise RuntimeError(
                "on_full_grid=True but mode_mixing=='no'. Either enable mode "
                "mixing or disable full grid."
            )

        # 7) Accumulate α²F(ω), λ, and PDOS
        self._accumulate_a2f_isotropic(tc, qpt_id=qpt_id, qpt_id1=qpt_id1, elph_full=elph_full, mode_mixing=mode_mixing, on_full_grid=on_full_grid, anharmonic=anharmonic)

        # 8) Final unit conversion and normalisation
        self.a2f_omega *= RY_TO_MEV
        norm = float(np.sum(self.elph_weights) - 1)
        self.a2f /= norm
        self.pdos /= norm * RY_TO_MEV

    def calculate_a2f_multiband(
        self,
        anharmonic=False,
        comm_dyn_filename=None,
        comm_nqirr=1,
        third_order_filename=None,
        nom=2000,
        scattering_mesh=(10, 10, 10),
        mode_mixing="no",
        temperature=0.0,
        a2f_smearing=0.5,
        automatic_a2f_smearing=False,
    ):
        """
        Compute multiband Eliashberg α²F(ω)_{ij}.

        Shapes:
            self.a2f.shape    = (ep_nsmear, nmultiband, nmultiband, nom)
            self.pdos.shape   = (nom,)
            self.lambdas.shape= (ep_nsmear, ep_nqirr, nphonon, nmultiband, nmultiband)
        """

        # 1) Build FC3 tensor
        comm_dyn, fc3 = self._build_fc3(anharmonic=anharmonic, comm_dyn_filename=comm_dyn_filename, comm_nqirr=comm_nqirr, third_order_filename=third_order_filename)

        # 2) TC engine
        tc = CC.ThermalConductivity.ThermalConductivity(self.dyn, fc3, kpoint_grid=self.elph_supercell, scattering_grid=scattering_mesh, smearing_scale=1.0, smearing_type="adaptive", cp_mode="quantum", off_diag=False, phase_conv="smooth")
        tc.setup_harmonic_properties()

        qpt_id, qpt_id1 = match_qpoints(self.elph_qpts, tc)

        # 3) a2F frequency grid
        freq_max = np.max(tc.freqs) * 2.1
        self.a2f_omega = (np.arange(nom, dtype=float) + 1.0) / float(nom) * freq_max
        smearing = self.a2f_omega[1] * a2f_smearing

        # 4) Unified allocation (multiband branch will be used)
        self._allocate_a2f_arrays(tc, nom=nom, mode_mixing=mode_mixing, on_full_grid=False, anharmonic=anharmonic)

        # 5) Projected DOS
        self.pq_dos = self._compute_projected_dos(tc, qpt_id, qpt_id1, anharmonic=anharmonic, temperature=temperature, smearing=smearing, mode_mixing=mode_mixing, on_full_grid=False, automatic_a2f_smearing=automatic_a2f_smearing)

        # 6) Accumulate α²F_ij(ω), λ, and PDOS
        self._accumulate_a2f_multiband(tc, qpt_id=qpt_id, qpt_id1=qpt_id1, mode_mixing=mode_mixing, anharmonic=anharmonic)

        # 7) Units / normalization
        self.a2f_omega *= RY_TO_MEV
        norm = float(np.sum(self.elph_weights) - 1)
        self.a2f /= norm
        self.pdos /= norm * RY_TO_MEV

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _build_fc3(
        self,
        anharmonic,
        comm_dyn_filename,
        comm_nqirr,
        third_order_filename,
    ):
        """
        Construct FC3 tensor need for ThermalConductivity object.
        """

        if not anharmonic:
            # Harmonic branch: interpolate self.dyn
            # Mock dynamical matrix and FC3
            comm_dyn = self.dyn.Interpolate(coarse_grid=self.dyn.GetSupercell(), fine_grid=[1, 1, 1])
            comm_supercell = comm_dyn.GetSupercell()

            fc3 = CC.ForceTensor.Tensor3(comm_dyn.structure, comm_dyn.structure.generate_supercell(comm_supercell), comm_supercell)

            return comm_dyn, fc3

        # Anharmonic branch, real dynamical matrix and FC3
        if comm_dyn_filename is None or third_order_filename is None:
            raise RuntimeError(
                "Anharmonic=True but dyn/third_order paths not provided."
            )

        comm_dyn = CC.Phonons.Phonons(comm_dyn_filename, comm_nqirr)
        comm_supercell = comm_dyn.GetSupercell()

        fc3 = CC.ForceTensor.Tensor3(comm_dyn.structure, comm_dyn.structure.generate_supercell(comm_supercell), comm_supercell)

        d3 = np.load(third_order_filename) * 2.0
        fc3.SetupFromTensor(d3)
        fc3 = CC.ThermalConductivity.centering_fc3(fc3)

        return comm_dyn, fc3

    def _scaled_eigvecs(self, tc, iq):
        """
        Mass-scale eigenvectors at q-point index 'iq'.
        """

        eig = tc.eigvecs[iq].copy()
        structure = tc.dyn.structure
        masses = structure.masses
        atoms = structure.atoms

        for iat, at in enumerate(atoms):
            eig[3 * iat : 3 * (iat + 1), :] /= np.sqrt(masses[at])

        return eig

    def _allocate_a2f_arrays(
        self,
        tc,
        nom,
        mode_mixing="no",
        on_full_grid=False,
        anharmonic=False,
    ):
        """
        Unified allocation for isotropic and multiband a2F calculations.

        Isotropic:
            self.a2f   : (ep_nsmear, nom)
            self.pdos  : (nom,)
            self.pq_dos:
                - no mixing / or not anharmonic:
                      (ep_nqirr, nphonon, nom)
                - mixing:
                      on_full_grid=False: (ep_nqirr, nphonon, nphonon, nom)
                      on_full_grid=True : (tc.nkpt, nphonon, nphonon, nom)
            self.lambdas:
                on_full_grid=False: (ep_nsmear, ep_nqirr, nphonon)
                on_full_grid=True : (ep_nsmear, tc.nkpt, nphonon)

        Multiband:
            self.a2f   : (ep_nsmear, nmultiband, nmultiband, nom)
            self.pdos  : (nom,)
            self.pq_dos:
                - no mixing / or not anharmonic:
                      (ep_nqirr, nphonon, nom)
                - mixing:
                      (ep_nqirr, nphonon, nphonon, nom)
            self.lambdas:
                (ep_nsmear, ep_nqirr, nphonon, nmultiband, nmultiband)
        """

        nsm = self.ep_nsmear
        nq = self.ep_nqirr
        nph = tc.nband

        # Common a2f / pdos
        if not getattr(self, "multiband", False):
            self.a2f = np.zeros((nsm, nom), dtype=float)
        else:
            self.a2f = np.zeros((nsm, self.nmultiband, self.nmultiband, nom), dtype=float)
        self.pdos = np.zeros(nom, dtype=float)

        # Isotropic branch
        if not getattr(self, "multiband", False):
            if mode_mixing == "no" or not anharmonic:
                # pq_dos: (q, phonon, w)
                self.pq_dos = np.zeros((nq, nph, nom), dtype=float)
                # lambdas: (smear, q, phonon)
                self.lambdas = np.zeros((nsm, nq, nph), dtype=float)
            else:
                # mode mixing
                if on_full_grid:
                    self.pq_dos = np.zeros((tc.nkpt, nph, nph, nom), dtype=complex)
                    self.lambdas = np.zeros((nsm, tc.nkpt, nph), dtype=float)
                else:
                    self.pq_dos = np.zeros((nq, nph, nph, nom), dtype=complex)
                    self.lambdas = np.zeros((nsm, nq, nph), dtype=float)
            return

        # Multiband branch
        if mode_mixing == "no" or not anharmonic:
            self.pq_dos = np.zeros((nq, nph, nom), dtype=float)
        else:
            self.pq_dos = np.zeros((nq, nph, nph, nom), dtype=float)

        self.lambdas = np.zeros((nsm, nq, nph, self.nmultiband, self.nmultiband), dtype=float)

    def _compute_projected_dos(
        self,
        tc,
        qpt_id,
        qpt_id1,
        anharmonic,
        temperature,
        smearing,
        mode_mixing,
        on_full_grid,
        automatic_a2f_smearing,
    ):
        """
        Compute phonon projected DOS pq_dos.

        Uses:
           - anharmonic lineshapes (tc.get_lineshapes)
           - or Gaussian broadening (gaussian)
        """

        pq_dos = self.pq_dos

        if anharmonic:
            key = format(temperature, ".1f")
            tc.get_lineshapes(temperature, write_lineshapes=False, energies=self.a2f_omega, method="fortran", mode_mixing=mode_mixing, gauss_smearing=False)

            # Full grid (isotropic only, mode mixing)
            if on_full_grid:
                for ik in range(tc.nkpt):
                    if mode_mixing == "cartesian":
                        eig = tc.eigvecs[ik]
                        pq_dos[ik] = np.einsum("ji,jkl,km->iml", eig, tc.lineshapes[key][ik], eig.conj()).real
                    else:
                        pq_dos[ik] = tc.lineshapes[key][ik]
            else:
                for i, iq in enumerate(qpt_id):
                    ls = tc.lineshapes[key][iq]
                    if mode_mixing == "cartesian":
                        eig = tc.eigvecs[iq]
                        pq_dos[i] = self.elph_weights[i] * np.einsum("ji,jkl,km->iml", eig,ls, eig.conj()).real
                    else:
                        pq_dos[i] = self.elph_weights[i] * ls

        else:
            # harmonic case: Gaussian broadening
            for i, iq in enumerate(qpt_id1):
                freqs = tc.freqs[iq]
                if automatic_a2f_smearing:
                    sigma = tc.sigmas[iq]
                else:
                    sigma = np.full_like(freqs, smearing, dtype=float)

                pq_dos[i] = self.elph_weights[i] * gaussian(self.a2f_omega, freqs, sigma)

        return pq_dos

    # ------------------------------------------------------------------
    # Accumulation routines
    # ------------------------------------------------------------------
    def _accumulate_a2f_isotropic(
        self,
        tc,
        qpt_id,
        qpt_id1,
        elph_full,
        mode_mixing,
        on_full_grid,
        anharmonic,
    ):
        """
        Accumulate isotropic α²F(ω), λ, and phonon DOS.
        Removes self.aq and writes directly to self.a2f.
        """

        nsm = self.ep_nsmear

        for ism in range(nsm):
            deformation_potentials = self.ep_deformation_potentials[:, ism, :, :]
            dos = self.elph_dos[ism]

            if on_full_grid:
                # Full grid case (only with mode mixing)
                for ik in range(tc.nkpt):
                    if np.linalg.norm(tc.qpoints[ik]) < 1e-6:
                        continue

                    eigs_scaled = self._scaled_eigvecs(tc, ik)
                    elph_mat = elph_full[ik]

                    if mode_mixing == "no" or not anharmonic:
                        # This branch is logically not used with full grid,
                        # but kept for completeness.
                        mat = np.einsum("ji,jk,ki->i", eigs_scaled, elph_mat,eigs_scaled.conj()).real
                        contrib = np.einsum("i,ik,k->k", mat, self.pq_dos[ik], 1.0 / self.a2f_omega)/ 4.0 / dos
                        self.a2f[ism] += contrib
                        self.lambdas[ism, ik] = mat / (2.0 * tc.freqs[ik] ** 2 * dos)
                        if ism == 0:
                            self.pdos += np.sum(self.pq_dos[ik], axis=0)
                    else:
                        mat = np.einsum("ji,jk,kl->il",eigs_scaled,elph_mat,eigs_scaled.conj()).real
                        contrib = np.einsum("ij,ijk,k->k", mat, self.pq_dos[ik], 1.0 / self.a2f_omega)/4.0/dos
                        self.a2f[ism] += contrib
                        self.lambdas[ism, ik] = np.diag(mat) / (2.0 * tc.freqs[ik] ** 2 * dos)
                        if ism == 0:
                            self.pdos += np.einsum("iik->k", self.pq_dos[ik])
            else:
                # Not on full grid (original mainstream isotropic path)
                for i, iq in enumerate(qpt_id1):
                    if np.linalg.norm(tc.qpoints[iq]) < 1e-6:
                        continue

                    eigs_scaled = self._scaled_eigvecs(tc, iq)

                    if mode_mixing == "no" or not anharmonic:
                        mat = np.einsum("ji,jk,ki->i",eigs_scaled, deformation_potentials[i], eigs_scaled.conj()).real
                        contrib = np.einsum("i,ik,k->k", mat, self.pq_dos[i],1.0 / self.a2f_omega)/4.0/dos
                        self.a2f[ism] += contrib
                        self.lambdas[ism, i] = mat / (2.0 * tc.freqs[iq] ** 2 * dos)
                        if ism == 0:
                            self.pdos += np.sum(self.pq_dos[i], axis=0)
                    else:
                        mat = np.einsum("ji,jk,kl->il", eigs_scaled, deformation_potentials[i], eigs_scaled.conj()).real
                        contrib = np.einsum("ij,ijk,k->k", mat, self.pq_dos[i], 1.0 / self.a2f_omega)/4.0/dos
                        self.a2f[ism] += contrib
                        self.lambdas[ism, i] = np.diag(mat) / (2.0 * tc.freqs[iq] ** 2 * dos)
                        if ism == 0:
                            self.pdos += np.einsum("iik->k", self.pq_dos[i])

    def _accumulate_a2f_multiband(
        self,
        tc,
        qpt_id,
        qpt_id1,
        mode_mixing,
        anharmonic,
    ):
        """
        Accumulate multiband α²F_ij(ω), λ_ij, and PDOS.
        Removes self.aq, writes directly to self.a2f.
        """

        nsm = self.ep_nsmear
        nmb = len(self.keep_bands)#self.nmultiband

        for ism in range(nsm):
            deformation_potentials = self.ep_deformation_potentials[:, :, :, ism, :, :]
            dos = np.zeros(self.nmultiband, dtype=float)
            for  ib in range(nmb):
                if(self.av_mapping is None):
                    dos[ib] = self.elph_dos[self.keep_bands[ib], ism]
                else:
                    ib2 = self.av_mapping[ib]
                    dos[ib2] += self.elph_dos[self.keep_bands[ib], ism]

            for i, iq in enumerate(qpt_id1):
                if np.linalg.norm(tc.qpoints[iq]) < 1.0e-6:
                    continue

                eigs_scaled = self._scaled_eigvecs(tc, iq)

                for ib in range(nmb):
                    if(self.av_mapping is None):
                        ib2 = ib
                    else:
                        ib2 = self.av_mapping[ib]
                    ib1 = self.keep_bands[ib]
                    for jb in range(nmb):
                        if(self.av_mapping is None):
                            jb2 = jb
                        else:
                            jb2 = self.av_mapping[jb]
                        ib1 = self.keep_bands[ib]
                        jb1 = self.keep_bands[jb]
                        V = deformation_potentials[i, ib1, jb1]

                        if mode_mixing == "no" or not anharmonic:
                            mat = np.einsum("ji,jk,ki->i", eigs_scaled, V, eigs_scaled.conj()).real
                            contrib = np.einsum("i,ik,k->k", mat, self.pq_dos[i], 1.0 / self.a2f_omega)/4.0/dos[ib2]
                            self.a2f[ism, ib2, jb2] += contrib
                            self.lambdas[ism, i, :, ib2, jb2] += mat / (2.0 * tc.freqs[iq] ** 2 * dos[ib2])
                            if ism == 0 and ib == 0 and jb == 0:
                                self.pdos += np.sum(self.pq_dos[i], axis=0)
                        else:
                            mat = np.einsum("ji,jk,kl->il", eigs_scaled, V, eigs_scaled.conj()).real

                            contrib = np.einsum("ij,ijk,k->k", mat, self.pq_dos[i], 1.0 / self.a2f_omega)/4.0/dos[ib2]
                            self.a2f[ism, ib2, jb2] += contrib
                            self.lambdas[ism, i, :, ib2, jb2] += np.diag(mat)/ (2.0 * tc.freqs[iq] ** 2 * dos[ib2])
                            if ism == 0 and ib == 0 and jb == 0:
                                self.pdos += np.einsum("iik->k", self.pq_dos[i])

    def write_a2f(self, smear_id=0, a2f_filename="a2f"):
        """
        Write α²F(ω), λ, and phonon DOS to a file.
        """

        if self.a2f is None:
            raise RuntimeError(r"\alpha^2F has not been calculated!")

        tot_lambda = self.get_lambda(smear_id=smear_id)
        tot_a2f = self.get_tot_a2f(smear_id=smear_id)

        with open(a2f_filename, "w", encoding="utf-8") as f:
            smearing = self.elph_smearings[smear_id]
            f.write(f"#  λ = {tot_lambda:.3f}  for electrical smearing {smearing:.6f}")

            if self.multiband:
                f.write(f"  and multiband calculation with {self.nmultiband} electronic bands.\n")
            else:
                f.write("\n")

            if self.multiband:
                f.write("# Frequency (meV)" + "   " + " ".join([f"a2f_{i}{j}".ljust(16) for i in range(self.nmultiband) for j in range(self.nmultiband)]) + "   tot_a2f      Phonon DOS (1/meV)\n")
            else:
                f.write("# Frequency (meV)          alpha2f       Phonon DOS (1/meV)\n")

            for i, omega in enumerate(self.a2f_omega):
                f.write(f"   {omega:.12f}")

                if self.multiband:
                    for ib in range(self.nmultiband):
                        for jb in range(self.nmultiband):
                            value = self.a2f[smear_id, ib, jb, i]
                            f.write(f"   {value:.12f}")

                    f.write(f"   {tot_a2f[i]:.12f}")
                    f.write(f"   {self.pdos[i]:.12f}\n")

                else:
                    f.write(f"   {self.a2f[smear_id, i]:.12f}")
                    f.write(f"   {self.pdos[i]:.12f}\n")


    def get_tot_a2f(self, smear_id = 0):

        if(not self.multiband):
            return self.a2f[smear_id]
        else:
            tot_a2f = np.zeros_like(self.a2f_omega)
            tot_dos = np.sum(self.elph_dos[:,smear_id])
            dos = np.zeros(self.nmultiband, dtype=float)
            for i in range(len(self.keep_bands)):
                if(self.av_mapping is None):
                    ib = i
                else:
                    ib = self.av_mapping[i]
                dos[ib] += self.elph_dos[[self.keep_bands[i]],smear_id]
            for iband in range(self.nmultiband):
                for jband in range(self.nmultiband):
                    tot_a2f += self.a2f[smear_id, iband, jband] * dos[iband]
            tot_a2f /= tot_dos
            return tot_a2f

    def get_lambda(self, smear_id=0, return_matrix = False, print_matrix = False):
        """
        Compute electron-phonon coupling λ.
        For isotropic:    returns scalar λ.
        For multiband:    prints λ_ij matrix and returns total λ.
        """

        if self.a2f is None:
            raise RuntimeError(r"\alpha^2F is not calculated!")

        omega = self.a2f_omega

        if not self.multiband:
            integrand = self.a2f[smear_id] / omega
            return 2.0 * np.trapz(integrand, omega)

        a2f_ij = self.a2f[smear_id]                          # (nb, nb, nom)
        integrand = a2f_ij / omega[np.newaxis, np.newaxis, :]  # broadcast ω
        lambda_matrix = 2.0 * np.trapz(integrand, omega, axis=2)  # (nb, nb)
        if(print_matrix):
            print(r"$\Lambda$ matrix:")
            for row in lambda_matrix:
                print("| " + "  ".join(f"{val:.6f}" for val in row) + " |")

        tot_a2f = self.get_tot_a2f(smear_id)
        total_lambda = 2.0 * np.trapz(tot_a2f / omega, omega)
        if(return_matrix):
            return total_lambda, lambda_matrix
        else:
            return total_lambda


    def get_lambda_from_mode_lambdas(self, smear_id = 0):

        if(self.lambdas is None):
            raise RuntimeError('Mode resolved lambdas not calculated!')
        
        if(self.multiband):

            tot_lambda = 0.0

            band_lambdas = np.average(self.lambdas[smear_id], axis=0, weights=self.elph_weights)
            tot_dos = np.sum(self.elph_dos[:,smear_id])
            for iband in range(self.nmultiband):
                for jband in range(self.nmultiband):
                    tot_lambda += np.sum(band_lambdas[:, iband, jband]) * self.elph_dos[iband, smear_id]
            tot_lambda /= tot_dos

            return tot_lambda
        else:
            band_lambdas = np.average(self.lambdas[smear_id], axis=0, weights=self.elph_weights)
        
            return np.sum(band_lambdas)

    def get_omega_log(self, units = '1/cm', smear_id = 0):

        if(self.multiband):
            lambda_tot = self.get_lambda(smear_id = smear_id)
            tot_a2f = self.get_tot_a2f(smear_id = smear_id)
            omegalog = np.trapz(tot_a2f/self.a2f_omega*np.log(self.a2f_omega), self.a2f_omega)*2.0/lambda_tot
            omegalog = np.exp(omegalog)
        else:
            lambda_tot = self.get_lambda(smear_id = smear_id)
            omegalog = np.trapz(self.a2f[smear_id]/self.a2f_omega*np.log(self.a2f_omega), self.a2f_omega)*2.0/lambda_tot
            omegalog = np.exp(omegalog)

        if(units == '1/cm'):
            omegalog = omegalog*RY_TO_CM/RY_TO_MEV
        elif(units == 'K'):
            omegalog = omegalog*RY_TO_EV/K_B/RY_TO_MEV
        elif(units == 'meV'):
            pass
        elif(units == 'RY'):
            omegalog = omegalog/RY_TO_MEV
        else:
            raise RuntimeError('Units in \"get_omega_log\" function not recognized!')
        
        return omegalog

    def get_omega_squared(self, units = '1/cm', smear_id = 0):

        if(self.multiband):
            lambda_tot = self.get_lambda(smear_id = smear_id)
            tot_a2f = self.get_tot_a2f(smear_id = smear_id)
            omegatwo = np.trapz(tot_a2f*self.a2f_omega, self.a2f_omega)*2.0/lambda_tot
            omegatwo = np.sqrt(omegatwo)
        else:
            lambda_tot = self.get_lambda(smear_id = smear_id)
            omegatwo = np.trapz(self.a2f[smear_id]*self.a2f_omega, self.a2f_omega)*2.0/lambda_tot
            omegatwo = np.sqrt(omegatwo)

        if(units == '1/cm'):
            omegatwo = omegatwo*RY_TO_CM
        elif(units == 'K'):
            omegatwo = omegatwo*RY_TO_EV/K_B
        elif(units == 'meV'):
            pass
        elif(units == 'RY'):
            omegatwo = omegatwo/RY_TO_MEV
        else:
            raise RuntimeError('Units in \"get_omega_squared\" function not recognized!')
        
        return omegatwo

    def get_McMillan_estimate(self, mu_star = 0.15, smear_id = 0):

        lambda_tot = self.get_lambda(smear_id)
        omegalog = self.get_omega_log(units = 'K', smear_id = smear_id)
        return omegalog/1.20*np.exp(-1.04*(1.0 + lambda_tot)/(lambda_tot - mu_star*(1.0 + 0.62*lambda_tot)))

    def get_AllenDynes_estimate(self, mu_star = 0.15, smear_id = 0):

        lambda_tot = self.get_lambda(smear_id = smear_id)
        omegalog = self.get_omega_log(units = 'K', smear_id = smear_id)
        omegatwo = self.get_omega_squared(units = 'K', smear_id = smear_id)

        L1 = 2.46*(1.0 + 3.8*mu_star)
        L2 = 1.82*(1.0 + 6.3*mu_star)*omegatwo/omegalog
        F1 = (1.0 + (lambda_tot/L1)**1.5)**(1.0/3.0)
        F2 = 1.0 + (omegatwo/omegalog - 1.0)*lambda_tot**2/(L2**2 + lambda_tot**2)

        return F1*F2*omegalog*np.exp(-(1.04*(1.0 + lambda_tot))/(lambda_tot - mu_star*(1.0 + 0.62*lambda_tot)))/1.2

    def get_initial_gap_estimate(self, mu_star = 0.15, smear_id = 0):

        if(mu_star > 0.0):
            T_ad = self.get_AllenDynes_estimate(mu_star, smear_id = smear_id)
        else:
            T_ad = self.get_AllenDynes_estimate(0.0, smear_id = smear_id)

        return T_ad*3.52/2.0*K_B * EV_TO_MEV

    def _find_energy_index(self, target_energy: float) -> int:
        """
        Find index j such that wee_energy[j] <= target_energy < wee_energy[j+1],
        clamped to the grid boundaries.
        """
        energies = self.wee_energy
        idx = np.searchsorted(energies, target_energy)

        if idx <= 0:
            return 0
        if idx >= len(energies):
            return len(energies) - 1
        return idx - 1

    def _anderson_pack(self, phi, z, phi_c):
        """
        Flatten (phi, z, phi_c) into a 1D vector for Anderson mixing.
        """
        return np.concatenate([phi.ravel(), z.ravel(), phi_c.ravel()])


    def _anderson_unpack(self, v, phi_shape, z_shape, phi_c_shape):
        """
        Inverse of _anderson_pack: reshape flat vector v back into
        (phi, z, phi_c) with the given shapes.
        """
        n1 = int(np.prod(phi_shape))
        n2 = int(np.prod(z_shape))

        phi_flat = v[:n1]
        z_flat = v[n1:n1 + n2]
        phi_c_flat = v[n1 + n2:]

        phi = phi_flat.reshape(phi_shape)
        z = z_flat.reshape(z_shape)
        phi_c = phi_c_flat.reshape(phi_c_shape)

        return phi, z, phi_c


    def _anderson_step(self, history_x, history_x_star, history_f, phi_shape, z_shape, phi_c_shape):
        """
        Perform one Anderson mixing step given histories:
            history_x       : list of packed previous (phi_old, z_old, phi_c_old)
            history_x_star  : list of packed previous (phi_new, z_new, phi_c_new)
            history_f       : list of packed residuals (phi_new-phi_old, ...)

        Returns
        -------
        (phi, z, phi_c) or None
            - If there is enough history (>=2), returns Anderson-mixed (phi,z,phi_c)
            - If not enough history, returns None (caller should fall back to linear mixing)
        """

        if len(history_f) < 2:
            return None

        F = np.array(history_f).T
        F = F / np.linalg.norm(F, axis=0)

        try:
            alpha = np.linalg.lstsq(F, history_f[-1], rcond=None)[0]
            alpha = alpha / np.sum(alpha)
        except np.linalg.LinAlgError:
            alpha = np.ones(len(history_f)) / float(len(history_f))

        X = np.array(history_x).T
        Xs = np.array(history_x_star).T

        new = X @ alpha
        new_s = Xs @ alpha

        mixed = (1.0 - self.beta) * new + self.beta * new_s

        return self._anderson_unpack(mixed, phi_shape, z_shape, phi_c_shape)

    def solve(self, T_start = 1.0, T_end = 300.0, wcut_ratio = 10.0, mu_star_approx = True, ntemp = 2, mu_star = 0.15, gap_filename = 'SC_GAP', log = False, smear_id = 0, constant_dos = False):

        if(self.multiband):
            return self.solve_multiband(T_start = T_start, T_end = T_end, wcut_ratio = wcut_ratio, mu_star_approx = mu_star_approx, ntemp = ntemp, mu_star = mu_star, gap_filename = gap_filename, log = log, smear_id = smear_id, constant_dos = constant_dos)
        else:
            return self.solve_isotropic(T_start = T_start, T_end = T_end, wcut_ratio = wcut_ratio, mu_star_approx = mu_star_approx, ntemp = ntemp, mu_star = mu_star, gap_filename = gap_filename, log = log, smear_id = smear_id, constant_dos = constant_dos)

    def solve_isotropic(
        self,
        T_start: float = 1.0,
        T_end: float = 300.0,
        wcut_ratio: float = 10.0,
        mu_star_approx: bool = True,
        ntemp: int = 2,
        mu_star: float = 0.15,
        gap_filename: str = "SC_GAP",
        log: bool = False,
        smear_id: int = 0,
        constant_dos: bool = False,
    ):
        """
        Solve isotropic Migdal–Eliashberg equations on a temperature grid.

        - If mu_star_approx is True, uses the μ* approximation.
        - Otherwise uses the full Coulomb kernel (self.wee).
        - When constant_dos is True, EF is assumed to be at 0 energy.
        """

        if (not mu_star_approx) and (self.wee is None):
            raise RuntimeError("You have not supplied Coulomb interaction (wee). "
                "Cannot continue calculation with full Coulomb.")

        w_cut = float(np.max(self.a2f_omega)) * float(wcut_ratio)

        self.temperatures = np.linspace(T_start, T_end, num=ntemp)

        # Storage
        self.iw = []
        self.delta = []
        self.Z = []
        self.indices = []

        if not mu_star_approx:
            self.chi = []
            self.phi_c = []
            self.ef = []
            starting_Ne = get_starting_number_electrons(self.wee_energy, self.wee_dos, 0.0, 0.0)
            if log:
                print(f"[iso] Starting number of electrons: {starting_Ne*2:.2f}")

        gap_is_zero = False
        delta_init = None  # initial gap at lowest T for Tc detection

        for it, T in enumerate(self.temperatures):
            if log:
                print(f"[iso] Solving at T = {T:.2f} K")

            if gap_is_zero:
                w_i = np.array([0.0], dtype=float)
                delta_i = np.array([0.0], dtype=float)
                z_i = np.array([0.0], dtype=float)
                index = 0

                if not mu_star_approx:
                    chi_i = np.array([0.0], dtype=float)
                    phi_c = np.zeros_like(self.wee_energy, dtype=float)
                    ef = 0.0

            else:
                temp_mev = T * K_B * EV_TO_MEV

                # Initial guess for gap at this T
                if it == 0:
                    delta0 = self.get_initial_gap_estimate(mu_star, smear_id=smear_id)
                    delta_init = delta0
                else:
                    delta0 = self.delta[-1][self.indices[-1]]

                # Choose appropriate solver
                if mu_star_approx:
                    w_i, delta_i, z_i, index = self.solve_isotropic_at_T_mu_star(mu_star, w_cut, temp_mev, delta0, log, smear_id=smear_id)
                else:
                    if constant_dos:
                        w_i, delta_i, z_i, phi_c, index = self.solve_isotropic_at_T_constant_dos(w_cut, temp_mev, delta0, starting_Ne, log, smear_id=smear_id)
                        chi_i = None
                        ef = 0.0  # EF assumed at 0 in constant DOS case
                    else:
                        w_i, delta_i, z_i, chi_i, phi_c, index, ef = self.solve_isotropic_at_T(w_cut, temp_mev, delta0, starting_Ne, log, smear_id=smear_id)

            # Store results
            self.iw.append(np.array(w_i, copy=True))
            self.delta.append(np.array(delta_i, copy=True))
            self.Z.append(np.array(z_i, copy=True))
            self.indices.append(index)

            if not mu_star_approx:
                self.phi_c.append(np.array(phi_c, copy=True))

                if constant_dos:
                    # EF assumed to be at 0
                    self.ef.append(0.0)
                    target_energy = 0.0
                else:
                    self.chi.append(np.array(chi_i, copy=True))
                    self.ef.append(ef)
                    target_energy = ef

                e_idx = self._find_energy_index(target_energy)
                curr_gap = (delta_i[index] + phi_c[e_idx]) / z_i[index]
            else:
                curr_gap = delta_i[index]

            if (not gap_is_zero and delta_init is not None and curr_gap < delta_init * 1.0e-3):
                gap_is_zero = True
                self.tc = T
                if log:
                    print(f"[iso] Gap vanished at T ≈ {T:.3f} K (Tc)")

        self.write_gap(gap_filename, mu_star_approx, constant_dos)

    def write_gap(self, gap_filename: str, mu_star_approx: bool, constant_dos: bool):
        """
        Write SC gap Δ(T) to file.
        """

        is_multiband = getattr(self, "multiband", False)

        # Column widths
        T_WIDTH = 12          # width for Temperature
        GAP_WIDTH = 22        # width for each gap column

        with open(gap_filename, "w", encoding="utf-8") as f:
            # ------------------------------------------------------------
            # Header
            # ------------------------------------------------------------
            if is_multiband:
                # e.g. "#   Temperature (K)     Gap_band0        Gap_band1   ..."
                header = "# "
                header += f"{'Temperature (K)':>{T_WIDTH - 3}}"
                for ib in range(self.nmultiband):
                    label = f"Gap_band{ib}"
                    header += f"{label:>{GAP_WIDTH - 7 + 4*ib}}"
                header += "\n"
            else:
                # e.g. "#   Temperature (K)          SC Gap (meV)"
                header = "# "
                header += f"{'Temperature (K)':>{T_WIDTH - 3}}"
                header += f"{'SC Gap (meV)':>{GAP_WIDTH - 4}}\n"

            f.write(header)

            # ------------------------------------------------------------
            # Loop over temperatures and write rows
            # ------------------------------------------------------------
            for i, T in enumerate(self.temperatures):
                idx = self.indices[i]

                # Base: temperature column
                line = f"{T:>{T_WIDTH}.3f}"

                # -------------------------------
                # Isotropic case
                # -------------------------------
                if not is_multiband:
                    delta_i = self.delta[i]
                    z_i = self.Z[i]

                    if delta_i[idx] == 0.0:
                        # Normal state
                        line += f"{0.0:>{GAP_WIDTH}.12f}\n"
                        f.write(line)
                        continue

                    if mu_star_approx:
                        gap_val = delta_i[idx]
                    else:
                        target_energy = 0.0 if constant_dos else self.ef[i]
                        e_idx = self._find_energy_index(target_energy)
                        gap_val = (delta_i[idx] + self.phi_c[i][e_idx]) / z_i[idx]

                    line += f"{gap_val:>{GAP_WIDTH}.12f}\n"
                    f.write(line)
                    continue

                # -------------------------------
                # Multiband case
                # -------------------------------
                delta_i = self.delta[i]   # shape: (nband, Nω)
                z_i = self.Z[i]           # shape: (nband, Nω)
                if(not mu_star_approx):
                    phi_i = self.phi_c[i]     # shape: (nband, Ne)

                if constant_dos:
                    target_energy = 0.0
                else:
                    target_energy = self.ef[i]

                if mu_star_approx:
                    e_idx = None
                else:
                    e_idx = self._find_energy_index(target_energy)

                for ib in range(self.nmultiband):
                    dval = delta_i[ib][idx]
                    zval = z_i[ib][idx]

                    if dval == 0.0:
                        line += f"{0.0:>{GAP_WIDTH}.12f}"
                        continue

                    if mu_star_approx:
                        gap_val = dval
                    else:
                        gap_val = (dval + phi_i[ib][e_idx]) / zval

                    line += f"{gap_val:>{GAP_WIDTH}.12f}"

                line += "\n"
                f.write(line)

    def get_matsubara_frequencies(self, w_cut, T):

        n_max = int((w_cut / (np.pi*T) - 1.0) / 2.0)
        n = np.arange(0, n_max + 1)
        w_pos = (2*n + 1) * np.pi * T
        w = np.concatenate([-w_pos, w_pos]).tolist() 
        w.sort()
        w = np.array(w)

        return w

    def get_matsubara_frequencies_only_positive(self, w_cut, T):

        n_max = int((w_cut / (np.pi*T) - 1.0) / 2.0)
        n = np.arange(0, n_max)
        w = (2.0*n + 1.0) * np.pi * T

        return w

    def generate_Lnm_matrix(self, w, smear_id = 0):

        lnm = np.zeros((len(w), len(w)))
        for j in range(len(w)):
            lnm[0,j] = np.trapz(2.0 * self.a2f_omega * self.a2f[smear_id] / ((w[0] - w[j])**2 + self.a2f_omega**2), self.a2f_omega)
        for i in range(len(w)):
            for j in range(i, len(w)):
                lnm[i,j] = lnm[0, abs(j-i)]
        lnm = lnm + lnm.T - np.diag(np.diag(lnm)) # Subtracts the diagonal that we added and we should not have !

        return lnm

    def generate_Lnm_matrix_multiband(self, w, smear_id = 0):

        lnm = np.zeros((self.nmultiband, self.nmultiband, len(w), len(w)))
        for j in range(len(w)):
            lnm[:,:,0,j] = np.trapz(2.0 * self.a2f_omega * self.a2f[smear_id,:,:] / ((w[0] - w[j])**2 + self.a2f_omega**2), self.a2f_omega)
        for i in range(len(w)):
            for j in range(i, len(w)):
                lnm[:,:,i,j] = lnm[:,:,0, abs(j-i)]
        for iband in range(self.nmultiband):
            for jband in range(self.nmultiband):
                lnm[iband, jband] = lnm[iband, jband] + lnm[iband, jband].T - np.diag(np.diag(lnm[iband, jband]))

        return lnm

    def solve_isotropic_at_T_mu_star(self, mu_star, w_cut, T, delta0, log=False, smear_id=0):
        """
        Solve isotropic Eliashberg equations at temperature T
        using the mu* approximation.
        """

        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_full = self.get_matsubara_frequencies(w_cut, T)

        n_max = len(w)          # number of positive frequencies
        index0 = 0              # index of w0 = πT (Matsubara fundamental)

        lnm = self.generate_Lnm_matrix(w_full, smear_id=smear_id)

        lp = np.zeros((n_max, n_max), dtype=float)
        lm = np.zeros_like(lp)

        if not (w_full[n_max] < 0.0 and w_full[n_max + 1] > 0.0):
            raise RuntimeError("Wrong choice of w0 or Matsubara grid inconsistent.")

        offset = n_max + 1
        for i in range(n_max):
            for j in range(n_max):
                lp[i, j] = lnm[offset + i, n_max - j] + lnm[offset + i, j + offset]
                lm[i, j] = -(lnm[offset + i, n_max - j] - lnm[offset + i, j + offset])

        lp_mu = lp - 2.0 * mu_star

        delta_old = np.zeros(n_max)
        delta_old[index0] = delta0

        Z_old = np.ones(n_max)
        Z_old[index0] = 2.0

        iteration = 1
        diff = 1.0

        while diff > self.thr and iteration < self.max_iter:

            delta_prev = delta_old.copy()
            Z_prev = Z_old.copy()

            denom = np.sqrt(w ** 2 + delta_old ** 2)

            v1 = w / denom           # w / sqrt(w^2 + Δ^2)
            v2 = delta_old / denom   # Δ / sqrt(w^2 + Δ^2)

            Z_new = 1.0 + np.dot(lm, v1) * np.pi * T / w
            delta_new = np.dot(lp_mu, v2) * np.pi * T / Z_prev

            delta_old = self.mix * delta_new + (1.0 - self.mix) * delta_prev
            Z_old = self.mix * Z_new + (1.0 - self.mix) * Z_prev

            diff = np.linalg.norm(delta_old - delta_prev) / np.linalg.norm(delta_prev)

            iteration += 1

        if iteration >= self.max_iter:
            if log:
                print("[mu*] WARNING: calculation did not converge.")
        else:
            if log:
                print(f"[mu*] Converged in {iteration} iterations. Final Δ-diff = {diff:.3e}")

        return w, delta_old, Z_old, index0

    def solve_isotropic_at_T_constant_dos(self, w_cut, T, delta0, Ne, log=False, smear_id=0):
        """
        Solve isotropic Eliashberg equations at temperature T using:
            - constant electronic DOS (EF = 0)
            - full Coulomb kernel
            - Anderson or simple mixing
        """

        mix = self.mix

        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_full = self.get_matsubara_frequencies(w_cut, T)
        n_max = len(w)
        index0 = 0

        lnm = self.generate_Lnm_matrix(w_full, smear_id=smear_id)

        phi1 = np.zeros(n_max)
        phi1[index0] = delta0

        phi_c = np.zeros_like(self.wee_dos)

        z1 = np.ones(n_max) + 1.0

        ef = 0.0
        ief = np.argmin(np.abs(self.wee_energy))
        Nef = self.wee_dos[ief]
        if Nef == 0.0:
            raise RuntimeError("Density of states at EF is zero!")

        phi_c[ief] = -1e-5  # initial guess on Coulomb correction

        iteration = 1
        diff = 1.0

        lp = np.zeros((n_max, n_max))
        lm = np.zeros_like(lp)
        offset = n_max + 1

        for i in range(n_max):
            for j in range(n_max):
                A = lnm[offset + i, offset + j]       # L(n+i, n+j)
                B = lnm[offset + i, n_max - j]        # L(n+i, -j)
                lp[i, j] = A + B
                lm[i, j] = A - B

        mu_star_init = 0.0
        w0, delta_init, z0, idx_tmp = self.solve_isotropic_at_T_mu_star(mu_star_init, w_cut, T, delta0, log=log, smear_id=smear_id)
        if not np.all(w0 == w):
            raise RuntimeError("Inconsistent Matsubara grid in μ* bootstrap!")
        phi1 = delta_init * z0
        z1 = z0.copy()
        theta = np.maximum((w * z1)[None, :] ** 2 + self.wee_energy[:, None] ** 2 + phi1 ** 2, 1e-12)
        term2 = T * np.sum(phi1[None, :] / theta, axis=1)
        dE = self.wee_energy[1] - self.wee_energy[0]
        phi_c = -2.0 * np.einsum("i,ji->j", self.wee_dos * term2, self.wee) * dE

        if self.aa:
            history_x = []
            history_x_star = []
            history_f = []

        while True:
            if abs(phi1[index0] / z1[index0]) <= 1e-6 and iteration > int(1.1 * self.starting_Coulomb_weight):
                break

            if (diff <= self.thr and iteration > 1) and (iteration >= int(1.1 * self.starting_Coulomb_weight)):
                break

            if iteration >= self.max_iter:
                break
            
            coulomb_weight = min(1.0, round(float(iteration) / self.starting_Coulomb_weight + 0.05, 1))
            prev_diff = diff

            phi2 = phi1.copy()
            z2 = z1.copy()
            phi_c2 = phi_c.copy()

            tot_phi = phi2[None, :] + phi_c2[:, None]
            theta = (w * z2)[None, :] ** 2 + self.wee_energy[:, None] ** 2 + tot_phi ** 2

            denom_ief = np.sqrt(theta[ief])
            z1 = 1.0 + np.pi * T / w * np.dot(lm, (w * z2) / denom_ief)

            phi1 = np.pi * T * np.dot(lp, tot_phi[ief] / denom_ief)

            sqrt_term = np.sqrt(phi_c2 ** 2 + self.wee_energy ** 2)
            term1 = phi_c2 / 2.0 * np.tanh(sqrt_term / (2 * T)) / np.maximum(sqrt_term, 1e-12)
            term2 = 2.0 * T * np.sum(tot_phi / theta - phi_c2[:, None] / ((w[None, :] ** 2) + (sqrt_term[:, None] ** 2)), axis=1)

            phi_c = -coulomb_weight * np.einsum("i,ij->j", self.wee_dos * (term1 + term2), self.wee) * dE

            diff = np.linalg.norm(phi1 - phi2) / np.linalg.norm(phi2)

            if self.aa:
                history_x.append(self._anderson_pack(phi2, z2, phi_c2))
                history_x_star.append(self._anderson_pack(phi1, z1, phi_c))
                history_f.append(self._anderson_pack(phi1 - phi2, z1 - z2, phi_c - phi_c2))

                if len(history_f) > self.anderson:
                    history_x.pop(0)
                    history_x_star.pop(0)
                    history_f.pop(0)

                anderson_result = self._anderson_step(history_x, history_x_star, history_f, phi1.shape, z1.shape, phi_c.shape)

                if anderson_result is not None:
                    phi1, z1, phi_c = anderson_result
                else:
                    if diff < prev_diff and iteration > 1:
                        mix = min(1.0, mix * 1.1)
                    elif diff > prev_diff:
                        mix = max(0.5, mix * 0.9)
                    phi1 = (1.0 - mix) * phi2 + mix * phi1
                    z1 = (1.0 - mix) * z2 + mix * z1
                    phi_c = (1.0 - mix) * phi_c2 + mix * phi_c
            else:
                # simple mixing only
                if diff < prev_diff:
                    mix = min(1.0, mix * 1.1)
                else:
                    mix = max(0.5, mix * 0.9)

                phi1 = (1 - mix) * phi2 + mix * phi1
                z1 = (1 - mix) * z2 + mix * z1
                phi_c = (1 - mix) * phi_c2 + mix * phi_c

            iteration += 1

        if log:
            if iteration >= self.max_iter:
                print("[cdos] WARNING: calculation did not converge.")
            print(f"[cdos] Finished in {iteration} iterations.")

        return w, phi1, z1, phi_c, index0

    def solve_isotropic_at_T(self, w_cut, T, delta0, Ne, log=False, smear_id=0):
        """
        Solve isotropic Eliashberg equations at temperature T with full
        Coulomb interaction and energy-dependent DOS.
        """

        coulomb_weight = self.starting_Coulomb_weight
        mix = self.mix

        diff = 1.0

        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_full = self.get_matsubara_frequencies(w_cut, T)
        lnm = self.generate_Lnm_matrix(w_full, smear_id=smear_id)

        index = 0
        n_max = len(w)

        phi1 = np.zeros(n_max, dtype=float)
        phi1[index] = delta0

        phi_c = np.zeros(len(self.wee_dos), dtype=float)

        z1 = np.ones(n_max, dtype=float)
        z1[index] = 2.0

        chi1 = np.zeros_like(phi1)
        iteration = 1

        lp = np.zeros((n_max, n_max), dtype=float)
        lm = np.zeros_like(lp)
        for i in range(n_max):
            for j in range(n_max):
                lp[i, j] = lnm[i + n_max + 1, n_max - j] + lnm[i + n_max + 1, j + n_max + 1]
                lm[i, j] = -(lnm[i + n_max + 1, n_max - j] - lnm[i + n_max + 1, j + n_max + 1])

        theta = np.zeros((len(self.wee_dos), n_max))
        en_shift = np.zeros_like(theta)
        tot_phi = np.zeros_like(theta)

        ef = 0.0
        de_wee_energy = self.wee_energy[1] - self.wee_energy[0]

        ief = np.argmin(np.abs(self.wee_energy))
        Nef = self.wee_dos[ief]
        if Nef == 0.0:
            raise RuntimeError("Density of states at the Fermi level is 0.0!")

        phi_c[ief] = -1.0e-5

        if self.aa:
            history_x = []
            history_x_star = []
            history_f = []

        mu_star = self.wee_dos[ief] * self.wee[ief, ief] / 5.0
        w0, delta1, z0, index0 = self.solve_isotropic_at_T_mu_star(mu_star, w_cut, T, delta0, log=log, smear_id=smear_id)

        if np.all(w0 == w):
            phi1 = delta1 * z0
            z1 = z0

            theta = np.maximum((w * z1)[np.newaxis, :]**2 + self.wee_energy[:, np.newaxis]**2 + phi1**2, 1.0e-12)
            sqrt_term = np.maximum(np.sqrt(phi_c**2 + self.wee_energy**2), 1.0e-12)
            term2 = T * np.sum(phi1[np.newaxis, :] / theta, axis=1)
            phi_c = -2.0 * np.einsum("i,ji->j", self.wee_dos * term2, self.wee) * de_wee_energy
        else:
            raise RuntimeError("This should not happen!")

        while ((diff > self.thr and iteration < self.max_iter and abs(phi1[index] / z1[index]) > 1.0e-6) or iteration < int(self.starting_Coulomb_weight) + 1):
            # Coulomb ramp (unchanged)
            coulomb_weight = min(1.0, round(float(iteration) / self.starting_Coulomb_weight + 0.05, 2))
            prev_diff = diff

            phi2 = phi1.copy()
            z2 = z1.copy()
            phi_c2 = phi_c.copy()
            chi2 = chi1.copy()
            ef2 = ef  

            en_shift = self.wee_energy[:, np.newaxis] - ef + chi1[np.newaxis, :]
            tot_phi = phi1[np.newaxis, :] + phi_c[:, np.newaxis]

            theta = (w * z1)[np.newaxis, :]**2 + en_shift**2 + tot_phi**2

            z1 = 1.0 + T / w / Nef * np.einsum("k,kj,ij->i", self.wee_dos, (w * z2)[np.newaxis, :] / theta, lm)* de_wee_energy

            chi1 = -T / Nef * np.einsum("k,kj,ij->i",self.wee_dos, en_shift / theta, lp)* de_wee_energy

            phi1 = T / Nef * np.einsum("k,kj,ij->i", self.wee_dos, tot_phi / theta, lp) * de_wee_energy

            sqrt_term = np.sqrt(phi_c**2 + (self.wee_energy - ef)**2)
            term1 = phi_c / 2.0 * np.tanh(sqrt_term / (2.0 * T)) / sqrt_term
            term2 = 2.0 * T * np.sum(tot_phi / theta - phi_c[:, np.newaxis] / (w[np.newaxis, :]**2 + sqrt_term[:, np.newaxis]**2), axis=1)
            phi_c = -coulomb_weight * np.einsum("i,ij->j", self.wee_dos * (term1 + term2), self.wee) * de_wee_energy

            ef = brentq(fermi_root, -1.0 * self.shift_guess + ef, self.shift_guess + ef, args=(self.wee_energy, self.wee_dos, T, w, chi2, theta, Ne), xtol=1e-6)

            diff = np.linalg.norm(phi1 - phi2) / np.linalg.norm(phi2)

            if self.aa:
                history_x.append(self._anderson_pack(phi2, z2, phi_c2))
                history_x_star.append(self._anderson_pack(phi1, z1, phi_c))
                history_f.append(self._anderson_pack(phi1 - phi2, z1 - z2, phi_c - phi_c2))

                if len(history_f) > self.anderson:
                    history_x.pop(0)
                    history_x_star.pop(0)
                    history_f.pop(0)

                if len(history_f) >= self.anderson:
                    result = self._anderson_step(history_x, history_x_star, history_f, phi1.shape, z1.shape, phi_c.shape)

                    if result is not None:
                        phi1, z1, phi_c = result
                    else:
                        # fallback linear mixing (should be rare)
                        if diff < prev_diff:
                            mix = max(1.0, mix * 1.1)
                        elif diff > prev_diff:
                            mix = min(0.1, mix / 2.0)
                        phi1 = (1.0 - mix) * phi2 + mix * phi1
                        z1 = (1.0 - mix) * z2 + mix * z1
                        phi_c = (1.0 - mix) * phi_c2 + mix * phi_c
                else:
                    # Same adaptive mixing as original
                    if diff < prev_diff:
                        mix = max(1.0, mix * 1.1)
                    elif diff > prev_diff:
                        mix = min(0.1, mix / 2.0)
                    phi1 = (1.0 - mix) * phi2 + mix * phi1
                    z1 = (1.0 - mix) * z2 + mix * z1
                    phi_c = (1.0 - mix) * phi_c2 + mix * phi_c
            else:
                # No Anderson: just adaptive linear mixing
                if diff < prev_diff and iteration > 1:
                    mix = min(1.0, mix * 1.1)
                elif diff > prev_diff:
                    mix = max(0.5, mix * 0.9)
                phi1 = (1.0 - mix) * phi2 + mix * phi1
                z1 = (1.0 - mix) * z2 + mix * z1
                phi_c = (1.0 - mix) * phi_c2 + mix * phi_c

            iteration += 1

        if iteration >= self.max_iter and log:
            print("Calculation did not converge!")

        return w, phi1, z1, chi1, phi_c, index, ef

    def solve_multiband(self, T_start=1.0, T_end=300.0, wcut_ratio=10.0, mu_star_approx=True, ntemp=2, mu_star=0.15, gap_filename="SC_GAP", log=False, smear_id=0, constant_dos=False):

        if not mu_star_approx and self.wee is None:
            raise RuntimeError("You have not supplied Coulomb interaction. Cannot continue calculation!")

        self.temperatures = np.linspace(T_start, T_end, num=ntemp)
        w_cut = max(self.a2f_omega) * wcut_ratio

        self.iw = []
        self.delta = []
        self.Z = []
        self.indices = []

        if not mu_star_approx:
            self.chi = []
            self.phi_c = []
            self.ef = []

            starting_Ne = get_starting_number_electrons(self.wee_energy, self.wee_dos, 0.0, 0.0)
            print(f"Starting number of electrons: {starting_Ne*2:.2f}")

        gap_is_zero = False
        mu_star_matrix = self.get_mu_matrix(mu_star, smear_id=smear_id)

        for it, Tcurr in enumerate(self.temperatures):

            if log:
                print(f"Calculating for temperature: {Tcurr:.2f} K!")

            # Early exit condition
            if gap_is_zero:
                w_i = np.zeros((self.nmultiband, 1))
                delta_i = np.zeros_like(w_i)
                z_i = np.zeros_like(w_i)
                index = 0

                if not mu_star_approx:
                    chi_i = np.zeros_like(w_i)
                    phi_c = np.zeros((self.nmultiband, len(self.wee_energy)))
                    ef = 0.0
            else:
                # Convert T → meV
                T_meV = Tcurr * K_B * EV_TO_MEV

                if it == 0:
                    delta0 = self.get_initial_gap_estimate(mu_star, smear_id=smear_id)
                    delta_init = delta0

                    if mu_star_approx:
                        w_i, delta_i, z_i, index = self.solve_multiband_at_T_mu_star(mu_star_matrix, w_cut, T_meV, delta0, log, smear_id=smear_id)
                    else:
                        w_i, delta_i, z_i, index = self.solve_multiband_at_T_mu_star(mu_star_matrix, w_cut, T_meV, delta0, log, smear_id=smear_id)
                        if constant_dos:
                            w_i, delta_i, z_i, phi_c, index = self.solve_multiband_at_T_constant_dos(w_cut, T_meV, delta_i[:, index], z_i[:, index], starting_Ne, log, smear_id=smear_id)
                        else:
                            w_i, delta_i, z_i, chi_i, phi_c, index, ef = self.solve_multiband_at_T(w_cut, T_meV, delta0, z_i[:, index], 0.0, starting_Ne, log, smear_id=smear_id)
                else:
                    delta_prev = self.delta[-1][:, self.indices[-1]]
                    z_prev = self.Z[-1][:, self.indices[-1]]

                    if mu_star_approx:
                        w_i, delta_i, z_i, index = self.solve_multiband_at_T_mu_star(mu_star_matrix, w_cut, T_meV, delta_prev, log, smear_id=smear_id)
                    else:
                        if constant_dos:
                            w_i, delta_i, z_i, phi_c, index = self.solve_multiband_at_T_constant_dos(w_cut, T_meV, delta_prev, z_prev, starting_Ne, log, smear_id=smear_id)
                        else:
                            w_i, delta_i, z_i, chi_i, phi_c, index, ef = self.solve_multiband_at_T(w_cut, T_meV, delta_prev, z_prev, self.ef[-1], starting_Ne, log, smear_id=smear_id)

            self.iw.append(w_i)
            self.delta.append(delta_i)
            self.Z.append(z_i)
            self.indices.append(index)

            if not mu_star_approx:
                if not constant_dos:
                    self.chi.append(chi_i)
                    self.ef.append(ef)
                self.phi_c.append(phi_c)

            if mu_star_approx:
                curr_gap = delta_i[:, index]
            else:
                if constant_dos:
                    target_energy = 0.0
                else:
                    target_energy = ef

                energy_index = self._find_energy_index(target_energy)
                curr_gap = (delta_i[:, index] + phi_c[:, energy_index]) / z_i[:, index]

            if (not gap_is_zero) and np.all(curr_gap < delta_init * 1e-3):
                gap_is_zero = True
                self.tc = Tcurr

        self.write_gap(gap_filename=gap_filename, mu_star_approx=mu_star_approx, constant_dos=constant_dos)

    def solve_multiband_at_T_mu_star(self, mu_star, w_cut, T, delta0, log=False, smear_id=0):
        """
        Multiband Eliashberg solver at temperature T using a μ* matrix.
        """

        nband = self.nmultiband

        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_inv = 1.0 / w
        w_full = self.get_matsubara_frequencies(w_cut, T)

        n_max = len(w)
        index = 0

        delta1 = np.zeros((nband, n_max), dtype=float)
        delta1[:, index] = delta0

        z1 = np.ones_like(delta1)
        z1[:, index] = 2.0

        diff = np.ones(nband, dtype=float)
        iteration = 1

        lnm = self.generate_Lnm_matrix_multiband(w_full, smear_id=smear_id)

        if not (w_full[n_max + 1] > 0.0 and w_full[n_max] < 0.0):
            raise RuntimeError("Wrong choice of w0")

        lp = np.zeros((nband, nband, n_max, n_max), dtype=float)
        lm = np.zeros_like(lp)
        offset = n_max + 1

        for i in range(n_max):
            for j in range(n_max):
                lp[:, :, i, j] = lnm[:, :, i + offset, j + offset] + lnm[:, :, i + offset, n_max - j]
                lm[:, :, i, j] = -1.0 * (lnm[:, :, i + offset, n_max - j] - lnm[:, :, i + offset, j + offset])

        lp_mu = lp - 2.0 * mu_star[:, :, np.newaxis, np.newaxis]

        v1 = np.zeros_like(delta1)
        v2 = np.zeros_like(delta1)

        while np.all(diff > self.thr) and iteration < self.max_iter:

            delta2 = delta1.copy()
            z2 = z1.copy()

            denom = np.sqrt(w**2 + delta1**2)          # (nband, n_max)
            v1[:] = w / denom
            v2[:] = delta1 / denom

            z1 = 1.0 + np.einsum("ijkl,jl,k->ik", lm, v1, w_inv) * np.pi * T

            delta1 = np.einsum("ijkl,jl,ik->ik", lp_mu, v2, 1.0 / z2) * np.pi * T

            delta1 = self.mix * delta1 + (1.0 - self.mix) * delta2
            z1 = self.mix * z1 + (1.0 - self.mix) * z2

            for b in range(nband):
                diff[b] = np.linalg.norm(delta1[b] - delta2[b]) / np.linalg.norm(delta2[b])

            iteration += 1

        if iteration >= self.max_iter:
            if log:
                print("Calculation did not converge!")
        else:
            if log:
                print(f"Converged in {iteration} iterations. The average difference was: {np.average(diff):.3e}")

        return w, delta1, z1, index

    def solve_multiband_at_T_constant_dos(self, w_cut, T, delta0, z0, Ne, log=False, smear_id=0):
        """
        Multiband isotropic Eliashberg solver at temperature T with:
          - constant electronic DOS (EF = 0)
          - full Coulomb interaction (multiband)
        Uses Anderson mixing if self.aa is True.
        """

        nband = self.nmultiband

        coulomb_weight = self.starting_Coulomb_weight
        mix = self.mix

        diff = np.ones(nband)

        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_inv = 1.0 / w
        w_full = self.get_matsubara_frequencies(w_cut, T)

        lnm = self.generate_Lnm_matrix_multiband(w_full, smear_id=smear_id)
        index = 0
        n_max = len(w)

        phi1 = np.zeros((nband, n_max), dtype=float)
        phi1[:, index] = delta0 * z0  # φ = Δ * Z

        phi_c = np.zeros((nband, self.wee_dos.shape[-1]), dtype=float)

        # Initial Z
        z1 = np.ones((nband, n_max), dtype=float)
        z1[:, index] = z0
        iteration = 1

        lp = np.zeros((nband, nband, n_max, n_max), dtype=float)
        lm = np.zeros_like(lp)
        offset = n_max + 1
        for i in range(n_max):
            for j in range(n_max):
                lp[:, :, i, j] = lnm[:, :, i + offset, n_max - j] + lnm[:, :, i + offset, j + offset]
                lm[:, :, i, j] = -1.0 * (lnm[:, :, i + offset, n_max - j] - lnm[:, :, i + offset, j + offset])

        theta = np.zeros((nband, len(self.wee_dos[0]), n_max))

        ef = 0.0
        de_wee_energy = self.wee_energy[1] - self.wee_energy[0]

        ief = np.argmin(np.abs(self.wee_energy))
        Nef = self.wee_dos[:, ief]
        if np.any(Nef == 0.0):
            raise RuntimeError("Density of states of some bands at the Fermi level is 0.0!")

        phi_c[:, ief] = -1.0e-5

        if self.aa:
            history_x = []
            history_x_star = []
            history_f = []

        while ((np.any(diff > self.thr) and iteration < self.max_iter and np.any(np.abs(phi1[:, index] / z1[:, index]) > 1.0e-6))
               or iteration < int(self.starting_Coulomb_weight * 1.1)):

            coulomb_weight = min(1.0, round(float(iteration) / self.starting_Coulomb_weight + 0.05, 2))
            prev_diff = diff.copy()

            phi2 = phi1.copy()
            z2 = z1.copy()
            phi_c2 = phi_c.copy()

            tot_phi = phi2[:, np.newaxis, :] + phi_c2[:, :, np.newaxis]
            theta = (w * z2)[:, np.newaxis, :]**2 + self.wee_energy[np.newaxis, :, np.newaxis]**2 + tot_phi**2

            z1 = 1.0 + w_inv * np.pi * T * np.einsum("kj,lkij->li", w[np.newaxis, :] * z2 / np.sqrt(theta[:, ief, :]), lm)

            phi1 = T * np.pi * np.einsum("kj,lkij->li", tot_phi[:, ief, :] / np.sqrt(theta[:, ief, :]), lp)

            sqrt_term = np.sqrt(phi_c2**2 + self.wee_energy[np.newaxis, :]**2)
            term1 = phi_c2 / 2.0 * np.tanh(sqrt_term / (2.0 * T)) / np.maximum(sqrt_term, 1.0e-12)
            term2 = 2.0 * T * np.sum(tot_phi / theta - phi_c2[:, :, np.newaxis] / ((w[np.newaxis, np.newaxis, :])**2 + (sqrt_term[:, :, np.newaxis])**2), axis=2)
            phi_c = -coulomb_weight * np.einsum("kj,lkij->li", self.wee_dos * (term1 + term2), self.wee) * de_wee_energy

            for iband in range(nband):
                diff[iband] = np.linalg.norm(phi1[iband] - phi2[iband]) / np.linalg.norm(phi2[iband])

            if self.aa:
                history_x.append(self._anderson_pack(phi2, z2, phi_c2))
                history_x_star.append(self._anderson_pack(phi1, z1, phi_c))
                history_f.append(self._anderson_pack(phi1 - phi2, z1 - z2, phi_c - phi_c2))

                if len(history_f) > self.anderson:
                    history_x.pop(0)
                    history_x_star.pop(0)
                    history_f.pop(0)

                if len(history_f) >= 2:
                    result = self._anderson_step(history_x, history_x_star, history_f, phi1.shape, z1.shape, phi_c.shape)
                    if result is not None:
                        phi1, z1, phi_c = result
            else:
                if np.all(diff < prev_diff) and iteration > 1:
                    mix = min(1.0, mix * 1.1)
                elif np.any(diff > prev_diff):
                    mix = max(0.5, mix * 0.9)
                phi1 = (1.0 - mix) * phi2 + mix * phi1
                z1 = (1.0 - mix) * z2 + mix * z1
                phi_c = (1.0 - mix) * phi_c2 + mix * phi_c

            iteration += 1

        if(log):
            print("Finished in " + str(iteration) + " iterations.")
        if iteration >= self.max_iter and log:
            print("Calculation did not converge!")

        return w, phi1, z1, phi_c, index

    def solve_multiband_at_T(self, w_cut, T, delta0, z0, ef, Ne, log=False, smear_id=0):
        """
        Multiband Eliashberg solver at temperature T with:
          - energy-dependent DOS
          - full Coulomb interaction (multiband)
        Uses Anderson mixing if self.aa is True.
        """

        nband = self.nmultiband

        coulomb_weight = self.starting_Coulomb_weight
        mix = self.mix

        diff = np.ones(nband)

        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_inv = 1.0 / w
        w_full = self.get_matsubara_frequencies(w_cut, T)

        lnm = self.generate_Lnm_matrix_multiband(w_full, smear_id=smear_id)
        index = 0
        n_max = len(w)

        phi1 = np.zeros((nband, n_max), dtype=float)
        phi1[:, index] = delta0 * z0

        phi_c = np.zeros_like(self.wee_dos, dtype=float)  # shape (nband, N_E)
        z1 = np.ones_like(phi1, dtype=float)
        z1[:, index] = z0

        chi1 = np.zeros_like(phi1)
        iteration = 1

        lp = np.zeros((nband, nband, n_max, n_max), dtype=float)
        lm = np.zeros_like(lp)
        offset = n_max + 1
        for i in range(n_max):
            for j in range(n_max):
                lp[:, :, i, j] = lnm[:, :, i + offset, n_max - j] + lnm[:, :, i + offset, j + offset]
                lm[:, :, i, j] = -1.0 * (lnm[:, :, i + offset, n_max - j] - lnm[:, :, i + offset, j + offset])

        theta = np.zeros((nband, len(self.wee_dos[0]), n_max), dtype=float)
        en_shift = np.zeros_like(theta)
        tot_phi = np.zeros_like(theta)

        de_wee_energy = self.wee_energy[1] - self.wee_energy[0]

        ief = np.argmin(np.abs(self.wee_energy))
        Nef = self.wee_dos[:, ief]
        if np.any(Nef == 0.0):
            raise RuntimeError("Density of states at the Fermi level is 0.0!")
        scaled_wee_dos = self.wee_dos / Nef[:, np.newaxis]

        phi_c[:, ief] = -1.0e-5

        if self.aa:
            history_x = []
            history_x_star = []
            history_f = []

        while ((np.all(diff > self.thr) and iteration < self.max_iter and np.all(np.abs(phi1[:, index] / z1[:, index]) > 1.0e-6))
               or iteration < int(self.starting_Coulomb_weight) + 1):

            coulomb_weight = min(1.0, round(float(iteration) / self.starting_Coulomb_weight + 0.05, 2))
            prev_diff = diff.copy()

            phi2 = phi1.copy()
            z2 = z1.copy()
            phi_c2 = phi_c.copy()
            chi2 = chi1.copy()
            ef2 = ef 

            en_shift = self.wee_energy[np.newaxis, :, np.newaxis] - ef + chi2[:, np.newaxis, :]
            tot_phi = phi2[:, np.newaxis, :] + phi_c2[:, :, np.newaxis]
            theta = (w * z2)[:, np.newaxis, :]**2 + en_shift**2 + tot_phi**2

            z1 = 1.0 + w_inv * T * np.einsum("lk,lkj,mlij->mi", scaled_wee_dos, (w * z2)[:, np.newaxis, :] / theta, lm) * de_wee_energy

            chi1 = -T * np.einsum("lk,lkj,mlij->mi", scaled_wee_dos, en_shift / theta, lp) * de_wee_energy

            phi1 = T * np.einsum("lk,lkj,mlij->mi", scaled_wee_dos, tot_phi / theta, lp) * de_wee_energy

            sqrt_term = np.sqrt(phi_c**2 + (self.wee_energy - ef)[np.newaxis, :]**2)
            term1 = phi_c / 2.0 * np.tanh(sqrt_term / (2.0 * T)) / sqrt_term
            term2 = 2.0 * T * np.sum(tot_phi / theta - phi_c2[:, :, np.newaxis] / (w[np.newaxis, np.newaxis, :]**2 + sqrt_term[:, :, np.newaxis]**2), axis=2)

            phi_c = -coulomb_weight * np.einsum("li,mlji->mj", self.wee_dos * (term1 + term2), self.wee) * de_wee_energy

            ef = brentq(fermi_root, -1.0 * self.shift_guess + ef, self.shift_guess + ef, args=(self.wee_energy, self.wee_dos, T, w, chi2, theta, Ne), xtol=1e-6)

            for iband in range(nband):
                diff[iband] = np.linalg.norm(phi1[iband] - phi2[iband]) / np.linalg.norm(phi2[iband])

            if self.aa:
                history_x.append(self._anderson_pack(phi2, z2, phi_c2))
                history_x_star.append(self._anderson_pack(phi1, z1, phi_c))
                history_f.append(self._anderson_pack(phi1 - phi2, z1 - z2, phi_c - phi_c2))

                if len(history_f) > self.anderson:
                    history_x.pop(0)
                    history_x_star.pop(0)
                    history_f.pop(0)

                if len(history_f) >= 2:
                    result = self._anderson_step(history_x, history_x_star, history_f, phi1.shape, z1.shape, phi_c.shape)
                    if result is not None:
                        phi1, z1, phi_c = result
            else:
                if np.all(diff < prev_diff) and iteration > 1:
                    mix = min(1.0, mix * 1.1)
                elif np.all(diff > prev_diff):
                    mix = max(0.5, mix * 0.9)
                phi1 = (1.0 - mix) * phi2 + mix * phi1
                z1 = (1.0 - mix) * z2 + mix * z1
                phi_c = (1.0 - mix) * phi_c2 + mix * phi_c

            iteration += 1

        if iteration >= self.max_iter and log:
            print("Calculation did not converge!")

        return w, phi1, z1, chi1, phi_c, index, ef

    def get_mu_matrix(self, mu, off_diag_scale = 0.1, way = 1, smear_id = 0):

        mu_matrix = np.zeros((self.nmultiband, self.nmultiband))
        dos = self.elph_dos[:, smear_id]
        if(way == 1):
            mu_matrix[:,:] = mu
            for iband in range(self.nmultiband):
                for jband in range(iband + 1, self.nmultiband):
                    mu_matrix[iband,jband] = mu*off_diag_scale
                    mu_matrix[jband,iband] = mu*off_diag_scale
        elif(way == 2):
            k = 0.0
            tot_dos = 0.0
            for iband in range(self.nmultiband):
                for jband in range(iband+1, self.nmultiband):
                    k += dos[iband]
                tot_dos += dos[iband] 
            k = 2.0*off_diag_scale*k/tot_dos + 1.0
            k = mu/k
            for iband in range(self.nmultiband):
                mu_matrix[iband, iband] = k
                for jband in range(iband + 1, self.nmultiband):
                    mu_matrix[iband, jband] = mu_matrix[iband, iband]*off_diag_scale
                    mu_matrix[jband, iband] = mu_matrix[iband, jband]*dos[iband]/dos[jband]
        elif(way == 3):
            k = 0.0
            k1 = 0.0
            tot_dos = 0.0
            for iband in range(self.nmultiband):
                for jband in range(iband+1, self.nmultiband):
                    k += 2.0*dos[iband]**2
                k1 += dos[iband]**2
                tot_dos += dos[iband] 
            k = k/tot_dos**2*off_diag_scale + k1/tot_dos**2
            k = mu/k
            for iband in range(self.nmultiband):
                mu_matrix[iband, iband] = k*dos[iband]/tot_dos
                for jband in range(iband + 1,self.nmultiband):
                    mu_matrix[iband, jband] = mu_matrix[iband, iband]*off_diag_scale
                    mu_matrix[jband, iband] = mu_matrix[iband, jband]*dos[iband]/dos[jband]
                            
        return mu_matrix

    def save(self, filename = 'sc.pkl'):

        import pickle

        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

