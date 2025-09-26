import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ThermalConductivity
from cellconstructor.Units import *
from scipy.optimize import brentq
from parsing import read_elph, read_wee
import numpy as np
import os

EV_TO_MEV = 1000.0000000000000000000
RY_TO_MEV = RY_TO_EV * EV_TO_MEV

def load_mesolver(filename = 'sc.pkl'):

    import pickle

    infile = open(filename, 'rb')
    sc = pickle.load(infile)
    infile.close()
    return sc

def get_starting_number_electrons(wee_energy, wee_dos, efermi, T):

    occs = Fermi_Dirac(wee_energy, efermi, T)
    return np.trapz(occs*wee_dos, wee_energy)

def get_number_electrons(efermi, wee_energy, wee_dos, T, w, chi, theta):

    occs = Fermi_Dirac(wee_energy, efermi, T)
    wee_diff = wee_energy - efermi
    wee_diff = wee_diff[:, np.newaxis]
    chi_b = chi[np.newaxis, :]      
    w_b = w[np.newaxis, :]
    sumand = np.sum((wee_diff + chi_b)/theta - wee_diff/(w_b**2 + wee_diff**2), axis=1)
    return np.trapz(occs*wee_dos - 2.0* T * sumand * wee_dos, wee_energy)

def fermi_root(efermi, wee_energy, wee_dos, T, w, chi, theta, Ne):
    return get_number_electrons(efermi, wee_energy, wee_dos, T, w, chi, theta) - Ne

def check_if_opposite_true(D, Z, W, T, lnm, lnm_mu, thr):
    D_old = D.copy()
    v1 = np.zeros_like(W)
    v2 = np.zeros_like(W)
    for i in range(len(W)):
        v1[i] = W[i]/np.sqrt(W[i]**2 + D[i]**2)
        v2[i] = D[i]/np.sqrt(W[i]**2 + D[i]**2)
    for i in range(len(W)):
        suma = 0.0
        for j in range(len(W)):
            suma += lnm[i,j]*v1[j]
        Z[i] = 1.0 + np.pi*T*suma/W[i]
        suma = 0.0
        for j in range(len(W)):
            suma += lnm_mu[i,j]*v2[j]
        D[i] = suma*np.pi*T/Z[i]
    diff = np.linalg.norm(D - D_old)/np.linalg.norm(D_old)
    if(diff <= thr):
        return True
    else:
        return False

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

    def __init__(self, a2f = None, wee = None):

        self.a2f = a2f
        self.wee = wee
        self.pdos = None
        self.lambdas = None

        self.thr = 1.0e-4
        self.mix = 1.0
        self.max_iter = 1000
        self.starting_Coulomb_weight = 100.0
        self.aa = False
        self.anderson = 4
        self.beta = 0.3

        self.wee_freq = None
        self.wee_energy = None
        self.wee_dos = None
        self.wee = None

    def load(self, dyn_filename = None, nqirr = 1, elphdyn_filename = None, wee_filename = None):
    
        from scipy.interpolate import RegularGridInterpolator

        if(dyn_filename is not None):
            dyn = CC.Phonons.Phonons(dyn_filename, nqirr)
        if(elphdyn_filename is not None):
            iqr = 1
            while(os.path.isfile(elphdyn_filename + str(iqr))):
                iqr += 1
            elph_nqirr = iqr - 1
            qpts, smearings, dos, elph, weights, qstar = read_elph(elphdyn_filename, elph_nqirr, dyn.structure.N_atoms)
            self.elph_qpts = qpts
            self.elph_smearings = smearings
            self.elph_dos = dos
            self.ep_deformation_potentials = elph
            self.ep_nqirr = np.shape(self.ep_deformation_potentials)[0]
            self.ep_nsmear = np.shape(self.ep_deformation_potentials)[1]
            self.elph_weights = weights
            self.elph_qstar = qstar
            elph_dyn = CC.Phonons.Phonons(elphdyn_filename, elph_nqirr)
            if(np.all(elph_dyn.GetSupercell() == dyn.GetSupercell())):
                self.dyn = dyn
            else:
                self.dyn = dyn.Interpolate(coarse_grid = dyn.GetSupercell(), fine_grid = elph_dyn.GetSupercell(), symmetrize = True)
                self.dyn.save_qe('interpolated_dyn')
        if(wee_filename is not None):
            freq, wee_energy, dos_energy, dos, re_wee, im_wee = read_wee(wee_filename)
            self.wee_freq = freq * EV_TO_MEV

            # Cut off parts of the dos for which we do not have W(e,e') calculated
            using_dos_energy = []
            using_dos = []
            for i in range(len(dos_energy)):
                if(dos_energy[i] >= min(wee_energy) and dos_energy[i] <= max(wee_energy)):
                    using_dos_energy.append(dos_energy[i])
                    using_dos.append(dos[i])

            self.wee_energy = np.array(using_dos_energy) * EV_TO_MEV
            self.wee_dos = np.array(using_dos) / EV_TO_MEV

            # Interpolate W(e,e')
            interp_re = RegularGridInterpolator((wee_energy, wee_energy), re_wee[0]) 
            
            X, Y = np.meshgrid(using_dos_energy, using_dos_energy, indexing='ij')

            self.wee = interp_re((X,Y)) * EV_TO_MEV

        if(dyn_filename is None and elphdyn_filename is None and wee_filename is None):
            #raise RuntimeError('You need provide a path to dynamical matrix  or deformation potential or Coulomb interaction file!')
            print('You need provide a path to dynamical matrix  or deformation potential or Coulomb interaction file!')

    def calculate_a2f(self, anharmonic = False, comm_dyn_filename = None, comm_nqirr = 1, third_order_filename = None, nom = 2000, scattering_mesh = [10,10,10], mode_mixing = 'no', on_full_grid = False, temperature = 0.0, a2f_smearing = 0.5, automatic_a2f_smearing = False):

        
        supercell_matrix = self.dyn.GetSupercell()

        if(anharmonic):
            if(comm_dyn is None or third_order_filename is None):
                raise RuntimeError('You need to provide paths for the dynamical matrices and third order force constants used for lineshape calculation!')
            else:
                comm_dyn = CC.Phonons.Phonons(comm_dyn_filename, comm_nqirr)
                comm_supercell_matrix = comm_dyn.GetSupercell()
                fc3 = CC.ForceTensor.Tensor3(comm_dyn.structure, comm_dyn.structure.generate_supercell(comm_supercell_matrix), comm_supercell_matrix)
                d3 = np.load(third_order)*2.0
                fc3.SetupFromTensor(d3)
                fc3 = CC.ThermalConductivity.centering_fc3(fc3)
        else:
            comm_dyn = self.dyn.Interpolate(coarse_grid = self.dyn.GetSupercell(), fine_grid = [1,1,1], symmetrize = True)
            comm_supercell_matrix = comm_dyn.GetSupercell()
            fc3 = CC.ForceTensor.Tensor3(comm_dyn.structure, comm_dyn.structure.generate_supercell(comm_supercell_matrix), comm_supercell_matrix)
        tc = CC.ThermalConductivity.ThermalConductivity(self.dyn, fc3, kpoint_grid = supercell_matrix, scattering_grid = scattering_mesh, smearing_scale = 1.0, smearing_type = 'adaptive', cp_mode = 'quantum', off_diag = False, phase_conv = 'step')
        temperature = temperature
        tc.setup_harmonic_properties()
        qpt_id, qpt_id1 = match_qpoints(self.elph_qpts, tc)

        freq_max = np.amax(tc.freqs)*2.1
        self.a2f_omega = (np.arange(nom, dtype = float) + 1.0)/float(nom)*freq_max
        smearing = self.a2f_omega[1]*a2f_smearing

        if(mode_mixing == 'no' or not anharmonic):
            self.aq = np.zeros((self.ep_nsmear, self.ep_nqirr, tc.nband, nom))
            self.pq_dos = np.zeros((self.ep_nqirr, tc.nband, nom))
            self.lambdas = np.zeros((self.ep_nsmear, self.ep_nqirr, tc.nband))
        else:
            if(on_full_grid):
                self.aq = np.zeros((self.ep_nsmear, tc.nkpt, tc.nband, tc.nband, nom))
                self.pq_dos = np.zeros((tc.nkpt, tc.nband, tc.nband, nom), dtype=complex)
                self.lambdas = np.zeros((self.ep_nsmear, tc.nkpt, tc.nband))
            else:
                self.aq = np.zeros((self.ep_nsmear, self.ep_nqirr, tc.nband, tc.nband, nom))
                self.pq_dos = np.zeros((self.ep_nqirr, tc.nband, tc.nband, nom), dtype=complex)
                self.lambdas = np.zeros((self.ep_nsmear, self.ep_nqirr, tc.nband))

        if(anharmonic):
            tc.get_lineshapes(temperature, write_lineshapes = False, energies = omega, method = 'fortran', mode_mixing = mode_mixing, gauss_smearing = False)
            if(on_full_grid):
                for ikpt in range(tc.nkpt):
                    self.pq_dos[ikpt] = tc.lineshapes[format(temperature, '.1f')][ikpt].copy()
                    if(mode_mixing == 'cartesian'):
                        self.pq_dos[ikpt] = np.einsum('ji, jkl, km->iml', tc.eigvecs[ikpt], self.pq_dos[ikpt], tc.eigvecs[ikpt].conj())
            else:
                for ikpt in range(len(qpt_id)):
                    self.pq_dos[ikpt] = float(self.elph_weights[ikpt]) * tc.lineshapes[format(temperature, '.1f')][qpt_id[ikpt]]
                    if(mode_mixing == 'cartesian'):
                        self.pq_dos[ikpt] = np.einsum('ji, jkl, km->iml', tc.eigvecs[qpt_id[ikpt]], self.pq_dos[ikpt], tc.eigvecs[qpt_id[ikpt]].conj())
        else:
            for ikpt in range(len(qpt_id1)):
                if(not automatic_a2f_smearing):
                    self.pq_dos[ikpt] = float(self.elph_weights[ikpt]) * gaussian(self.a2f_omega, tc.freqs[qpt_id1[ikpt]], np.array([smearing for x in range(len(tc.freqs[qpt_id1[ikpt]]))]))
                else:
                    self.pq_dos[ikpt] = float(self.elph_weights[ikpt]) * gaussian(self.a2f_omega, tc.freqs[qpt_id1[ikpt]], tc.sigmas[qpt_id1[ikpt]])


        if(on_full_grid and mode_mixing != 'no'):
            elph_full = get_elph_on_full_grid(tc, deformation_potentials, qpt_id1, qpt_id)
        elif(on_full_grid and mode_mixing == 'no'):
            raise RuntimeError('You do not need full grid if you are not using mode mixing approach! Set mode_mixing different to no or set on_full_grid to True!')

        self.a2f = np.zeros((self.ep_nsmear, nom), dtype=float)
        self.pdos = np.zeros(nom)

        for ism in range(self.ep_nsmear):
            deformation_potentials = self.ep_deformation_potentials[:,ism,:,:]
            dos = self.elph_dos[ism]
            if(on_full_grid):
                for ikpt in range(tc.nkpt):
                    if(np.linalg.norm(tc.qpoints[ikpt]) > 1.0e-6):
                        eigs_scaled = np.zeros_like(tc.eigvecs[iqpt])
                        for iat in range(tc.dyn.structure.N_atoms):
                            eigs_scaled[3*iat:3*(iat + 1),:] = tc.eigvecs[ikpt][3*iat:3*(iat + 1),:]/np.sqrt(tc.dyn.structure.masses[tc.dyn.structure.atoms[iat]])
                        matprod = np.einsum('ji,jk,kl->il', eigs_scaled, elph_full[ikpt], eigs_scaled.conj())
                        self.aq[ism, ikpt] = np.einsum('ij,ijk,k->ijk',matprod, self.pq_dos[ikpt] ,1.0/self.a2f_omega)/4.0/dos
                        self.a2f[ism] += np.einsum('ijk->k', self.aq[ism, ikpt].real)
                        if(ism == 0):
                            self.pdos += np.einsum('iik->k', self.pq_dos[ikpt].real)
                        self.lambdas[ism, ikpt] = np.diag(matprod.real)/2.0/tc.freqs[ikpt]**2/dos
            else:
                for ikpt in range(len(qpt_id)):
                    if(np.linalg.norm(tc.qpoints[qpt_id1[ikpt]]) > 1.0e-6):
                        eigs_scaled = np.zeros_like(tc.eigvecs[qpt_id1[ikpt]])
                        for iat in range(tc.dyn.structure.N_atoms):
                            eigs_scaled[3*iat:3*(iat + 1),:] = tc.eigvecs[qpt_id1[ikpt]][3*iat:3*(iat + 1),:]/np.sqrt(tc.dyn.structure.masses[tc.dyn.structure.atoms[iat]])
                        if(mode_mixing == 'no' or not anharmonic):
                            matprod = np.einsum('ji,jk,ki->i', eigs_scaled, deformation_potentials[ikpt], eigs_scaled.conj()).real
                            self.aq[ism, ikpt] = np.einsum('i,ik,k->ik',matprod, self.pq_dos[ikpt] ,1.0/self.a2f_omega)/4.0/dos
                            self.a2f[ism] += np.einsum('ik->k', self.aq[ism, ikpt].real)
                            if(ism == 0):
                                self.pdos += np.einsum('ik->k', self.pq_dos[ikpt].real)
                            self.lambdas[ism, ikpt] = matprod.real/2.0/tc.freqs[qpt_id1[ikpt]]**2/dos
                        else:
                            matprod = np.einsum('ji,jk,kl->il', eigs_scaled, deformation_potentials[ikpt], eigs_scaled.conj())
                            self.aq[ism, ikpt] = np.einsum('ij,ijk,k->ijk',matprod, self.pq_dos[ikpt] ,1.0/self.a2f_omega)/4.0/dos
                            self.a2f[ism] += np.einsum('ijk->k', self.aq[ism, ikpt].real)
                            if(ism == 0):
                                self.pdos += np.einsum('iik->k', self.pq_dos[ikpt].real)
                            self.lambdas[ism, ikpt] = np.diag(matprod.real)/2.0/tc.freqs[qpt_id1[ikpt]]**2/dos

        self.a2f_omega = self.a2f_omega * RY_TO_MEV
        self.a2f = self.a2f/float(np.sum(self.elph_weights) - 1)
        self.pdos = self.pdos/float(np.sum(self.elph_weights) - 1) / RY_TO_MEV 

    def write_a2f(self, smear_id = 0, a2f_filename = 'a2f'):

        if(self.a2f is None):
            raise RuntimeError(r'\alpha^2F is not calculated!')

        tot_lambda = self.get_lambda()
        with open(a2f_filename, 'w+') as outfile:
            outfile.write('# ')
            outfile.write(r' \lambda = ' + format(tot_lambda, '.3f'))
            outfile.write(r' for electical smearing ' + format(self.elph_smearings[smear_id], '.6f') + '\n')
            outfile.write('# Frequency (meV)          alpha2f             Phonon DOS (1/meV)  \n')
            for i in range(len(self.a2f_omega)):
                outfile.write(3*' ' + format(self.a2f_omega[i], '.12f'))
                outfile.write(3*' ' + format(self.a2f[smear_id, i], '.12f'))
                outfile.write(3*' ' + format(self.pdos[i], '.12f') + '\n')
                        
    def get_lambda(self, smear_id = 0):

        if(self.a2f is None):
            raise RuntimeError(r'\alpha^2F is not calculated!')

        return np.trapz(self.a2f[smear_id]/self.a2f_omega, self.a2f_omega)*2.0

    def get_lambda_from_mode_lambdas(self, smear_id = 0):

        if(self.lambdas is None):
            raise RuntimeError('Mode resolved lambdas not calculated!')
        
        band_lambdas = np.average(self.lambdas[smear_id], axis=0, weights=self.elph_weights)
    
        return np.sum(band_lambdas)

    def get_omega_log(self, units = '1/cm', smear_id = 0):

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


    def solve_isotropic(self, T_start = 1.0, T_end = 300.0, wcut_ratio = 10.0, mu_star_approx = True, ntemp = 2, mu_star = 0.15, gap_filename = 'SC_GAP', log = False, smear_id = 0, constant_dos = False):

        if(not mu_star_approx and self.wee is None):
            raise RuntimeError('You have not supplied Coulomb interaction. Can not continue calculation!')

        w_cut = max(self.a2f_omega)*wcut_ratio
        self.temperatures = np.linspace(T_start, T_end, num=ntemp)

        self.iw = []
        self.delta = []
        self.Z = []
        self.indices = []
        if(not mu_star_approx):
            self.chi = []
            self.phi_c = []
            self.ef = []
            starting_Ne = get_starting_number_electrons(self.wee_energy, self.wee_dos, 0.0, 0.0)
            print('Starting number of electrons: ' + format(starting_Ne, '.1f') + '.')
        gap_is_zero = False

        for i in range(ntemp):
            if(log):
                print('Calculating for temperature: ' +  format(self.temperatures[i], '.2f') + ' K!')
            if(not gap_is_zero):
                temp_mev = self.temperatures[i] * K_B * EV_TO_MEV
                if(i == 0):
                    delta0 = self.get_initial_gap_estimate(mu_star, smear_id = smear_id)
                    delta_init = delta0
                    if(mu_star_approx):
                        w_i, delta_i, z_i, index = self.solve_isotropic_at_T_mu_star(mu_star, w_cut, temp_mev, delta0, log, smear_id = smear_id)
                    else:
                        if(constant_dos):
                            w_i, delta_i, z_i, phi_c, index = self.solve_isotropic_at_T_constant_dos(w_cut, temp_mev, delta0, starting_Ne, log, smear_id = smear_id)
                        else:
                            w_i, delta_i, z_i, chi_i, phi_c, index, ef = self.solve_isotropic_at_T(w_cut, temp_mev, delta0, starting_Ne, log, smear_id = smear_id)
                else:
                    delta0 = self.delta[-1][self.indices[-1]]
                    if(mu_star_approx):
                        w_i, delta_i, z_i, index = self.solve_isotropic_at_T_mu_star(mu_star, w_cut, temp_mev, delta0, log, smear_id = smear_id)
                    else:
                        if(constant_dos):
                            w_i, delta_i, z_i, phi_c, index = self.solve_isotropic_at_T_constant_dos(w_cut, temp_mev, delta0, starting_Ne, log, smear_id = smear_id)
                        else:
                            w_i, delta_i, z_i, chi_i, phi_c, index, ef = self.solve_isotropic_at_T(w_cut, temp_mev, delta0, starting_Ne, log, smear_id = smear_id)
            else:
                w_i = [0.0]
                delta_i = [0.0]
                z_i = [0.0]
                index = 0
                if(not mu_star_approx):
                    chi_i = [0.0]
                    phi_c = np.zeros_like(self.wee_energy)
                    ef = 0.0


            self.iw.append(w_i)
            self.delta.append(delta_i)
            self.Z.append(z_i)
            self.indices.append(index)
            if(not mu_star_approx):
                if(not constant_dos):
                    self.chi.append(chi_i)
                    self.ef.append(ef)
                self.phi_c.append(phi_c)
                if(not constant_dos):
                    for j in range(1, len(self.wee_energy)):
                        if(self.wee_energy[j] > self.ef[i] and self.wee_energy[j-1] < self.ef[i]):
                            energy_index = j - 1
                            break
                else:
                    for j in range(1, len(self.wee_energy)):
                        if(self.wee_energy[j] > 0.0 and self.wee_energy[j-1] < 0.0):
                            energy_index = j - 1
                            break
                        elif(self.wee_energy[j] == 0.0):
                            energy_index = j
                            break
                if(z_i[index] != 0.0 or index == 0):
                    curr_gap = (delta_i[index] + phi_c[energy_index])/z_i[index]
                else:
                    curr_gap = 0.0
            else:
                curr_gap = delta_i[index]

            
            if(not gap_is_zero and curr_gap < delta_init*1.0e-3):
                gap_is_zero = True
                self.tc = self.temperatures[i]

        with open(gap_filename, 'w+') as outfile:
            outfile.write('# Temperature (K)            SC Gap (meV)      \n')
            for i in range(ntemp):
                if(self.delta[i][self.indices[i]] != 0.0):
                    outfile.write(8*' ' + format(self.temperatures[i], '.3f'))
                    if(mu_star_approx):
                        outfile.write(8*' ' + format(self.delta[i][self.indices[i]], '.12f') + '\n')
                    else:
                        if(not constant_dos):
                            for j in range(1, len(self.wee_energy)):
                                if(self.wee_energy[j] > self.ef[i] and self.wee_energy[j-1] < self.ef[i]):
                                    energy_index = j - 1
                                    break
                        gap = (self.delta[i][self.indices[i]] + self.phi_c[i][energy_index])/self.Z[i][self.indices[i]]
                        outfile.write(8*' ' + format(gap, '.12f') + '\n')
                else:
                    outfile.write(8*' ' + format(0.0, '.12f') + '\n')

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
        lnm = lnm + lnm.T - np.diag(np.diag(lnm))

        return lnm
        
    def solve_isotropic_at_T_mu_star(self, mu_star, w_cut, T, delta0, log = False, smear_id = 0):

        diff = 1.0
        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_full = self.get_matsubara_frequencies(w_cut, T)
        lnm = self.generate_Lnm_matrix(w_full, smear_id = smear_id)
        index = 0
        n_max = len(w)
        delta1 = np.zeros(len(w))
        delta1[index] = delta0
        z1 = np.ones(len(w))
        z1[index] = 2.0
        iteration = 1
        lp = np.zeros((n_max, n_max), dtype = float) 
        lm = np.zeros_like(lp)
        if(w_full[n_max + 1] > 0.0 and w_full[n_max] < 0.0):
            for i in range(n_max):
                for j in range(n_max):
                    lp[i,j] = lnm[i + n_max + 1, n_max - j] + lnm[i + n_max + 1, j + n_max + 1]
                    lm[i,j] = -1.0*(lnm[i + n_max + 1, n_max - j] - lnm[i + n_max + 1, j + n_max + 1])
        else:
            raise RuntimeError('Wrong choice of w0')
        lp_mu = lp - 2.0 * mu_star
        while(diff > self.thr and iteration < self.max_iter):

            delta2 = delta1.copy()
            z2 = z1.copy()

            v1 = w/np.sqrt(w**2 + delta1**2)
            v2 = delta1/np.sqrt(w**2 + delta1**2)

            z1 = 1.0 + np.dot(lm, v1)*np.pi*T/w
            delta1 = np.dot(lp_mu, v2)*np.pi*T/z2

            delta1 = self.mix*delta1 + (1.0 - self.mix)*delta2
            z1 = self.mix*z1 + (1.0 - self.mix)*z2

            diff = np.linalg.norm(delta1 - delta2)/np.linalg.norm(delta2)

            iteration += 1
        if(iteration >= self.max_iter):
            if(log):
                print('Calculation did not converge!')
        else:
            if(log):
                print('Converged in ' + str(iteration) + ' iterations. The difference was: ' + format(diff, '.3e'))
            if(delta1[index] < 0.0):
                if(log):
                    print('Negative gap, hmmm. Checking if the opposite solution satisfies self-consistency ...')
                same = check_if_opposite_true(-1.0*delta1, z1, w, T, lnm, lnm_mu, self.thr)
                if(same):
                    if(log):
                        print('Opposite solution satisfies self-consistency! Using opposite solution.')
                    delta1 = -1.0*delta1

        return w, delta1, z1, index

    def solve_isotropic_at_T(self, w_cut, T, delta0, Ne, log = False, smear_id = 0):

        coulomb_weight = self.starting_Coulomb_weight
        mix = self.mix

        diff = 1.0
        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_full = self.get_matsubara_frequencies(w_cut, T)
        lnm = self.generate_Lnm_matrix(w_full, smear_id = smear_id)
        index = 0
        n_max = len(w)

        phi1 = np.zeros(n_max, dtype=float)
        phi1[index] = delta0
        phi_c = np.zeros(len(self.wee_dos), dtype=float) 
        z1 = np.ones(n_max, dtype=float) 
        z1[index] = 2.0
        chi1 = np.zeros_like(phi1)
        iteration = 1

        lp = np.zeros((n_max, n_max), dtype = float) 
        lm = np.zeros_like(lp)
        for i in range(n_max):
            for j in range(n_max):
                lp[i,j] = lnm[i + n_max + 1, n_max - j] + lnm[i + n_max + 1, j + n_max + 1]
                lm[i,j] = -1.0*(lnm[i + n_max + 1, n_max - j] - lnm[i + n_max + 1, j + n_max + 1])

        theta = np.zeros((len(self.wee_dos), n_max))
        en_shift = np.zeros_like(theta)
        tot_phi = np.zeros_like(theta)
        ef = 0.0
        de_wee_energy = (self.wee_energy[1] - self.wee_energy[0])
        Nef = 0.0
        ief = 0
        ief = np.argmin(np.abs(self.wee_energy))
        Nef = self.wee_dos[ief]
        if(Nef == 0.0):
            raise RuntimeError('Density of states at the Fermi level is 0.0!')
        phi_c[ief] = -0.00001

        if(self.aa):
            history_x = []
            history_x_star = []
            history_f = []

        mu_star = self.wee_dos[ief]*self.wee[ief,ief]/5.0
        w0, delta1, z0, index0 = self.solve_isotropic_at_T_mu_star(mu_star, w_cut, T, delta0, log = log, smear_id = smear_id)
        if(np.all(w0 == w)):
            phi1 = delta1*z0
            z1 = z0
            theta = np.maximum(((w*z1)[np.newaxis, :])**2 + (self.wee_energy[:, np.newaxis])**2 + phi1**2, 1.0e-12)
            sqrt_term = np.maximum(np.sqrt(phi_c ** 2 + self.wee_energy** 2), 1.0e-12)
            term2 =  T * np.sum(phi1[np.newaxis, :]/theta, axis=1)
            phi_c = -2.0*np.einsum('i,ji->j', self.wee_dos * term2, self.wee) * de_wee_energy
        else:
            raise RuntimeError('This should not happen!')

        while((diff > self.thr and iteration < self.max_iter and abs(phi1[index]/z1[index]) > 1.0e-6) or iteration < int(self.starting_Coulomb_weight) + 1):

            coulomb_weight = min(1.0, round(float(iteration)/self.starting_Coulomb_weight + 0.05, 1))
            prev_diff = diff
            phi2 = phi1.copy()
            z2 = z1.copy()
            phi_c2 = phi_c.copy()
            chi2 = chi1.copy()
            ef2 = ef

            en_shift = self.wee_energy[:,np.newaxis] - ef + chi1[np.newaxis,:]
            tot_phi = phi1[np.newaxis,:] + phi_c[:,np.newaxis]
            theta = (w*z1)[np.newaxis,:]**2 + en_shift**2 + tot_phi**2

            z1 = 1.0 + T/w/Nef * np.einsum('k,kj,ij ->i', self.wee_dos, (w*z2)[np.newaxis,:]/theta, lm) * de_wee_energy

            chi1 = -1.0*T/Nef * np.einsum('k,kj,ij ->i', self.wee_dos, en_shift/theta, lp) * de_wee_energy

            phi1 =  T/Nef * np.einsum('k,kj,ij ->i', self.wee_dos, tot_phi/theta, lp) * de_wee_energy

            sqrt_term = np.sqrt(phi_c ** 2 + (self.wee_energy - ef) ** 2)
            term1 =  phi_c/2.0 * np.tanh(sqrt_term/2.0/T)/sqrt_term
            term2 = np.zeros_like(term1)
            term2 = 2.0 * T * np.sum(tot_phi/theta - phi_c[:, np.newaxis]/(w[np.newaxis,:]**2 + sqrt_term[:, np.newaxis]**2), axis=1)
            phi_c = -coulomb_weight*np.einsum('i,ij->j', self.wee_dos * (term1 + term2), self.wee) * de_wee_energy

            ef = brentq(fermi_root, -100.0 + ef, 100.0 + ef, args=(self.wee_energy, self.wee_dos, T, w, chi2, theta, Ne), xtol=1e-6)

            diff = np.linalg.norm(phi1 - phi2)/np.linalg.norm(phi2)

            if(self.aa):
                history_x.append(np.concatenate([phi2, z2, phi_c2]))
                history_x_star.append(np.concatenate([phi1, z1, phi_c]))
                history_f.append(np.concatenate([phi1 - phi2, z1 - z2, phi_c - phi_c2]))
                if(len(history_f) > self.anderson):
                    history_x.pop(0)
                    history_x_star.pop(0)
                    history_f.pop(0)

                if(len(history_f) >= self.anderson):
                    F = np.array(history_f).T
                    F =  F / np.linalg.norm(F, axis=0)
                    try:
                        alpha = np.linalg.lstsq(F, history_f[-1], rcond=None)[0]
                        alpha = alpha / np.sum(alpha)
                    except np.linalg.LinAlgError:
                        alpha = np.ones(len(history_f)) / len(history_f)
                    X = np.array(history_x).T
                    new_guess = X @ alpha
                    new_guess_star = np.array(history_x_star).T @ alpha  
                    phi1 = (1.0 - self.beta) * new_guess[:len(phi1)] + self.beta * new_guess_star[:len(phi1)]
                    z1 = (1.0 - self.beta) * new_guess[len(phi1):len(phi1) + len(z1)] + self.beta * new_guess_star[len(phi1):len(phi1) + len(z1)]
                    phi_c = (1.0 - self.beta) * new_guess[len(phi1) + len(z1):] + self.beta * new_guess_star[len(phi1) + len(z1):]
                else:
                    if(diff < prev_diff):
                        mix = max(1.0, mix*1.1)
                    elif(diff > prev_diff):
                        mix = min(0.1, mix/2.0)
                    phi1 = (1.0 - mix)*phi2 + mix*phi1
                    z1 = (1.0 - mix)*z2 + mix*z1
                    phi_c = (1.0 - mix)*phi_c2 + mix*phi_c
            else:
                if(diff < prev_diff and iteration > 1):
                    mix = min(1.0, mix*1.1)
                elif(diff > prev_diff):
                    mix = max(0.5, mix*0.9)
                phi1 = (1.0 - mix)*phi2 + mix*phi1
                z1 = (1.0 - mix)*z2 + mix*z1
                phi_c = (1.0 - mix)*phi_c2 + mix*phi_c

            iteration += 1
        if(iteration >= self.max_iter):
            if(log):
                print('Calculation did not converge!')

        return w, phi1, z1, chi1, phi_c, index, ef

    def solve_isotropic_at_T_constant_dos(self, w_cut, T, delta0, Ne, log = False, smear_id = 0):

        coulomb_weight = self.starting_Coulomb_weight
        mix = self.mix

        diff = 1.0
        w = self.get_matsubara_frequencies_only_positive(w_cut, T)
        w_full = self.get_matsubara_frequencies(w_cut, T)
        lnm = self.generate_Lnm_matrix(w_full, smear_id = smear_id)
        index = 0
        n_max = len(w)

        phi1 = np.zeros(n_max, dtype=float)
        phi1[index] = delta0
        phi_c = np.zeros(len(self.wee_dos), dtype=float) 
        z1 = np.ones(n_max, dtype=float)  + 1.0
        iteration = 1

        lp = np.zeros((n_max, n_max), dtype = float) 
        lm = np.zeros_like(lp)
        for i in range(n_max):
            for j in range(n_max):
                lp[i,j] = lnm[i + n_max + 1, j + n_max + 1] + lnm[i + n_max + 1, n_max - j]
                lm[i,j] = lnm[i + n_max + 1, j + n_max + 1] - lnm[i + n_max + 1, n_max - j]
        
        theta = np.zeros((len(self.wee_dos), n_max))

        ef = 0.0
        de_wee_energy = (self.wee_energy[1] - self.wee_energy[0])
        Nef = 0.0
        ief = 0
        ief = np.argmin(np.abs(self.wee_energy))
        Nef = self.wee_dos[ief]
        if(Nef == 0.0):
            raise RuntimeError('Density of states at the Fermi level is 0.0!')
        phi_c[ief] = -0.00001

        if(self.aa):
            history_x = []
            history_x_star = []
            history_f = []

        mu_star = 0.0 
        w0, delta1, z0, index0 = self.solve_isotropic_at_T_mu_star(mu_star, w_cut, T, delta0, log = log, smear_id = smear_id)
        if(np.all(w0 == w)):
            phi1 = delta1*z0
            z1 = z0
            theta = np.maximum(((w*z1)[np.newaxis, :])**2 + (self.wee_energy[:, np.newaxis])**2 + phi1**2, 1.0e-12)
            term2 =  T * np.sum(phi1[np.newaxis, :]/theta, axis=1)
            phi_c = -2.0*np.einsum('i,ji->j', self.wee_dos * term2, self.wee) * de_wee_energy
        else:
            raise RuntimeError('This should not happen!')
        while((diff > self.thr and iteration < self.max_iter and abs(phi1[index]/z1[index]) > 1.0e-6) or iteration < int(self.starting_Coulomb_weight*1.1)):

            coulomb_weight = min(1.0, round(float(iteration)/self.starting_Coulomb_weight + 0.05, 1))
            prev_diff = diff
            phi2 = phi1.copy()
            z2 = z1.copy()
            phi_c2 = phi_c.copy()

            tot_phi = phi2[np.newaxis,:] + phi_c2[:, np.newaxis]
            theta = ((w*z2)[np.newaxis, :])**2 + (self.wee_energy[:, np.newaxis])**2 + tot_phi**2

            z1 = 1.0 + np.pi*T/w * np.einsum('j,ij ->i',w*z2/np.sqrt(theta[ief]), lm) 

            phi1 =  T * np.pi * np.einsum('j,ij ->i', tot_phi[ief]/np.sqrt(theta[ief]), lp)
           
            sqrt_term = np.sqrt(phi_c ** 2 + self.wee_energy** 2)
            term1 = phi_c/2.0 * np.tanh(sqrt_term/2.0/T)/np.maximum(sqrt_term, 1.0e-12)
            term2 = np.zeros_like(term1)
            term2 =  2.0 * T * np.sum(tot_phi/theta - phi_c[:, np.newaxis]/((w[np.newaxis,:])**2 + (sqrt_term[:, np.newaxis])**2), axis=1)
            phi_c = -coulomb_weight * np.einsum('i,ij->j', self.wee_dos * (term1 + term2), self.wee) * de_wee_energy

            diff = np.linalg.norm(phi1 - phi2)/np.linalg.norm(phi2)

            if(self.aa):
                history_x.append(np.concatenate([phi2, z2, phi_c2]))
                history_x_star.append(np.concatenate([phi1, z1, phi_c]))
                history_f.append(np.concatenate([phi1 - phi2, z1 - z2, phi_c - phi_c2]))
                if(len(history_f) > self.anderson):
                    history_x.pop(0)
                    history_x_star.pop(0)
                    history_f.pop(0)

                if(len(history_f) >= 2):
                    F = np.array(history_f).T
                    F =  F / np.linalg.norm(F, axis=0)
                    try:
                        alpha = np.linalg.lstsq(F, history_f[-1], rcond=None)[0]
                        alpha = alpha / np.sum(alpha)
                    except np.linalg.LinAlgError:
                        alpha = np.ones(len(history_f)) / len(history_f)
                    X = np.array(history_x).T
                    new_guess = X @ alpha
                    new_guess_star = np.array(history_x_star).T @ alpha  
                    phi1 = (1.0 - self.beta) * new_guess[:len(phi1)] + self.beta * new_guess_star[:len(phi1)]
                    z1 = (1.0 - self.beta) * new_guess[len(phi1):len(phi1) + len(z1)] + self.beta * new_guess_star[len(phi1):len(phi1) + len(z1)]
                    phi_c = (1.0 - self.beta) * new_guess[len(phi1) + len(z1):] + self.beta * new_guess_star[len(phi1) + len(z1):]
                else:
                    if(diff < prev_diff):
                        mix = max(1.0, mix*1.1)
                    elif(diff > prev_diff):
                        mix = min(0.1, mix/2.0)
                    phi1 = (1.0 - mix)*phi2 + mix*phi1
                    z1 = (1.0 - mix)*z2 + mix*z1
                    phi_c = (1.0 - mix)*phi_c2 + mix*phi_c
            else:
                if(diff < prev_diff and iteration > 1):
                    mix = min(1.0, mix*1.1)
                elif(diff > prev_diff):
                    mix = max(0.5, mix*0.9)
                phi1 = (1.0 - mix)*phi2 + mix*phi1
                z1 = (1.0 - mix)*z2 + mix*z1
                phi_c = (1.0 - mix)*phi_c2 + mix*phi_c

            iteration += 1
        print('Finished in ' + str(iteration) + ' iterations.')
        if(iteration >= self.max_iter):
            if(log):
                print('Calculation did not converge!')

        return w, phi1, z1, phi_c, index

    def save(self, filename = 'sc.pkl'):

        import pickle

        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

