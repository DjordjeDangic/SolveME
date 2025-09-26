import sys
sys.path.append("/home/dangic/Softwares/ME_w_C/src/")

from cellconstructor.Units import RY_TO_CM, RY_TO_EV
import mesolver
import numpy as np
import time

wee_name = 'Wee.nc'
dyn_filename = './sscha_dense.dyn'
elphdyn_filename = './H3S.dyn'
nqirr = 16


me = mesolver.mesolver()
me.load(dyn_filename = dyn_filename, nqirr = nqirr, elphdyn_filename = elphdyn_filename, wee_filename = wee_name)
step = np.amax(me.dyn.DiagonalizeSupercell()[0])*2.1/2000.0
me.calculate_a2f(nom = 2000, temperature = 0.0, a2f_smearing = 5.0, automatic_a2f_smearing = True)
me.write_a2f()
print(me.wee_freq[0])

np.savetxt('DOS', np.array([me.wee_energy, me.wee_dos]).T)

print('Lambda from mode resolved lambdas: ', me.get_lambda_from_mode_lambdas())

start_time = time.time()
#me.mix = 0.1
me.max_iter = 10000
me.starting_Coulomb_weight = 100.0
me.aa = True
me.anderson = 21
for i in range(12, 13, 2):
    wcut_ratio = float(i)
    me.solve_isotropic(T_start = 101.0, T_end = 300.0, wcut_ratio = wcut_ratio, mu_star_approx = False, ntemp = 201, mu_star = 0.20, gap_filename = 'SC_GAP_w_cut_' + format(wcut_ratio, '.1f'), smear_id = 3, constant_dos = False)
print('Solved equations in ' + format(time.time() - start_time, '.2f') + ' seconds.')

me.save()
