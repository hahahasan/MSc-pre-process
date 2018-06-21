import idlbridge as idl
import matplotlib.pyplot as plt

# example for getting MAST data on friea

getdata = idl.export_function("getdata")

## want to get Te profiles

if False:
  ip = getdata('AMC_PLASMA CURRENT', 30417, noecho=1)
  plt.figure(1)
  plt.plot(ip['time'],ip['data'])

psi=getdata('efm_psi(r,z)', 30417, noecho=1)['data']
plt.contour(psi[20,:,:], 50)
plt.show()



ayc_te = getdata('ayc_te', 28819, noecho=1)
ayc_ne = getdata('ayc_ne', 28819, noecho=1)
ayc_r = getdata('ayc_r', 28819, noecho=1)

efm_grid_r = getdata('efm_grid(r)', 28819, noecho=1)
efm_grid_z = getdata('efm_grid(z)', 28819, noecho=1)
efm_f_psi_c = getdata('efm_f(psi)_(c)', 28819, noecho=1)
efm_chisq_magnetic = getdata('efm_chisq_magnetic', 28819, noecho=1)
efm_lcfs_r_c = getdata('efm_lcfs(r)_(c)', 28819, noecho=1)
efm_lcfs_z_c = getdata('efm_lcfs(z)_(c)', 28819, noecho=1)
efm_magnetic_axis_r = getdata('efm_magnetic_axis_r', 28819, noecho=1)
efm_magnetic_axis_z = getdata('efm_magnetic_axis_z', 28819, noecho=1)

efm_magpri_chisq = getdata('efm_magpri_chisq', 28819, noecho=1)
efm_magpr_c = getdata('efm_magpr_(c)', 28819, noecho=1)
efm_magpr_x = getdata('efm_magpr_(x)', 28819, noecho=1)
efm_magpr_r = getdata('efm_magpr_r', 28819, noecho=1)
efm_magpr_z = getdata('efm_magpr_z', 28819, noecho=1)

efm_minor_radius = getdata('efm_magpr_z', 28819, noecho=1)

efm_p_psi_c = getdata('efm_p(psi)_(c)', 28819, noecho=1)

efm_plasma_curr_c = getdata('efm_plasma_curr(c)', 28819, noecho=1)
efm_plasma_curr_rz = getdata('efm_plasma_curr(r,z)', 28819, noecho=1)
efm_plasma_curr_x = getdata('efm_plasma_curr(x)', 28819, noecho=1)

efm_pprime = getdata('efm_pprime', 28819, noecho=1)
efm_q_psi_c = getdata('efm_q(psi)_(c)', 28819, noecho=1)

efm_q_100 = getdata('efm_q_100', 28819, noecho=1)
efm_q_90 = getdata('efm_q_90', 28819, noecho=1)
efm_q_95 = getdata('efm_q_95', 28819, noecho=1)
efm_q_axis = getdata('efm_q_axis', 28819, noecho=1)

efm_silop_c = getdata('efm_silop_(c)', 28819, noecho=1)
efm_silop_x = getdata('efm_silop_(x)', 28819, noecho=1)
efm_silop_chisq = getdata('efm_silop_chisq', 28819, noecho=1)

efm_silop_r = getdata('efm_silop_r', 28819, noecho=1)
efm_silop_z = getdata('efm_silop_z', 28819, noecho=1)





plt.subplot(121)
plt.contourf(ayc_te['time'],ayc_te['x'],ayc_te['data'].T, 50)
plt.colorbar()
plt.subplot(122)
plt.contourf(ayc_ne['time'],ayc_ne['x'],ayc_ne['data'].T, 50)
plt.colorbar()

plt.show()







