import numpy as np

r_ea = 6378
mu_e = 398600

orbit1_peri = 480 + r_ea
orbit1_apo = 800 + r_ea

orbit2_peri = 480 + r_ea
orbit2_apo = 16000 + r_ea

orbit3_peri = 16000 + r_ea
orbit3_apo = 16000 + r_ea

def energy_ellipse(peri, apo, mu):
    a = (peri + apo)/2
    return -1*mu/(2*a)

orbit1_energy = energy_ellipse(orbit1_peri, orbit1_apo, mu_e)
orbit2_energy = energy_ellipse(orbit2_peri, orbit2_apo, mu_e)
orbit3_energy = energy_ellipse(orbit3_peri, orbit3_apo, mu_e)

de_1 = orbit2_energy -  orbit1_energy
de_2 = orbit3_energy - orbit2_energy

e_o1 = (orbit1_apo-orbit1_peri)/(orbit1_peri+orbit1_apo)
h_o1 = np.sqrt(orbit1_peri*mu_e*(1+e_o1))
#v_peri_o1 = h_o1/orbit1_peri

def v_ellipse_peri(peri, apo, mu):
    e = (apo - peri)/(apo + peri)
    h = np.sqrt(peri*mu*(1+e))
    v_peri = h/peri
    return v_peri

def v_ellipse_apo(peri, apo, mu):
    e = (apo - peri)/(apo + peri)
    h = np.sqrt(peri*mu*(1+e))
    v_apo = h/apo
    return v_apo

def v_circle(r, mu):
    return np.sqrt(mu/r)

v_peri_o1 = v_ellipse_peri(orbit1_peri, orbit1_apo, mu_e)
v_peri_o2 = v_ellipse_peri(orbit2_peri, orbit2_apo, mu_e)

dv_1 = v_peri_o2 - v_peri_o1

v_apo_o2 = v_ellipse_apo(orbit2_peri, orbit2_apo, mu_e)
v_o3 = v_circle(orbit3_apo, mu_e)

dv_2 = v_o3 - v_apo_o2

total_dv = dv_1 + dv_2

