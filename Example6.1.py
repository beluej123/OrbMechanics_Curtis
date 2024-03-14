# Curtis example 6.1, p.323 in my book; also see Orbit_from_r0v0.py
#   based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#   by Howard D. Curtis
# Given: earth altitude's r1 & r2 (not vectors), dt, and delta true anomaly;
#   Find: periapsis altitude, time to periapsis
import numpy as np

# A spacecraft is in a 480km by 800km earth orbit.
# (a) find the v required at perigee A to place the spacecraft in a 480km by
#   16000km transfer orbit (orbit 2);
# (b) the v (apogee kick) required at B of the transfer orbit to
#   establish a circular orbit of 16000km altitude (orbit 3)
# (c) total propellant if specific impulse is 300s

r_ea = 6378  # earth radius [km]
mu_e = 3.986e5  # earth mu [km^3/s^2]
mass_sc = 2000  # [kg]
specImp = 300  # specific impulse [s]

orbit1_peri = 480 + r_ea
orbit1_apo = 800 + r_ea

orbit2_peri = 480 + r_ea
orbit2_apo = 16000 + r_ea

orbit3_peri = 16000 + r_ea
orbit3_apo = 16000 + r_ea


def energy_ellipse(peri, apo, mu):
    a = (peri + apo) / 2
    return -1 * mu / (2 * a)


orbit1_energy = energy_ellipse(orbit1_peri, orbit1_apo, mu_e)
orbit2_energy = energy_ellipse(orbit2_peri, orbit2_apo, mu_e)
orbit3_energy = energy_ellipse(orbit3_peri, orbit3_apo, mu_e)

de_1 = orbit2_energy - orbit1_energy
de_2 = orbit3_energy - orbit2_energy

e_o1 = (orbit1_apo - orbit1_peri) / (orbit1_peri + orbit1_apo)
h_o1 = np.sqrt(orbit1_peri * mu_e * (1 + e_o1))
# v_peri_o1 = h_o1/orbit1_peri


def v_ellipse_peri(peri, apo, mu):
    e = (apo - peri) / (apo + peri)
    h = np.sqrt(peri * mu * (1 + e))
    v_peri = h / peri
    return v_peri


def v_ellipse_apo(peri, apo, mu):
    e = (apo - peri) / (apo + peri)
    h = np.sqrt(peri * mu * (1 + e))
    v_apo = h / apo
    return v_apo


def v_circle(r, mu):
    return np.sqrt(mu / r)


# part a
v_peri_o1 = v_ellipse_peri(orbit1_peri, orbit1_apo, mu_e)
v_peri_o2 = v_ellipse_peri(orbit2_peri, orbit2_apo, mu_e)
dv_1 = v_peri_o2 - v_peri_o1

# part b
v_apo_o2 = v_ellipse_apo(orbit2_peri, orbit2_apo, mu_e)
v_o3 = v_circle(orbit3_apo, mu_e)
dv_2 = v_o3 - v_apo_o2
total_dv = dv_1 + dv_2  # [km/s]
#   remember to manage units; convert specific impulse defined for 9.807 [m/s^2] not [km/s^2]
total_dv_m = total_dv * 1000  # convert -> [m/s]

# part c;
delta_mass = mass_sc * (1 - np.exp(-total_dv_m / (specImp * 9.807)))

print("Hohmann Transfer:")
print("delta v1", dv_1)
print("delta v2", dv_2)
print("total delta v", total_dv)
print("propellant mass", delta_mass)
