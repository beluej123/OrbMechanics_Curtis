# Curtis example 6.2, p.326 in my book
#   based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#   by Howard D. Curtis
# Given: orbit transfer, central body=earth,
#   s/c approach; periapsis 5000km, vel 10km/s; s/c final, circular, 500km
#  Find:
import numpy as np

# A spacecraft returning from a lunar mission approaches earth on a hyperbolic
# trajectory. At its closest approach A it is at an altitude of 5000 km,
# traveling at 10 km/s. At A retrorockets are fired to lower the spacecraft
# into a 500 km altitude circular orbit, where it is to rendezvous with a
# space station. Find the location of the space station
# at retrofire so that rendezvous will occur at B.

r_ea = 6378  # earth radius [km]
mu_e = 3.986e5  # earth mu [km^3/s^2]
r_hyp = 5000 + r_ea  # [km]
v_hyp = 10  # [km/s]

ra_o2 = 5000 + r_ea
rp_o2 = 500 + r_ea

a_o2 = (ra_o2 + rp_o2) / 2

T_o2 = ((2 * np.pi) / np.sqrt(mu_e)) * a_o2**1.5

time_taken = T_o2 / 2

T_o3 = ((2 * np.pi) / np.sqrt(mu_e)) * rp_o2**1.5

orbital_portion = time_taken / T_o3
orbital_angle = orbital_portion * 360
print("Hyperbolic approach, Hohmann transfer:")
print("given; hyperbolic closest:", r_hyp, "[km]")
print("given; hyperbolic velocity @ r_hyp:", v_hyp, "[km/s]")
print("given; inner orbit altitude:", ra_o2, "[deg]")
print("\norbit transfer time:", time_taken, "[s]")
print("randevous phasing:", orbital_angle, "[deg]")
