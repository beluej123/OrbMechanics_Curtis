import numpy as np

#A spacecraft returning from a lunar mission approaches earth on a hyperbolic
#trajectory. At its closest approach A it is at an altitude of 5000 km,
#traveling at 10 km/s. At A retrorockets are fired to lower the spacecraft
#into a 500 km altitude circular orbit, where it is to rendezvous with a
#space station. Find the location of the space station
#at retrofire so that rendezvous will occur at B.

mu = 398600
r_hyp = 5000 + 6378
v_hyp = 10

ra_o2 = 5000 + 6378
rp_o2 = 500 + 6378

a_o2 = (ra_o2 + rp_o2)/2

T_o2 = ((2*np.pi)/np.sqrt(mu))*a_o2**1.5

time_taken = T_o2/2

T_o3 = ((2*np.pi)/np.sqrt(mu))*rp_o2**1.5

orbital_portion = time_taken/T_o3
orbital_angle = orbital_portion*360