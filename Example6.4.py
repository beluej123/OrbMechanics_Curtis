import numpy as np

#Functions

def E_from_theta(e, theta):
    A = np.sqrt((1-e)/(1+e))
    B = np.tan(theta/2)
    return 2*np.arctan(A*B)

def orbit_equation_h(r, mu, e, theta):
    A = r*mu
    B = 1 + e*np.cos(theta)
    return np.sqrt(A*B)

def t_from_Me(Me, mu, h, e):
    A = Me
    B = (mu**2)/(h**3)
    C = (1 - e**2)**1.5
    return A/(B*C)

def t_ellipse(r_p, r_a, mu):
    a = (r_a + r_p)/2
    return ((2*np.pi)/np.sqrt(mu))*a**1.5

def v_ellipse_peri(peri, apo, mu):
    e = (apo - peri)/(apo + peri)
    h = np.sqrt(peri*mu*(1+e))
    v_peri = h/peri
    return v_peri

#Spacecraft at A and B are in the same orbit (1).
#At the instant shown, the chaser vehicle at A executes a phasing
#maneuver so as to catch the target spacecraft back at A after just
#one revolution of the chaser’s phasing orbit (2).
#What is the required total delta-v?

#Initial orbit is an ellipse given by A and C.
#Phasing orbit (of A) reduces the apogee to D
#Inital difference in true anomaly is 90 deg

r_a = 6800
r_c = 13600
mu_e = 398600
d_theta = 90*(np.pi/180)

#Part 1: what is the time difference (for anomaly of 90deg)

e_o1 = (r_c - r_a)/(r_c + r_a)

E_B = E_from_theta(e_o1, d_theta)
Me_B = E_B - e_o1*np.sin(E_B)
h_A = orbit_equation_h(r_a, mu_e, e_o1, 0)
T_o1 = t_ellipse(r_a, r_c, mu_e)

dt = t_from_Me(Me_B, mu_e, h_A, e_o1)

T_phase = T_o1 - dt

a_phase = (T_phase*np.sqrt(mu_e)/(2*np.pi))**(2/3)

ra_phase = 2*a_phase - r_a

v_o1 = v_ellipse_peri(r_a, r_c, mu_e)
v_phase = v_ellipse_peri(r_a, ra_phase, mu_e)
delta_v = v_o1 - v_phase

#Total maneuver is double
total_delta_v = 2*delta_v


