import numpy as np
import scipy.optimize

#Example 3.5
#A geocentric trajectory has a perigee velocity of 15 km/s and
#a perigee altitude of 300 km. Find:

r_p = 300 + 6378
v_p = 15
mu = 398600

#(a) the radius when the true anomaly is 100◦ 
theta_a = 100*(np.pi/180)

#find h
h = r_p*v_p

#find e from orbit equation
e = (1/np.cos(0))*((h**2/(r_p*mu)) - 1)

#use e to find r with orbit equation

def orbit_equation_r(h, mu, e, theta):
    A = h**2/mu
    B = 1 + e*np.cos(theta)
    return A/B
    
r_100deg = orbit_equation_r(h, mu, e, theta_a)

#(b) the position and speed three hours later. (after 100 deg)

def F_from_theta(e, theta):
    A = np.tan(theta/2)
    B = np.sqrt((e-1)/(e+1))
    return 2*np.arctanh(A*B)

F_a = F_from_theta(e, theta_a)

Mh_a = e*np.sinh(F_a) - F_a

t_a = Mh_a*(h**3/mu**2)*((e**2 - 1)**(-3/2))

t_3h = 3*60*60 + t_a

Mh_3h = (mu**2/h**3)*((e**2-1)**1.5)*t_3h

def F_zerosolver(F, args):
    Mh = args[0]
    e = args[1]
    return - F + e*np.sinh(F) - Mh

def solve_for_F(Mh, e):
    sols = scipy.optimize.fsolve(F_zerosolver, x0 = Mh, args = [Mh, e])
    return sols

F_3h = solve_for_F(Mh_3h, e)[0]

def F_to_theta(e, F):
    A = np.sqrt((e+1)/(e-1))
    B = np.tanh(F/2)
    return 2*np.arctan(A*B)

theta_3h = F_to_theta(e, F_3h) 
theta_3h_deg = (180/np.pi)*theta_3h

r_3h = orbit_equation_r(h, mu, e, theta_3h)
v_3h_tangent = h/r_3h
v_3h_radial = (mu/h)*e*np.sin(theta_3h)
v_3h = np.linalg.norm([v_3h_tangent, v_3h_radial])







