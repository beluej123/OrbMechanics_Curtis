import numpy as np
import scipy.optimize

#Example 3.6
#An earth satellite has an initial true anomaly of theta_0 = 30◦,
#a radius of r0 = 10 000 km, and a speed of v0 = 10 km/s.
#Use the universal Kepler’s equation to find the change in
#universal anomaly χ after one hour and use that information
#to determine the true anomaly theta at that time.

theta_0 = 30*(np.pi/180)
r_0 = 10000
mu = 398600
v_0 = 10

t_1h = 3600

def orbit_equation_h(r, mu, e, theta):
    A = r*mu
    B = 1 + e*np.cos(theta)
    return np.sqrt(A*B)

#Solve for e; working in example

def e_zerosolver(e, args):
    v_0, r_0, mu, theta_0 = args
    
    A = v_0
    B = orbit_equation_h(r_0, mu, e, theta_0)/r_0
    C = (mu*e*np.sin(theta_0))/(orbit_equation_h(r_0, mu, e, theta_0))
    return A**2 - B**2 - C**2

e = scipy.optimize.fsolve(e_zerosolver, x0 = 1,
                          args = [v_0, r_0, mu, theta_0])[0]

h = orbit_equation_h(r_0, mu, e, theta_0)

vr_0 = (mu*e*np.sin(theta_0))/h

#Universal kepler equation

def stumpff_S(z):
    if z > 0:
        x = np.sqrt(z)
        return (x - np.sin(x))/(x)**3
    elif z < 0:
        y = np.sqrt(-z)
        return (np.sinh(y) - y)/(y)**3
    else:
        return (1/6)
        
def stumpff_C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z)))/z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1)/(-z)
    else:
        return(1/2)
        
#Find semimajor axis
a_orbit = 1/(2/r_0 - (v_0)**2/mu)

#Find initial F
def F_from_theta(e, theta):
    A = np.tan(theta/2)
    B = np.sqrt((e-1)/(e+1))
    return 2*np.arctanh(A*B)

F_0 = F_from_theta(e, theta_0)

#Use Universal Kepler to find delta x

def universalx_zerosolver(x, args):
    r0, vr0, mu, dt, a = args
    
    A = stumpff_C((x**2)/a)*((r0*vr0)/(np.sqrt(mu)))*(x**2)
    B = stumpff_S((x**2)/a)*(1 - r0/a)*(x**3)
    C = r0*x
    D = np.sqrt(mu)*dt
    return A + B + C - D

x0_guess = t_1h*np.sqrt(mu)*np.absolute(1/a_orbit)

x_1h = scipy.optimize.fsolve(universalx_zerosolver, x0 = x0_guess,
                             args = [r_0, vr_0, mu, t_1h, a_orbit])[0]

F_1h = F_0 + x_1h/np.sqrt(-a_orbit)

def F_to_theta(e, F):
    A = np.sqrt((e+1)/(e-1))
    B = np.tanh(F/2)
    return 2*np.arctan(A*B)

theta_1h = F_to_theta(e, F_1h)
theta_1h_deg = theta_1h*(180/np.pi)




