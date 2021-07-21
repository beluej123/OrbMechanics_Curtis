import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#At time t0, the state vector of an Earth satellite is

r0_vector = np.array([1600, 5310, 3800])
v0_vector = np.array([-7.350, 0.4600, 2.470])

#Determine the position and velocity 3200 seconds later
#and plot the orbit in three dimensions.

mu = 398600
dt = 3200

#Use Algorithm 3.4
r0 = np.linalg.norm(r0_vector)
v0 = np.linalg.norm(v0_vector)

vr0 = np.dot(r0_vector, v0_vector)/r0

#semimajor axis
a_orbit = 1/((2/r0) - (v0**2/mu))

#Find x
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

def universalx_zerosolver(x, args):
    r0, vr0, mu, dt, a = args
    
    A = stumpff_C((x**2)/a)*((r0*vr0)/(np.sqrt(mu)))*(x**2)
    B = stumpff_S((x**2)/a)*(1 - r0/a)*(x**3)
    C = r0*x
    D = np.sqrt(mu)*dt
    return A + B + C - D

x0_guess = dt*np.sqrt(mu)*np.absolute(1/a_orbit)

x_dt = scipy.optimize.fsolve(universalx_zerosolver, x0 = x0_guess,
                             args = [r0, vr0, mu, dt, a_orbit])[0]


#write f,g functions for x

def find_f_x(x, r0, a):
    A = x**2/r0
    B = stumpff_C(x**2/a)
    return 1 - A*B

def find_g_x(x, dt, mu, a):
    A = x**3/np.sqrt(mu)
    return dt - A*stumpff_S(x**2/a)

def find_f_dot_x(x, mu, r, r0, a):
    A = np.sqrt(mu)/(r*r0)
    B = stumpff_S(x**2/a)*(x**3/a)
    return A*(B - x)

def find_g_dot_x(x, r, a):
    A = x**2/r
    return 1 - A*stumpff_C(x**2/a)

f_dt = find_f_x(x_dt, r0, a_orbit)
g_dt = find_g_x(x_dt, dt, mu, a_orbit)

r_dt_vector = f_dt*r0_vector + g_dt*v0_vector
r_dt = np.linalg.norm(r_dt_vector)

f_dot_dt = find_f_dot_x(x_dt, mu, r_dt, r0, a_orbit)
g_dot_dt = find_g_dot_x(x_dt, r_dt, a_orbit)

v_dt_vector = f_dot_dt*r0_vector + g_dot_dt*v0_vector
g_dt = np.linalg.norm(v_dt_vector)

#Plot the orbit

#First, find x_max
x_max = np.sqrt(a_orbit)*(2*np.pi)

#Create a linspace for x, set resolution
resolution = 1000
x_array = np.linspace(0, x_max, resolution)

#Find delta_t for each x

def dt_from_x(x, args):
    r0, vr0, mu, a = args
    #Equation 3.46
    A = (r0*vr0/np.sqrt(mu))*(x**2)*stumpff_C(x**2/a)
    B = (1 - r0/a)*(x**3)*stumpff_S(x**2/a)
    C = r0*x
    LHS = A + B + C
    return LHS/np.sqrt(mu)
    
def r_from_x(r0_vector, v0_vector, x, dt, a, mu):
    r0 = np.linalg.norm(r0_vector)
    f = find_f_x(x, r0, a)
    g = find_g_x(x, dt, mu, a)
    return f*r0_vector + g*v0_vector

pos_array = np.array([r_from_x(r0_vector, v0_vector,
                               x, dt_from_x(x, [r0, vr0, mu, a_orbit]),
                               a_orbit, mu) for x in x_array])
    
#plot in 3d
fig = plt.figure(dpi = 120)
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])
ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], 'x')
ax.plot([r_dt_vector[0]], [r_dt_vector[1]], [r_dt_vector[2]], 'x')





