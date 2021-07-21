import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Required functions:

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

def e_from_r0v0(r0_v, v0_v, mu):
    r0_vector = np.array(r0_v)
    v0_vector = np.array(v0_v)
    
    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)
    vr0 = np.dot(r0_vector, v0_vector)/r0

    #Find eccentricity
    A = (v0**2 - (mu/r0))*r0_vector
    B = -r0*vr0*v0_vector
    e_vector = (1/mu)*(A + B)
    e = np.linalg.norm(e_vector)
    return e

#Actual functions:

def orbit_r0v0(r0_v, v0_v, mu, resolution=1000, hyp_span=1):
    r0_vector = np.array(r0_v)
    v0_vector = np.array(v0_v)
    
    #Use Algorithm 3.4
    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)

    vr0 = np.dot(r0_vector, v0_vector)/r0
    a_orbit = 1/((2/r0) - (v0**2/mu))
    
    #Check for orbit type, define x_range
    #resolution = number of points plotted
    #span = width of parabolic orbit plotted\

    e = e_from_r0v0(r0_v, v0_v, mu)
    if e >= 1:
        x_max = np.sqrt(np.abs(a_orbit))
        x_array = np.linspace(-hyp_span*x_max, hyp_span*x_max, resolution)
        pos_array = np.array([r_from_x(r0_vector, v0_vector,
                              x, dt_from_x(x, [r0, vr0, mu, a_orbit]),
                              a_orbit, mu) for x in x_array])         
        
    else:
        x_max = np.sqrt(a_orbit)*(2*np.pi)
        x_array = np.linspace(0, x_max, resolution)
        pos_array = np.array([r_from_x(r0_vector, v0_vector,
                              x, dt_from_x(x, [r0, vr0, mu, a_orbit]),
                              a_orbit, mu) for x in x_array]) 
    
    return pos_array

def maneuver_plot(r0_v, v0_v, dv_v, mu, resolution=1000, hyp_span=1):
    #Plot initial orbit
    initial_orbit = orbit_r0v0(r0_v, v0_v, mu, resolution=resolution,
                               hyp_span=hyp_span)
      
    fig = plt.figure(dpi = 120)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(initial_orbit[:, 0], initial_orbit[:, 1],
            initial_orbit[:, 2])
    ax.plot([np.array(r0_v)[0]], [np.array(r0_v)[1]],
             [np.array(r0_v)[2]], '.')
    ax.plot([0], [0], [0], 'o', color='k') 
    
    #Find new orbit
    v0_dv = np.array(v0_v) + np.array(dv_v)
    new_orbit = orbit_r0v0(r0_v, v0_dv, mu, resolution=resolution,
                           hyp_span=hyp_span)
    
    #Plot new orbit 
    ax.plot(new_orbit[:, 0], new_orbit[:, 1],
            new_orbit[:, 2])

######

#Units of r0 in km, v0 in km/s, mu in km3/s2
#Change units as necessary (all consistent)
#mu is G*M, m mass of primary body, G is gravitational constant

maneuver_plot([203, -30, 0],
              [-18, 33, 0],
              [23, 1, -30],
              344000, hyp_span=5)

