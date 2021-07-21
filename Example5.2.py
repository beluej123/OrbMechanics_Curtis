import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Given r1, r2, and dt, find orbital elements

#Auxiliary functions

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

def y_lambert(z, r1, r2, A):
    K = (z*stumpff_S(z) - 1)/np.sqrt(stumpff_C(z))
    return r1 + r2 + A*K

def A_lambert(r1, r2, d_theta):
    K1 = np.sin(d_theta)
    K2 = np.sqrt((r1*r2)/(1 - np.cos(d_theta)))
    return K1*K2

def lambert_zerosolver(z, args):
    dt, mu, r1, r2, A = args
    K1 = ((y_lambert(z, r1, r2, A)/stumpff_C(z))**1.5)*stumpff_S(z)
    K2 = A*np.sqrt(y_lambert(z, r1, r2, A))
    K3 = -1*dt*np.sqrt(mu)
    return K1 + K2 + K3

def find_f_y(y, r1):
    return 1 - y/r1

def find_g_y(y, A, mu):
    return A*np.sqrt(y/mu)

def find_f_dot_y(y, r1, r2, mu, z):
    K1 = np.sqrt(mu)/(r1*r2)
    K2 = np.sqrt(y/stumpff_C(z))
    K3 = z*stumpff_S(z) - 1
    return K1*K2*K3

def find_g_dot_y(y, r2):
    return 1 - y/r2


#Main function
#Assumes prograde trajectory, can be changed in function

def Lambert_v1v2_solver(r1_v, r2_v, dt, mu, prograde=True):
    r1 = np.linalg.norm(r1_v)
    r2 = np.linalg.norm(r2_v)
    
    r1r2z = np.cross(r1_v, r2_v)[2]
    cos_calc = np.dot(r1_v, r2_v)/(r1*r2)
    
    if prograde:
        if r1r2z < 0:
            d_theta = 2*np.pi - np.arccos(cos_calc)
        else:
            d_theta = np.arccos(cos_calc)
    else:
        if r1r2z < 0:
            d_theta = np.arccos(cos_calc)
        else:
            d_theta = 2*np.pi - np.arccos(cos_calc)        
    
    A = A_lambert(r1, r2, d_theta)
    
    #find a way to get a good starting estimate
    z = scipy.optimize.fsolve(lambert_zerosolver, x0 = 1.5,
                                    args = [dt, mu, r1, r2, A])[0]
    
    y = y_lambert(z, r1, r2, A)
    
    f_dt = find_f_y(y, r1)
    g_dt = find_g_y(y, A, mu)
    f_dot_dt = find_f_dot_y(y, r1, r2, mu, z)
    g_dot_dt = find_g_dot_y(y, r2)
    
    v1_v = (1/g_dt)*(r2_v - f_dt*r1_v)
    v2_v = (g_dot_dt/g_dt)*r2_v - ((f_dt*g_dot_dt - f_dot_dt*g_dt)/g_dt)*r1_v
    
    return v1_v, v2_v

#Auxiliary functions
def R1(angle):
    A = [1, 0, 0]
    B = [0, np.cos(angle), np.sin(angle)]
    C = [0, -1*np.sin(angle), np.cos(angle)]
    return [A, B, C]

def R3(angle):
    A = [np.cos(angle), np.sin(angle), 0]
    B = [-1*np.sin(angle), np.cos(angle), 0]
    C = [0, 0, 1]
    return [A, B, C]

def r_vector_perifocal(theta, h, mu, e):
    A = h**2/mu
    B = 1 + e*np.cos(theta)
    C = np.array([np.cos(theta), np.sin(theta), 0])
    return (A/B)*C

def v_vector_perifocal(theta, h, mu, e):
    A = mu/h
    B = np.array([-1*np.sin(theta), e + np.cos(theta), 0])
    return A*B
    
def geo_to_peri(arg_p, incl, ra_node):
    A = np.array(R3(arg_p))
    B = np.array(R1(incl))
    C = np.array(R3(ra_node))
    return A @ B @ C

#Main functions
def orbit_elements_from_vector(r0_v, v0_v, mu):
    r0_vector = np.array(r0_v)
    v0_vector = np.array(v0_v)
    
    r0 = np.linalg.norm(r0_vector)
    v0 = np.linalg.norm(v0_vector)
    vr0 = np.dot(r0_vector, v0_vector)/r0
    #Find h
    h_vector = np.cross(r0_vector, v0_vector)
    h = np.linalg.norm(h_vector)
    #Find inclination
    incl = np.arccos(h_vector[2]/h)
    #Find node vector
    N_vector = np.cross([0, 0, 1], h_vector)
    N = np.linalg.norm(N_vector)
    #Find right ascension of ascending node
    if N_vector[1] < 0:
        ra_node = 2*np.pi - np.arccos(N_vector[0]/N)
    else:
        ra_node = np.arccos(N_vector[0]/N)
    #Find eccentricity
    A = (v0**2 - (mu/r0))*r0_vector
    B = -r0*vr0*v0_vector
    e_vector = (1/mu)*(A + B)
    e = np.linalg.norm(e_vector)
    #Find argument of perigee
    if e_vector[2] < 0:
        arg_p = 2*np.pi - np.arccos(np.dot(N_vector, e_vector)/(N*e))
    else:
        arg_p = np.arccos(np.dot(N_vector, e_vector)/(N*e))
    #Find true anomaly:
    if vr0 < 0:
        theta = 2*np.pi - np.arccos(np.dot(e_vector, r0_vector)/(e*r0))
    else:
        theta = np.arccos(np.dot(e_vector, r0_vector)/(e*r0))
        
    #Convert to degrees (can change units here)    
    deg_conv = 180/np.pi
    theta *= deg_conv
    incl *= deg_conv
    arg_p *= deg_conv
    ra_node *= deg_conv
    
    return [h, e, theta, ra_node, incl, arg_p]

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

#Actual function:
    
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

def plot_orbit_r0v0(r0_v, v0_v, mu, resolution=1000, hyp_span=1):
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
        
    #plot in 3d
    fig = plt.figure(dpi = 120)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])
    ax.plot([r0_vector[0]], [r0_vector[1]], [r0_vector[2]], '.')
    ax.plot([0], [0], [0], 'o', color='k')



#####

r1 = np.array([5000, 10000, 2100])
r2 = np.array([-14600, 2500, 7000])
dt = 60*60
mu = 398600

v1, v2 = Lambert_v1v2_solver(r1, r2, dt, mu)
orbit_els = orbit_elements_from_vector(r1, v1, mu)
plot_orbit_r0v0(r2, v2, mu, resolution=3000)
    