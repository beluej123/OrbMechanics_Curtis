import numpy as np
import scipy.optimize

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
        
def vector_from_orbit_elements(h, e, theta, ra_node, incl, arg_p, mu):
    #convert from deg to rad here
    rad_conv = np.pi/180
    theta *= rad_conv
    incl *= rad_conv
    arg_p *= rad_conv
    ra_node *= rad_conv
    
    #find r, v in perifocal
    r_vector_peri = r_vector_perifocal(theta, h, mu, e)
    v_vector_peri = v_vector_perifocal(theta, h, mu, e)
    
    Q = np.transpose(geo_to_peri(arg_p, incl, ra_node))
    
    r_vector_geo = Q @ r_vector_peri
    v_vector_geo = Q @ v_vector_peri
    return r_vector_geo, v_vector_geo
    





