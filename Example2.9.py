import numpy as np

#constants:
mu = 398600

###Problem statement: r0 v0 given, find r and v after 120deg

r0_vector = np.array([8182.4, -6865.9, 0])
v0_vector = np.array([0.47572, 8.8116, 0])
d_theta = 120*(2*np.pi/360)

#write up formulas

def find_r(h, mu, r0, vr0, d_theta):
    A = h**2/mu
    B = ((h**2/(mu*r0))-1)*(np.cos(d_theta))
    C = -(h*vr0/mu)*(np.sin(d_theta))
    return A*(1/(1+B+C))

def find_f(mu, r, h, d_theta):
    A = mu*r/(h**2)
    B = 1 - np.cos(d_theta)
    return 1 - A*B

def find_g(r, r0, h, d_theta):
    A = r*r0/h
    B = np.sin(d_theta)
    return A*B

def find_f_dot(mu, h, d_theta, r0, r):
    A = mu/h
    B = (1-np.cos(d_theta))/(np.sin(d_theta))
    C = mu/(h**2)
    D = 1-np.cos(d_theta)
    E = 1/r0
    F = 1/r
    return A*B*(C*D-E-F)
    
def find_g_dot(mu, r0, h, d_theta):
    A = mu*r0/(h**2)
    B = 1 - np.cos(d_theta)
    return 1 - A*B
    
def find_position(f, g, r0, v0):
    return f*r0 + g*v0
    
def find_velocity(f_dot, g_dot, r0, v0):
    return f_dot*r0 + g_dot*v0

###solve problem

#Calculate h (constant), r0, vr0
h = np.linalg.norm(np.cross(r0_vector, v0_vector))
r0 = np.linalg.norm(r0_vector)
vr0 = np.dot(v0_vector, (r0_vector/r0))

r = find_r(h, mu, r0, vr0, d_theta)

f = find_f(mu, r, h, d_theta)
g = find_g(r, r0, h, d_theta)
f_dot = find_f_dot(mu, h, d_theta, r0, r)
g_dot = find_g_dot(mu, r0, h, d_theta)

final_position = find_position(f, g, r0_vector, v0_vector)
final_velocity = find_velocity(f_dot, g_dot, r0_vector, v0_vector)

    
    
    
    
    
    
    
    
    
    
    
    
    