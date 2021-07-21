#Solver for Lagrangian points L1, L2, L3 given mass ratio pi_2
#pi_2 = m2/(m1 + m2)
#Solver returns eta = x/r12
#To return actual positions, x = eta*r12

import scipy.optimize
import numpy as np

def zero_solver(eta, mass_ratio):
    A = (1 - mass_ratio)/(np.absolute(eta + mass_ratio))**3
    B = eta + mass_ratio
    C = (mass_ratio)/(np.absolute(eta + mass_ratio - 1))**3
    D = eta + mass_ratio - 1
    E = -1*eta
    return A*B + C*D + E
    
def find_eta(pi):
    roots = scipy.optimize.fsolve(zero_solver, x0 = 1, args = pi)
    return roots

def find_lagrangians(mass_ratio):
    #L1:
    L1 = scipy.optimize.fsolve(zero_solver, x0 = 0.5, args = mass_ratio)
    #L2:
    L2 = scipy.optimize.fsolve(zero_solver, x0 = 1.5, args = mass_ratio)
    #L3:
    L3 = scipy.optimize.fsolve(zero_solver, x0 = -1, args = mass_ratio)
    return [L1[0], L2[0], L3[0]]
    