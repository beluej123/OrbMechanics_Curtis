# Curtis example's 3.1 (p.x)
#   based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#   by Howard D. Curtis
import numpy as np
import scipy.optimize

##Example 3.1
# A geocentric elliptical orbit has a perigee radius of 9600 km
# and an apogee radius of 21 000 km.
# Calculate the time to fly from perigee P to a true anomaly of 120◦.

peri = 9600
apo = 21000
theta = 120 * (2 * np.pi / 360)

# mu
mu = 398600

# Find e
e = (apo - peri) / (apo + peri)

# find h
h = np.sqrt(peri * mu * (1 + e))

# find total period T
T = (2 * np.pi / mu**2) * (h / np.sqrt(1 - e**2)) ** 3


# find E
def calculate_E(e, theta):
    A = np.sqrt((1 - e) / (1 + e))
    B = np.tan(theta / 2)
    return 2 * np.arctan(A * B)


# eccentric anomaly
E = calculate_E(e, theta)

# mean anomaly
Me = E - e * np.sin(E)

# time since periapsis
time_sp = Me * T / (2 * np.pi)
print(time_sp)

##Example 3.2
# Find the true anomaly at three hours after perigee passage.
# Since the time (10 800 seconds) is greater than one-half the period,
# the true anomaly must be greater than 180◦.

time_3h = 3 * 60 * 60

Me_3h = time_3h * 2 * np.pi / T


def E_zerosolver(E, args):
    Me = args[0]
    e = args[1]
    return E - e * np.sin(E) - Me


def solve_for_E(Me, e):
    sols = scipy.optimize.fsolve(E_zerosolver, x0=Me, args=[Me, e])
    return sols


E_3h = solve_for_E(Me_3h, e)[0]

theta_3h = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E_3h / 2))

theta_3h_degrees = (180 / np.pi) * theta_3h + 360
print(theta_3h_degrees)
