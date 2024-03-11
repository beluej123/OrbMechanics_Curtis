# Curtis example 5.3, p.273 in my book; also see Orbit_from_r0v0.py
#   based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#   by Howard D. Curtis
# Given earth altitude's r1 & r2 (not vector), dt, delta thue anomaly; find
#   periapsis altitude, time to periapsis
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# A meteoroid is sighted at an altitude of 267,000 km.
# 13.5 hours later, after a change in true anomaly of 5◦,
# the altitude is observed to be 140 000 km. Calculate the perigee
# altitude and the time to perigee after the second sighting.

# Functions needed:


# Main calcs:
