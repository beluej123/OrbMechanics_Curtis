import numpy as np
import matplotlib.pyplot as pl


# Orbit Equation Plotter; polar projection
def orbit_equation_r(h, mu, e, theta):
    A = h**2 / mu
    B = 1 + e * np.cos(theta)
    return A / B


theta_array = np.linspace(0, 2 * np.pi, 200)
r_array = [orbit_equation_r(30, 40, 0, theta) for theta in theta_array]
r_array_1 = [orbit_equation_r(30, 40, 0.2, theta) for theta in theta_array]

fig, ax = pl.subplots(subplot_kw={"projection": "polar"})
ax.plot(theta_array, r_array)
ax.plot(theta_array, r_array_1)
ax.set_rmax(1.4 * max((r_array)))
pl.show()
