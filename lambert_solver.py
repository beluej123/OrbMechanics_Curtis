"""
Curtis [3] part of algorithm 5.2 (p.263+; p.270).
Curtis [9] algorithm 5.2. pp244 (appendix D.25, p.e71, lambert.m).

Given r1_vec, r2_vec, and dt;
Find v1 & v2, and orbital elements;
    Note Gauss problem, or Lamberts theory, and solution
References:
----------
    See references.py for references list.
"""

import time

import numpy as np
import scipy.optimize

from func_gen import (
    a_lambert,
    find_f_dot_y,
    find_f_y,
    find_g_dot_y,
    find_g_y,
    lambert_zerosolver,
    y_lambert,
)

# note expected Braeunig subdirectory
# trouble with relative imports
# https://stackoverflow.com/questions/77091688/relative-python-imports-what-is-the-difference-between-single-dot-and-no-dot/77120636#77120636
try:
    from .Braeunig.validations_1 import assert_parameters_are_valid
except ImportError:
    from Braeunig.validations_1 import assert_parameters_are_valid


# Lambert Solver
# Prograde trajectory can be changed in function call
def lambert_v1v2_solver(r1_v, r2_v, dt, mu, prograde=True, M=0):
    """
    Given position vectors r1_vec, r2_vec, and the delta-time, calculate
    velocity vectors, v1 and v2.

    Input Args:
    ----------
        r1_vec   : numpy.array, initial position vector.
        r2_vec   : numpy.array, final position vector.
        dt       : float, delta time between r1_vec & r2_vec;
                    aka time of flight (tof).
        mu       : float, Central body gravitational parameter (GM)
        prograde : bool
            If `True`, specifies prograde motion. Otherwise, retrograde motion is imposed.
        M        : int, number of revolutions; M >= 0, default 0.

    Returns
    -------
        v1_v     : numpy.array, initial velocity vector.
        v2_v     : numpy.array, final velocity vector.
        tti      : float, time to iteration in seconds.

    Notes
    ----------
        Curtis [3] p.270, algorithm 5.2; follows work by Bond and Allman (1996).
        Other algorythms; BMWS [2], Vallado [2] (2013), Vallado [4]
    """

    # Verify input parameters are safe/valid
    assert_parameters_are_valid(mu, r1_v, r2_v, dt, M)

    # step 1:
    r1_mag = np.linalg.norm(r1_v)
    r2_mag = np.linalg.norm(r2_v)

    # step 2:
    r1r2z = np.cross(r1_v, r2_v)[2]
    cos_calc = np.dot(r1_v, r2_v) / (r1_mag * r2_mag)
    if prograde:
        if r1r2z < 0:
            d_theta = 2 * np.pi - np.arccos(cos_calc)
        else:
            d_theta = np.arccos(cos_calc)
    else:
        if r1r2z < 0:
            d_theta = np.arccos(cos_calc)
        else:
            d_theta = 2 * np.pi - np.arccos(cos_calc)

    # step 3:
    A = a_lambert(r1_mag, r2_mag, d_theta)

    # step 4:
    # find starting estimate & iterate
    start_time = time.perf_counter()  # measure iteration performance
    z = scipy.optimize.fsolve(
        lambert_zerosolver, x0=1.5, args=[dt, mu, r1_mag, r2_mag, A]
    )[0]
    end_time = time.perf_counter()
    tti = end_time - start_time  # time to iterate; can compare with other methods

    # step 5:
    y = y_lambert(z, r1_mag, r2_mag, A)

    # step 6:
    f_dt = find_f_y(y, r1_mag)
    g_dt = find_g_y(y, A, mu)
    f_dot_dt = find_f_dot_y(y, r1_mag, r2_mag, mu, z)
    g_dot_dt = find_g_dot_y(y, r2_mag)

    # step 7:
    v1_v = (1 / g_dt) * (r2_v - f_dt * r1_v)
    v2_v = (g_dot_dt / g_dt) * r2_v - (
        (f_dt * g_dot_dt - f_dot_dt * g_dt) / g_dt
    ) * r1_v

    return v1_v, v2_v, tti


def test_lambert_solver() -> None:
    """Test Braeunig problem 5.2, below"""
    print("Test Curtis lambert_sover; with Braeunig parameters:")
    # Solar system constants
    au = 149.597870e6  # [km/au], for unit conversions
    # GM_sun_km = 132712.4e6  # [km^3/s^2] sun
    GM_sun_au = 3.964016e-14  # [au^3/s^2]
    # GM_earth_km = 398600.5  # [km^3/s^2] earth
    # GM_mars_km = 42828.31  # [km^3/s^2] mars
    # GM_jup_km = 1.26686e8  # [km^3/s^2] jupiter

    tof = 207 * 24 * 60 * 60  # [s] time of flight
    dt = tof
    # **********************

    # Ecliptic coordinates
    r1_vec = np.array([0.473265, -0.899215, 0])  # [au]
    # r1_mag = np.linalg.norm(r1_vec)
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # [au]
    # r2_mag = np.linalg.norm(r2_vec)
    tof = 207 * 24 * 60 * 60  # [s] time of flight

    v1_vec, v2_vec, tti = lambert_v1v2_solver(
        r1_v=r1_vec, r2_v=r2_vec, dt=tof, mu=GM_sun_au, prograde=True, M=0
    )
    print(f"v1_vec= {v1_vec*au} [km/s], v2_vec= {v2_vec*au} [km/s]")
    print(f"time to iterate to solution, tti= {tti:.8f} [s], computation performance")
    # Test Braeunig problem 5.2, above

    # Curtis example 5.2, below: parameters, p.270
    print("\nTest Curtis lambert_sover; with Curtis parameters:")
    r1_vec = np.array([5000, 10000, 2100])
    r2_vec = np.array([-14600, 2500, 7000])
    dt = 60 * 60  # time of flight between r1 and r2
    mu_earth_km = 3.986e5  # earth mu [km^3/s^2]

    v1_vec, v2_vec, tti = lambert_v1v2_solver(r1_vec, r2_vec, dt, mu=mu_earth_km)
    print(f"v1_vec= {v1_vec} [km/s], v2_vec= {v2_vec} [km/s]")
    print(f"time to iterate to solution, tti= {tti:.8f} [s], computation performance")


def main():
    """placeholder"""
    return None


# If you wish to test this Lambert solver.
if __name__ == "__main__":
    test_lambert_solver()  #
