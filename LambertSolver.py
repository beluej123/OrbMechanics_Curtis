# Curtis part of algorithm 5.2 (p.263+; p.270 for example in my book). H.W. Curtis
# Orbital Mechanics for Engineering Students, 2nd ed., 2009
# Given r1_mag, r2_mag, and dt;
# Find v1 & v2, and orbital elements;
#   Note Gauss problem, or Lamberts theory, and solution
import time

import numpy as np
import scipy.optimize


# Auxiliary functions
def stumpff_S(z):
    if z > 0:
        x = np.sqrt(z)
        return (x - np.sin(x)) / (x) ** 3
    elif z < 0:
        y = np.sqrt(-z)
        return (np.sinh(y) - y) / (y) ** 3
    else:
        return 1 / 6


def stumpff_C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 1 / 2


def y_lambert(z, r1_mag, r2_mag, A):
    K = (z * stumpff_S(z) - 1) / np.sqrt(stumpff_C(z))
    return r1_mag + r2_mag + A * K


def A_lambert(r1_mag, r2_mag, d_theta):
    K1 = np.sin(d_theta)
    K2 = np.sqrt((r1_mag * r2_mag) / (1 - np.cos(d_theta)))
    return K1 * K2


def lambert_zerosolver(z, args):
    dt, mu, r1_mag, r2_mag, A = args
    K1 = ((y_lambert(z, r1_mag, r2_mag, A) / stumpff_C(z)) ** 1.5) * stumpff_S(z)
    K2 = A * np.sqrt(y_lambert(z, r1_mag, r2_mag, A))
    K3 = -1 * dt * np.sqrt(mu)
    return K1 + K2 + K3


def find_f_y(y, r1_mag):
    return 1 - y / r1_mag


def find_g_y(y, A, mu):
    return A * np.sqrt(y / mu)


def find_f_dot_y(y, r1_mag, r2_mag, mu, z):
    K1 = np.sqrt(mu) / (r1_mag * r2_mag)
    K2 = np.sqrt(y / stumpff_C(z))
    K3 = z * stumpff_S(z) - 1
    return K1 * K2 * K3


def find_g_dot_y(y, r2_mag):
    return 1 - y / r2_mag


# note expected Braeunig subdirectory
# trouble with relative imports
# https://stackoverflow.com/questions/77091688/relative-python-imports-what-is-the-difference-between-single-dot-and-no-dot/77120636#77120636
try:
    from .Braeunig.validations_1 import assert_parameters_are_valid
except ImportError:
    from Braeunig.validations_1 import assert_parameters_are_valid


# Lambert Solver
# Assumes prograde trajectory, can be changed in function call
def Lambert_v1v2_solver(r1_v, r2_v, dt, mu, prograde=True, M=0):
    """
    Given position vectors r1_v, r2_v, and the delta-time, calculate required
    velocity vectors, v1 and v2.

    Parameters
    ----------
    r1_mag: numpy.array
        Initial position vector.
    r2_mag: numpy.array
        Final position vector.
    dt: float
        delta time between r1_mag & r2_mag; aka time of flight (tof).
    mu: float
        Gravitational parameter (GM) of attractor body.
    prograde: bool
        If `True`, specifies prograde motion. Otherwise, retrograde motion is imposed.
    M: int
        Number of revolutions; M >= 0, default 0.

    Returns
    -------
    v1_v: numpy.array
        Initial velocity vector.
    v2_v: numpy.array
        Final velocity vector.
    tti: float
        Time to iteration in seconds.

    Notes
    -----
    This algorithm is presented as algorithm 5.2 (p.270 for example in my book)
    by H.W. Curtis following Bond and Allman (1996).  Note other algorythms by
    BMWS (2020), Vallado (2013)
    in 1971 (or BMWS 2020).

    References
    ----------
    [1] Bond and Allman(1996), ??.
    [2] BMWS; Bate, R. R., Mueller, D. D., White, J. E., & Saylor, W. W. (2020).
    Fundamentals of astrodynamics. Courier Dover Publications.
    [3] Vallado, D. A. (2013, 4th ed.). Fundamentals of Astrodynamics and Applications
    Microcosm Press, Hawthorn, Ca. Section 7.6, pp.467+
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
    A = A_lambert(r1_mag, r2_mag, d_theta)

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
    # Test Braeunig problem 5.2, below
    print(f"Test Curtis LambertSolver; with Braeunig parameters:")
    # Solar system constants
    au = 149.597870e6  # [km/au], for unit conversions
    GM_sun_km = 132712.4e6  # [km^3/s^2] sun
    GM_sun_au = 3.964016e-14  # [au^3/s^2]
    GM_earth_km = 398600.5  # [km^3/s^2] earth
    GM_mars_km = 42828.31  # [km^3/s^2] mars
    GM_jup_km = 1.26686e8  # [km^3/s^2] jupiter

    tof = 207 * 24 * 60 * 60  # [s] time of flight
    dt = tof
    # **********************
    mu = GM_sun_au

    # Ecliptic coordinates
    r1_vec = np.array([0.473265, -0.899215, 0])  # [au]
    r1_mag = np.linalg.norm(r1_vec)
    r2_vec = np.array([0.066842, 1.561256, 0.030948])  # [au]
    r2_mag = np.linalg.norm(r2_vec)
    tof = 207 * 24 * 60 * 60  # [s] time of flight

    v1_vec, v2_vec, tti = Lambert_v1v2_solver(
        r1_v=r1_vec, r2_v=r2_vec, dt=tof, mu=GM_sun_au, prograde=True, M=0
    )
    print(f"v1_vec= {v1_vec*au} [km/s], v2_vec= {v2_vec*au} [km/s]")
    print(f"time to iterate to solution, tti= {tti:.8f} [s], computation performance")
    # Test Braeunig problem 5.2, above

    # Curtis example 5.2, below: parameters, p.270
    print(f"\nTest Curtis LambertSolver; with Curtis parameters:")
    r1_vec = np.array([5000, 10000, 2100])
    r2_vec = np.array([-14600, 2500, 7000])
    dt = 60 * 60  # time of flight between r1 and r2
    mu_earth_km = 3.986e5  # earth mu [km^3/s^2]

    v1_vec, v2_vec, tti = Lambert_v1v2_solver(r1_vec, r2_vec, dt, mu=mu_earth_km)
    print(f"v1_vec= {v1_vec} [km/s], v2_vec= {v2_vec} [km/s]")
    print(f"time to iterate to solution, tti= {tti:.8f} [s], computation performance")

    return None


def main():
    pass  # placeholder
    return None


# If you wish to test this Lambert solver.
if __name__ == "__main__":
    test_lambert_solver()  #

    # main()  # placeholder function
