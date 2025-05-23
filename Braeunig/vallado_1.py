"""
Methods devised by David A. Vallado [3].
This file (filename changed) copied from LamberHub (vallado.py) via GitHub, 2024-August.
2024-August +, minor edits & ref updates by Jeff Belue.

    All credits of the implementation go to Juan Luis Cano Rodríguez and the
    poliastro development team, from which this routine inherits. Some changes
    were made to adapt it to `lamberthub` API.

    Copyright (c) 2012-2021 Juan Luis Cano Rodríguez and the poliastro
    development team.

References:
----------
    See references.py for references list.
    Additional note, Vallado (2013, 4th ed.) section 7.6, pp.467+
"""

import time

import numpy as np
from braeunigFunctions import c2, c3, get_transfer_angle
from validations_1 import assert_parameters_are_valid


def vallado2013(
    mu,
    r1,
    r2,
    tof,
    M=0,
    prograde=True,
    low_path=True,
    maxiter=100,
    atol=1e-5,
    rtol=1e-7,
    full_output=True,
):
    """
    Vallado's algorithm makes use of the universal formulation to solve for the
    Lambert's problem. By making use of a bisection method, it guarantees the
    convergence to the solution but the amount of iterations require
    dramatically increases.


    Notes
    -----
    This algorithm is presented as an alternative to the one developed by Bate
    in 1971 (or BMWS 2020). Bate does not impose a particular numerical solver
    for his algorithm but cited both bisection and Newton's one. However, for
    some values of the boundary problem, the initial guess might diverge if
    Newton's solver is used. That's why Vallado [2] decided to employ a bisection
    method instead. Although detrimental from the point of view of performance,
    this algorithm properly reaches solution in the majority of the cases.
    Input Parameters:
    ----------
    mu: float
        Gravitational parameter, equivalent to :math:`GM` of attractor body.
    r1: numpy.array
        Initial position vector.
    r2: numpy.array
        Final position vector.
    M: int
        Number of revolutions. Must be equal or greater than 0 value.
    prograde: bool
        If `True`, specifies prograde motion. Otherwise, retrograde motion is imposed.
    low_path: bool
        If two solutions are available, it selects between high or low path.
    maxiter: int
        Maximum number of iterations.
    atol: float
        Absolute tolerance.
    rtol: float
        Relative tolerance.
    *full_output: bool
        If True, the number of iterations and time per iteration are also returned.

    Returns:
    -------
    v1: numpy.array
        Initial velocity vector.
    v2: numpy.array
        Final velocity vector.
    numiter: int
        Number of iterations.
    tpi: float
        Time per iteration in seconds.

    """

    # Verify input parameters are safe/valid
    assert_parameters_are_valid(mu, r1, r2, tof, M)

    # Retrieve the fundamental geometry of the problem
    r1_norm, r2_norm, c_norm = [np.linalg.norm(vec) for vec in [r1, r2, r2 - r1]]
    dtheta = get_transfer_angle(r1, r2, prograde)

    # Compute Vallado's transfer angle parameter; vallado(2013), p.492
    A = _get_A(r1_norm, r2_norm, dtheta)
    if A == 0.0:
        print("problem in _get_A() routine")
        raise RuntimeError("Cannot compute orbit, phase angle is 180 degrees")

    # Bisection method initial guess and limits; vallado(2013) pp.492, algo-58
    psi, psi_low, psi_up = 0.0, -4 * np.pi**2, 4 * np.pi**2

    tic = time.perf_counter()
    for numiter in range(1, maxiter + 1):
        # Evaluate the value of y at a given psi
        y = _y_at_psi(psi, r1_norm, r2_norm, A)

        if A > 0.0:
            # Readjust psi_low until y > 0.0
            while y < 0.0:
                psi_low = psi
                psi = (
                    0.8
                    * (1.0 / c3(psi))
                    * (1.0 - (r1_norm * r2_norm) * np.sqrt(c2(psi)) / A)
                )
                y = _y_at_psi(psi, r1_norm, r2_norm, A)

        X = _X_at_psi(psi, y)
        tof_new = _tof_vallado(mu, psi, X, A, y)

        # Convergence check
        if np.abs((tof_new - tof) / tof) < rtol:
            tac = time.perf_counter()
            tpi = (tac - tic) / numiter
            break

        # Bisection check
        condition = tof_new <= tof
        psi_low = psi_low + (psi - psi_low) * condition
        psi_up = psi_up + (psi - psi_up) * (not condition)

        psi = (psi_up + psi_low) / 2
    else:
        raise ValueError("Exceeded maximum iterations!")

    f = 1 - y / r1_norm
    g = A * np.sqrt(y / mu)

    gdot = 1 - y / r2_norm

    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g

    # 2024-08-10 added new_tof to output
    # return (v1, v2, numiter, tpi) if full_output is True else (v1, v2)
    return (v1, v2, tof_new, numiter, tpi)  # if full_output is True else (v1, v2)


def _tof_vallado(mu, psi, X, A, y):
    """Evaluates universal Kepler's equation.

    Parameters
    ----------
    mu: float
        The gravitational parameter.
    psi: float
        The free-parameter or independent variable.
    X: float
        Auxiliary variable.
    A: float
        The transfer angle parameter.
    y: float
        Auxiliary variable.

    Returns
    -------
    tof: float
        The computed time of flight.

    """
    tof = (X**3 * c3(psi) + A * np.sqrt(y)) / np.sqrt(mu)
    return tof


def _X_at_psi(psi, y):
    """Computes the value of X at given psi.

    Parameters
    ----------
    psi: float
        The free-parameter or independent variable.
    y: float
        Auxiliary variable.

    Returns
    -------
    X: float
        Auxiliary variable.

    """
    X = np.sqrt(y / c2(psi))
    return X


def _get_A(r1_norm, r2_norm, dtheta):
    """Computes the A constant.

    Parameters
    ----------
    r1_norm: float
        Initial position vector norm.
    r2_norm: float
        Final position vector norm.
    dtheta: float
        The transfer angle in radians.

    Returns
    -------
    A: float
        The transfer angle parameter.

    """
    t_m = 1 if dtheta < np.pi else -1
    A = t_m * (r1_norm * r2_norm * (1 + np.cos(dtheta))) ** 0.5
    return A


def _y_at_psi(psi, r1_norm, r2_norm, A):
    """Evaluates the value of y at given psi.

    Parameters
    ----------
    psi: float
        The free-parameter or independent variable.
    r1_norm: float
        Initial position vector norm.
    r2_norm: float
        Final position vector norm.
    A: float
        The transfer angle parameter.

    Returns
    -------
    y: float
        Auxiliary variable.

    Notes
    -----
    This is equation (7-59) simplified, similarly as made in Vallado(2001).
    Also see Vallado(2013) algo.58, pp.493.

    """
    y = (r1_norm + r2_norm) + A * (psi * c3(psi) - 1) / c2(psi) ** 0.5
    return y
