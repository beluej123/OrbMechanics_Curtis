"""
Function collection related to coordinate transformations.
    Generally the functions are units-aware.
    REMEMBER, choices for solar system coordinates are often confused:
        barycentric or heliocentric; equatorial or ecliptic; but always J2000.
"""

import numpy as np
import pint  # units management

# import func_gen as fg  # method helps prevent circular imports...
from constants_1 import DEG2RAD, RAD, TAU

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~"  # pint default short unit print


def ecliptic_to_equatorial(ecl_elements, epsilon=23.439 * DEG2RAD):
    """
    Convert ecliptic orbital elements to equatorial orbital elements.
    Input Parameters:
    ----------
        ecl_elements : list or tuple, Ecliptic orbital elements
                        a, e, i_ecl, omega_ecl, argp_ecl, m_ecl
        epsilon      : [rad] Obliquity of the ecliptic
    Returns:
    ----------
        tuple        : equatorial orbital elements; angles in [rad]
                        [rad] a, e, i_eq, omega_eq, argp_eq, m_eq
    """
    a, e, i_ecl, omega_ecl, argp_ecl, m_ecl = ecl_elements
    # make sure angular inputs in radians
    if hasattr(ecl_elements[2:], "_units"):  # pint needs _units; astropy has _units too
        u_container = ecl_elements[2]._units
        if "radian" in u_container:
            out1 = True
        elif "degree" in u_container:
            out1 = True
        else:  # angle unit not found
            out1 = False
    else:  # object does not units attr
        out1 = False
        print(f"units not found in: {ecl_elements[2]}")

    # i_ecl = i_ecl.to(ureg.rad)
    # omega_ecl = omega_ecl.to(ureg.rad)
    # argp_ecl = argp_ecl.to(ureg.rad)
    # m_ecl = m_ecl.to(ureg.rad)

    # Transformation matrix from ecliptic to equatorial coordinates
    matrix_ecl_to_eq = np.array(
        [
            [1, 0, 0],
            [0, np.cos(epsilon), -np.sin(epsilon)],
            [0, np.sin(epsilon), np.cos(epsilon)],
        ]
    )

    # Compute the rotation matrices for ecliptic elements
    rx_omega_ecl = np.array(
        [
            [np.cos(omega_ecl), -np.sin(omega_ecl), 0],
            [np.sin(omega_ecl), np.cos(omega_ecl), 0],
            [0, 0, 1],
        ]
    )
    rx_i_ecl = np.array(
        [
            [1, 0, 0],
            [0, np.cos(i_ecl), -np.sin(i_ecl)],
            [0, np.sin(i_ecl), np.cos(i_ecl)],
        ]
    )
    rx_argp_ecl = np.array(
        [
            [np.cos(argp_ecl), -np.sin(argp_ecl), 0],
            [np.sin(argp_ecl), np.cos(argp_ecl), 0],
            [0, 0, 1],
        ]
    )

    # Combine the rotation matrices
    matrix_combined_ecl = np.dot(rx_omega_ecl, np.dot(rx_i_ecl, rx_argp_ecl))

    # Transform to equatorial coordinates
    matrix_combined_eq = np.dot(matrix_ecl_to_eq, matrix_combined_ecl)

    # Extract the equatorial elements
    i_eq = np.arccos(matrix_combined_eq[2, 2])
    omega_eq = np.arctan2(matrix_combined_eq[0, 2], -matrix_combined_eq[1, 2]) % TAU
    argp_eq = np.arctan2(matrix_combined_eq[2, 0], matrix_combined_eq[2, 1]) % TAU

    i_eq *= 1 * RAD  # assign radian unit
    omega_eq *= 1 * RAD
    argp_eq *= 1 * RAD

    return a, e, i_eq, omega_eq, argp_eq, m_ecl


def test_ecliptic_to_equatorial():
    """
    Coordinate transformation.
    """
    planet_id = 3  # earth
    # JPL data, ecliptic referenced orbital elements & angles in degrees
    e_j_coe_ecl, e_j_rates_ecl = fg.planetary_elements(planet_id, d_set=0)

    # convert JPL coe and rates from ecliptic to equatorial
    e_j_coe_equ = ecliptic_to_equatorial(ecl_elements=e_j_coe_ecl)
    print("\nEcliptic Elements: sma, ecc, i, Ω, ω, M")
    for coe_val in e_j_coe_ecl:
        print(f"{coe_val:0.6g~}, ", end="")
    print("\n\nEquatorial Elements: sma, ecc, i, Ω, ω, M")
    for coe_val in e_j_coe_equ:
        print(f"{coe_val:0.6g~}, ", end="")


def main():
    """just a placeholder to help with editor navigation:--)"""
    return


# use the following to test/examine functions
if __name__ == "__main__":
    test_ecliptic_to_equatorial()  # compare Curtis [3] tbl 8.1 & JPL Horizons
