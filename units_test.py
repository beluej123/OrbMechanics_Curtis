"""
Explore units management; basically a test file.
"""

import astropy.units as u
import numpy as np
import pint
from pint import Quantity, UnitRegistry

import constants
from constants import AU_, TAU

ureg = UnitRegistry()  # pint units management
Q_ = ureg.Quantity  # Q_ is an alias not an object


def angle_norm_deg(angle):
    """
    Checks for pint angle (pint.Quantity), 0->360.  If so, check if it's
        dimensionless, assumed to be in degrees or has explicit angle units.
        Then calculate the modulo 360, accommodating both dimensionless and
        angular quantities.
    """
    if isinstance(angle, ureg.Quantity):  # check for pint quanity
        if angle.dimensionality == ureg.dimensionless:
            return (angle.to(ureg.deg).magnitude % 360) * ureg.deg
            # return (angle % (360 * ureg.degree)).to('degree')
        elif angle.check("[angle]"):
            print("check2")
            return angle.magnitude % 360 * angle.units
        else:
            # I know there's a way to automatically get the module name...
            print("Error in angle_norm_deg()")
            raise ValueError("Input angle must be dimensionless or have angle units")
    else:  # not a pint quanity, but normalize value
        return angle % 360


def angle_norm_rad(angle):
    """
    Checks for pint angle (pint.Quantity), 0->2π.  If so, check if it's
        dimensionless, assumed to be in radians or has explicit angle units.
        Then calculate the modulo 2π, accommodating both dimensionless and
        angular quantities.

    Input Args:
    ----------
        angle: pint quantity OR a number, assumes radian value.

    Returns:
    ----------
        Angle modulo 2π, with units same the input angle;
            radians if the input was dimensionless.
    """
    if isinstance(angle, ureg.Quantity):
        if angle.dimensionality == ureg.dimensionless:
            return (angle.to(ureg.rad).magnitude % (TAU)) * ureg.rad
        elif angle.check("[angle]"):
            return (angle.magnitude % (TAU)) * angle.units
        else:
            # I know there a way to automatically get the module name...
            print("Error in angle_norm_rad()")
            raise ValueError("Input angle must be dimensionless or have angle units")
    else:
        return angle % (TAU)


def test_units_astropy_1():
    """
    Explore units conversions with astropy.
    """
    print("Explore unit conversions with astropy.")
    # Define variables with units
    r0_mag = 10 * u.km
    time = 2 * u.s
    speed = r0_mag / time
    print(f"Speed: {speed}")

    # NumPy arrays with units
    r0_vec = np.array([1, 5, 10]) * u.km
    times = np.array([10, 20, 30]) * u.s
    speeds = r0_vec / times  # [km/s]
    print(f"Speeds: {speeds}")

    # verify unit conversion
    speeds = speeds.to(u.m / u.s)
    print(f"Speeds: {speeds}")
    # verify no double unit conversion
    speeds = speeds.to(u.km / u.s)
    print(f"Speeds: {speeds}")

    # verify au and time conversions
    speeds = speeds.to(u.au / u.day)
    print(f"Speeds: {speeds}")
    speeds = speeds.to(u.km / u.day)
    print(f"Speeds in m/s: {speeds}")
    # examine unit attribute
    if hasattr(speeds, "unit"):
        print(f"speeds: {speeds}")
        print(f"speeds units assigned: {getattr(speeds,'unit')}")
    else:
        print("speeds needs unit assigned.")

    # review variable dictionary
    print(f"dir(speeds):\n{dir(speeds)}")

    return


def test_units_pint_1():
    """
    Explore unit conversions with pint.
    Note:
    ----------
        ~ prints in short units notation; i.e. km instead of kilometer
    """

    def is_unit_aware(variable):
        return isinstance(variable, Quantity)

    print("Explore units with pint.")
    # Define variables with units
    r0_mag = 10 * ureg.km
    time = 2 * ureg.second
    speed = r0_mag / time

    print(f"r0_mag: {r0_mag:.4f}")
    print(f"r0_mag: {r0_mag:.4f~}")  # ~ short units notation (for pint)

    # NumPy arrays with units
    r0_vec = np.array([1, 5, 10]) * ureg.km
    times = np.array([10, 20, 30]) * ureg.sec
    speeds = r0_vec / times  # [km/s]
    print(f"Speeds: {speeds:.4f~}")  # verified [m/s]
    # verify unit conversion
    speeds = speeds.to(ureg.m / ureg.s)
    print(f"Speeds: {speeds:.4g~}")  # ~ short units notation
    # verify not double unit conversion
    speeds = speeds.to(ureg.km / ureg.s)
    print(f"Speeds: {speeds:.4g~}")  # ~ short units notation
    # verify au and time conversions
    speeds = speeds.to(ureg.au / ureg.day)
    print(f"Speeds: {speeds:.4g~}")  # ~ short units notation

    # examine unit attribute
    if hasattr(r0_vec, "units"):
        print(f"r0_vec: {r0_vec:~}")
        print(f"r0_units assigned: {getattr(r0_vec,'units'):~}")
        print(f"r0_units assigned: {r0_vec.units}")
    else:
        print("r0_vec needs units assigned.")

    # review variable dictionary
    # print(f"dir(speeds):\n{dir(speeds)}")
    return


def test_pint_constants():
    """Reset the pint au conversion constant."""
    print("\nExplore pint units, conversions, reassignments:")
    print("   Compare pint value for au with Vallado au value: ")
    au_c = Q_(AU_, "km")  # au in km from my constants library
    au_p = Q_(1, "au").to("km")  # convert pint au to km
    print(f"   au pint: {au_p:~}")  # ~ = short unit form (pint)
    print(f"   au cons: {au_c:~}, direct from Vallado")

    # next explore override pint au to km conversion to use Vallado
    ctx = pint.Context()  # context for new conversion constant
    ctx.redefine(f"au = {AU_} km")  # see constants.py for reference
    print(f"   au Vallado: {au_c.to("km", ctx):~}, pint conversion")


def test_pint_angles():
    """angle's, but note they are dimensionless :--)"""
    print("\nPint radian and degree angles:")
    t_rad = 1 * ureg.rad
    t_deg = 1 * ureg.deg
    print(f"radian: {t_rad}")
    print(f"degree: {t_deg}")
    print(f"convert 1 rad to deg: {t_rad.to(ureg.deg)}")

    print("\nExplore pint angle normalization:")
    incl_rad = Q_(60, "rad")
    incl_rad_norm = angle_norm_rad(incl_rad)
    incl_deg_norm = angle_norm_deg(incl_rad.to("deg"))
    # check radians conversions
    print(f"   incl: {incl_rad:~}, angle not normalized")  # ~ = short unit form (pint)
    print(f"   incl: {angle_norm_rad(60)}, non-pint angle, normalized")
    print(f"   incl: {incl_rad_norm:~}, normalized pint angle")
    print(f"   incl: {incl_deg_norm:~}, normalized pint angle")
    print(f"   incl: {angle_norm_rad(60*ureg.rad):~}, normalized pint angle")

    # next check degrees conversions
    print("\nExplore pint degrees angles:")
    incl_deg = Q_(7283, "deg")
    incl_rad_norm = angle_norm_rad(incl_deg)
    incl_deg_norm = angle_norm_deg(incl_deg.to("rad"))
    # below, ~ = short unit form (pint)
    print(f"   incl: {incl_deg:~}, angle not normalized")
    print(f"   incl: {angle_norm_deg(7283)}, non-pint angle, normalized")
    print(f"   incl: {incl_rad_norm:~}, normalized pint angle")
    print(f"   incl: {incl_deg_norm:~}, normalized pint angle")
    print(f"   incl: {angle_norm_deg(365*ureg.deg):~}, normalized pint angle")
    
    incl = 1*ureg.deg
    print(f"incl, deg: {incl}")


def main():
    """just a placeholder to help with editor navigation:--)"""
    return


# use the following to test/examine functions
if __name__ == "__main__":
    # test_units_pint_1()
    # test_pint_constants()  # explore use Vallado au constant, etc.
    test_pint_angles()  # explore angle deg/rad conversions & normalization
    # test_units_astropy_1()
    main()  # do nothing placeholder :--)
