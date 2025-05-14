"""
Explore units management and tests for func_units.py.
"""

import astropy.units as ap_u
import numpy as np
import pint

from constants_1 import AU_, AU_KM, CENT, DEG, GM_SUN, RAD, TAU
from func_units import contains_angle, units_aware

ureg = pint.UnitRegistry()  # pint units management


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


def test_units_aware():
    """Units-aware may be either pint or astropy"""
    print("Explore units with pint and astropy.")
    print("DO NOT MIX units BETWEEN pint and astropy.")
    # define variables with pint units
    r0_mag = 10 * ureg.km  # pint units
    t0 = 2 * ureg.second
    speed0 = r0_mag / t0  # pint units-aware
    print("Pint units:")
    print(f"r0_mag: {r0_mag:~.4f}, speed: {speed0:~}")  # ~ = pint print short units
    # define variables with astro units
    print("Astropy units:")
    r1_mag = 53.1 * ap_u.km  # astropy units; known linter problem - so sad
    t1 = 21.3 * ap_u.second
    speed1 = r1_mag / t1
    print(f"r1_mag: {r1_mag:.4f}, speed: {speed1:.4f}")

    ck_units = units_aware(value=r1_mag)
    print("NOW, pint or astropy?`")
    if ck_units == "pint":
        # ~ = pint print short units
        print(f"pint units, r0_mag = {getattr(r0_mag,'units'):~}")
    elif ck_units == "astropy":
        # unit for astropy, units for pint
        print(f"astropy units, r1_mag = {getattr(r1_mag,'unit')}")


def test_units_pint1():
    """
    Explore unit conversions with pint.
    Note:
    ----------
        ~ prints in short units notation; i.e. km instead of kilometer
    """
    print("Explore units with pint; test_units_pint1.")
    # Define variables with units
    r0_mag = 10 * ureg.km
    time = 2 * ureg.second
    speed = r0_mag / time

    print(f"r0_mag: {r0_mag:.4f}")
    print(
        f"r0_mag: {r0_mag:.4f~} print in short form."
    )  # ~ short units notation (for pint)
    print(f"speed: {speed:.4f~}")  # ~ short units notation (for pint)

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

    # look for unit attribute
    if hasattr(r0_vec, "units"):  # units for pint; unit for astropy
        print(f"r0_vec: {r0_vec:~}")
        print(f"r0_units assigned: {getattr(r0_vec,'units'):~}")
        print(f"r0_units assigned: {r0_vec.units}")
    else:
        print("r0_vec needs units assigned.")


def test_pint_constants1():
    """Reset the pint au conversion constant."""
    print("\nExplore pint units, conversions, reassignments:")
    print("   Compare AU value from Pint vs. Vallado: ")
    au_c = AU_  # au in km from constants_1.py; units-aware
    au_p = 1 * ureg.au  # pint au unit
    au_p = au_p.to("km")  # convert pint au to km
    print(f"   au Pint: {au_p:~}")  # ~ = short unit form (pint)
    print(f"   au Constant: {au_c}, direct from Vallado")
    print(f"   AU_: {AU_.m}, direct from Vallado")  # .m = magnitude

    # explore override pint au to km conversion to use Vallado
    au1 = ureg.Quantity("1 au")
    au1.to("km")
    print(f"   au1 = {au1.to('km')}, pint unit")
    ctx = pint.Context()  # context for new conversion constant
    ctx.redefine(f"au = {AU_.magnitude} km")  # see constants.py for reference
    print(f"   au Vallado: {au1.to("km", ctx)}")
    print(f"   au Pint: {au_p:~}")


def test_pint_constants2():
    """pint units constants/objects to manage constants."""
    print("\nTest_pint_constants2():")
    print(f"AU_KM: {AU_KM:~}")
    print(f"GM_SUN: {GM_SUN:~}")

    rates_1 = 57.4 * DEG / CENT
    print(f"rates_1: {rates_1}")
    # print(f"deg/cy to rad/cy: {rates_1.to(ureg.rad/ureg.cy)}")
    print(f"deg/cy to rad/cy: {rates_1.to(RAD/CENT)}")
    print(f"rates_1.units: {rates_1.units}")
    print(f"rates_1 contains angle: {contains_angle(rates_1)}")

    unit1 = ureg.meter / ureg.second
    unit2 = ureg.rad / ureg.second
    unit3 = ureg.deg * ureg.sec * ureg.meter
    unit4 = ureg.cycle / ureg.second
    unit5 = ureg.meter

    print(f"\nunit1, {unit1} contains angle: {contains_angle(unit1)}")
    print(f"unit2, {unit2} contains angle: {contains_angle(unit2)}")
    print(f"unit3, {unit3} contains angle: {contains_angle(unit3)}")
    print(f"unit4, {unit4} contains angle: {contains_angle(unit4)}")
    print(f"unit5, {unit5} contains angle: {contains_angle(unit5)}")


def test_pint_angles():
    """Angle's, but note they are dimensionless :--)"""
    print("\nPint radian and degree angles:")
    t_rad = 1 * ureg.rad
    t_deg = 1 * ureg.deg
    print(f"radian: {t_rad}")
    print(f"degree: {t_deg}")
    print(f"convert 1 rad to deg: {t_rad.to('deg')}")

    print("\nExplore pint angle normalization (range 0->2pi):")
    incl_rad = 60 * ureg.rad
    incl_rad_norm = angle_norm_rad(incl_rad)
    incl_deg_norm = angle_norm_deg(incl_rad.to("deg"))
    # check radians conversions
    print(f"   incl: {incl_rad:~}, angle not normalized")  # ~ = short unit form (pint)
    print(
        f"   incl: {incl_rad.to('deg'):~}, auto-normalized"
    )  # ~ = short unit form (pint)
    print(f"   incl: {angle_norm_rad(60)}, non-pint angle, normalized")
    print(f"   incl: {incl_rad_norm:~}, normalized pint angle")
    print(f"   incl: {incl_deg_norm:~}, normalized pint angle")
    print(f"   incl: {angle_norm_rad(60*ureg.rad):~}, normalized pint angle")

    # next check degrees conversions
    print("\nExplore pint degrees angles:")
    incl_deg = 7283 * ureg.deg
    incl_rad_norm = angle_norm_rad(incl_deg)
    incl_deg_norm = angle_norm_deg(incl_deg.to("rad"))
    # below, ~ = short unit form (pint)
    print(f"   incl: {incl_deg:~}, angle not normalized")
    print(f"   norm 7283 deg = {angle_norm_deg(7283)}, non-pint angle, normalized")
    print(f"   incl: {incl_rad_norm:~}, normalized pint angle")
    print(f"   incl: {incl_deg_norm:~}, normalized pint angle")
    print(f"   norm 365 deg = {angle_norm_deg(365*ureg.deg):~}, normalized pint angle")

    incl_1 = DEG
    # incl_1 = 55.5 # no angle units assigned
    incl_2 = RAD
    if contains_angle(incl_2):  # if true
        print(f"incl_1 look for angle: {incl_1}")
        print(f"incl_2 look for angle: {incl_2}")
    else:
        print(f"no angular units found: {incl_2}")


def test_units_astropy1():
    """
    Explore units conversions with astropy.
    """
    print("Astropy unit conversions, test_units_astropy1:")
    # Define variables with units
    r0_mag = 10 * ap_u.km
    time = 2 * ap_u.s
    speed = r0_mag / time
    print(f"Speed: {speed}")

    # NumPy arrays with units
    r0_vec = np.array([1, 5, 10]) * ap_u.km
    times = np.array([10, 20, 30]) * ap_u.s
    speeds = r0_vec / times  # [km/s]
    print(f"Speeds: {speeds}")

    # verify unit conversion
    speeds = speeds.to(ap_u.m / ap_u.s)
    print(f"Speeds: {speeds}")
    # verify no double unit conversion
    speeds = speeds.to(ap_u.km / ap_u.s)
    print(f"Speeds: {speeds}")

    # verify au and time conversions
    speeds = speeds.to(ap_u.au / ap_u.day)
    print(f"Speeds: {speeds}")
    speeds = speeds.to(ap_u.km / ap_u.day)
    print(f"Speeds in m/s: {speeds}")
    # examine unit attribute
    if hasattr(speeds, "unit"):  # astropy "unit"; pint "units" :--)
        print(f"speeds: {speeds}")
        print(f"speeds units assigned: {getattr(speeds,'unit')}")
    else:
        print("speeds needs unit assigned.")


def main():
    """just a placeholder to help with editor navigation:--)"""
    return


# use the following to test/examine functions
if __name__ == "__main__":
    test_units_aware() # check for pint and astropy units
    # test_units_pint1()
    # test_pint_constants1()  # explore use Vallado au constant
    # test_pint_constants2()  # explore use Vallado au constant, etc.
    # test_pint_angles()  # explore angle deg/rad conversions & normalization
    # test_units_astropy1()
    main()  # do nothing placeholder :--)
