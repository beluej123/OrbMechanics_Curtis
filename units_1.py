"""
Explore units management; basically a test file.
"""
import astropy.units as u
import numpy as np
import pint
from pint import Quantity, UnitRegistry


def test_units_astropy_1():
    """
    Explore units conversions with astropy.
    """
    print(f"Explore unit conversions with astropy.")
    # Define variables with units
    r0_mag = 10 * u.km
    time = 2 * u.s
    speed = r0_mag / time
    print(f"Speed: {speed}")

    # NumPy arrays with units
    r0_vec = np.array([1, 5, 10]) * u.km
    times = np.array([10, 20, 30]) * u.s
    speeds = r0_vec / times # [km/s]
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
    if hasattr(speeds,'unit'):
        print(f"speeds: {speeds}")
        print(f"speeds units assigned: {getattr(speeds,'unit')}")
    else:
        print(f"speeds needs unit assigned.")
    
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
        
    print(f"Explore units with pint.")
    ureg = UnitRegistry()
    # Define variables with units
    r0_mag = 10 * ureg.km
    time = 2 * ureg.second
    speed = r0_mag / time

    print(f"r0_mag: {r0_mag:.4f}")
    print(f"r0_mag: {r0_mag:.4f~}") # ~ short units notation (for pint)

    # NumPy arrays with units
    r0_vec = np.array([1, 5, 10]) * ureg.km
    times = np.array([10, 20, 30]) * ureg.sec
    speeds = r0_vec / times # [km/s]
    print(f"Speeds: {speeds:.4f~}") # verified [m/s]
    # verify unit conversion
    speeds = speeds.to(ureg.m / ureg.s)
    print(f"Speeds: {speeds:.4g~}") # ~ short units notation
    # verify not double unit conversion
    speeds = speeds.to(ureg.km / ureg.s)
    print(f"Speeds: {speeds:.4g~}") # ~ short units notation
    # verify au and time conversions
    speeds = speeds.to(ureg.au / ureg.day)
    print(f"Speeds: {speeds:.4g~}") # ~ short units notation
    
    # examine unit attribute
    if hasattr(r0_vec,'units'):
        print(f"r0_vec: {r0_vec:~}")
        print(f"r0_units assigned: {getattr(r0_vec,'units'):~}")
        print(f"r0_units assigned: {r0_vec.units}")
    else:
        print(f"r0_vec needs units assigned.")
    
    # review variable dictionary
    # print(f"dir(speeds):\n{dir(speeds)}")
    return


def main():
    # just a placeholder to help with editor navigation:--)
    return


# use the following to test/examine functions
if __name__ == "__main__":
    test_units_pint_1()
    # test_units_astropy_1()
    main()  # do nothing placeholder :--)