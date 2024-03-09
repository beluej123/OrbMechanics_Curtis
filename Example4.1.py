#  based on: Orbital Mechanics for Engineering Students, 2nd ed., 2009
#  by Howard D. Curtis
import numpy as np


def position_to_RA_dec(pos):
    i, j, k = pos
    magnitude = np.linalg.norm(pos)
    unit_vec = np.array(pos) / magnitude
    # Find declination
    dec = np.arcsin(unit_vec[2])

    if min(i, j) == 0:
        return ["One coordinate is 0, check again!"]

    # Find right ascension; check for corrrect quadrant

    if np.sign(j) == 1:
        RA = np.arccos(unit_vec[0] / np.cos(dec))
    else:
        RA = 2 * np.pi - np.arccos(unit_vec[0] / np.cos(dec))

    return [RA * (180 / np.pi), dec * (180 / np.pi), magnitude, unit_vec]
    # Return degrees; can convert to hours if necessary


# test example 4.1, from Curtis, p.205 in my book
ra, dec, magP, uVec = position_to_RA_dec([-5368, -1784, 3691])
print(
    "right ascension =",
    ra,
    "\ndeclination=",
    dec,
    "\ndistance=",
    magP,
    "\nunit vector=",
    uVec,
)
