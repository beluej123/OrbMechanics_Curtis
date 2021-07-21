import numpy as np

def position_to_RA_dec(pos):
    i, j, k = pos
    magnitude = np.linalg.norm(pos)
    unit_vec = np.array(pos)/magnitude
    #Find declination
    dec = np.arcsin(unit_vec[2])
    
    if min(i, j) == 0:
        return ['One coordinate is 0, check again!']
    
    #Find right ascension; check for corrrect quadrant  

    if np.sign(j) == 1:
        RA = np.arccos(unit_vec[0]/np.cos(dec))
    else:
        RA = 2*np.pi - np.arccos(unit_vec[0]/np.cos(dec))
    
    return [RA*(180/np.pi), dec*(180/np.pi), magnitude, unit_vec]
    
#Returns in deg, deg; can convert to hours if necessary