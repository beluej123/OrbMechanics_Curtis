import numpy as np

def deltaV_Hohmann_circular(r_a, r_b, mu):
    #rb greater than ra
    a = r_b/r_a
    A = 1/np.sqrt(a)
    B = -1*(np.sqrt(2)*(1-a))/np.sqrt(a*(1+a))
    C = np.sqrt(mu/r_a)
    return (A + B - 1)*C

def deltaV_Bielliptic_circular(r_a, r_b, r_c, mu):
    #rb is transfer ellipse
    a = r_c/r_a
    b = r_b/r_a
    A = np.sqrt((2*(a + b))/(a*b))
    B = -1*((1 + np.sqrt(a))/np.sqrt(a))
    C = -1*np.sqrt(2/(b*(1+b)))*(1-b)
    D = np.sqrt(mu/r_a)
    return (A + B + C)*D

def t_circular(r, mu):
    return ((2*np.pi)/np.sqrt(mu))*r**1.5

def t_ellipse(r_p, r_a, mu):
    a = (r_a + r_p)/2
    return ((2*np.pi)/np.sqrt(mu))*a**1.5

#Find the total delta-v requirement for a bi-elliptical Hohmann
#transfer from a geocentric circular orbit of 7000 km radius to
#one of 105 000 km radius. Let the apogee of the first ellipse
#be 210 000 km. Compare the delta-v schedule and total flight time
#with that for an ordinary single Hohmann transfer ellipse.

r_o1 = 7000
r_o2 = 210000
r_o3 = 105000
mu = 398600

#Compare delta v

dv_hohmann = deltaV_Hohmann_circular(r_o1, r_o3, mu)
dv_biell = deltaV_Bielliptic_circular(r_o1, r_o2, r_o3, mu)

if dv_biell < dv_hohmann:
    print('Bi-elliptic transfer more efficient by ' + 
          str(round(dv_hohmann - dv_biell, 4)) + ' km/s')


#Compare flight times
#Hohmann:
dt_hohmann = t_ellipse(r_o1, r_o3, mu)/2

#Bi-elliptic:
dt_biell_1 = t_ellipse(r_o1, r_o2, mu)/2
dt_biell_2 = t_ellipse(r_o2, r_o3, mu)/2
dt_biell = dt_biell_1 + dt_biell_2

print('Bi-elliptic transfer takes ' +
      str(round((dt_biell - dt_hohmann)/3600, 4)) + ' hours longer')


