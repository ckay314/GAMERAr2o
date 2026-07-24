import numpy as np

dtor  = np.pi / 180.
radeg = 180. / np.pi
pi    = np.pi 

#|-------------------------------|
#|---- Geometry helper funcs ----|
#|-------------------------------|
def rotx(vec, ang):
    """
    
    Rotate a 3D vector by ang (input in degrees) about the x-axis 
    
    """
    ang *= dtor
    yout = np.cos(ang) * vec[1] - np.sin(ang) * vec[2]
    zout = np.sin(ang) * vec[1] + np.cos(ang) * vec[2]
    return [vec[0], yout, zout]

def roty(vec, ang):
    """
    
    Rotate a 3D vector by ang (input in degrees) about the y-axis 
    
    """
    ang *= dtor
    xout = np.cos(ang) * vec[0] + np.sin(ang) * vec[2]
    zout =-np.sin(ang) * vec[0] + np.cos(ang) * vec[2]
    return [xout, vec[1], zout]

def rotz(vec, ang):
    """
    
    Rotate a 3D vector by ang (input in degrees) about the z-axis 
    
    """
    ang *= dtor
    xout = np.cos(ang) * vec[0] - np.sin(ang) * vec[1]
    yout = np.sin(ang) * vec[0] + np.cos(ang) * vec[1]
    return [xout, yout, vec[2]]

def SPH2CART(sph_in):
    """
    
    Conversion between spherical and cartesian coordinates
    
    This assumes input [lon, colat, R] with angs in degrees
    """
    r = sph_in[2]
    colat =  sph_in[1] * dtor
    lon = sph_in[0] * dtor
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)
    return [x, y, z]
    
def CART2SPH(x_in):
    """
    Conversion between cartesian and spherical coordinates
    
    This returns output [lon, colat, R] with angs in degrees
    
    """
    r_out = np.sqrt(x_in[0]**2 + x_in[1]**2 + x_in[2]**2)
    colat = np.arccos(x_in[2] / r_out)  * radeg
    lon_out = (np.arctan2(x_in[1] , x_in[0]) * radeg) % 360

    return [lon_out, colat, r_out]

