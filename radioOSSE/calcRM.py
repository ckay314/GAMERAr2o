import numpy as np
import matplotlib.pyplot as plt
import geom 
from extractGAMERA import GAMERAres
from itertools import combinations
from matplotlib import cm
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
from sunpy.coordinates import HeliographicStonyhurst
import pandas as pd
import datetime

# Gamera coords in (lon, colat, r) = (phi, theta, r) = (deg, deg, rs)
# Geom funcs want in same order


Re2Rs = 0.009164023 # 1 rE to rSun
dtor  = np.pi / 180.

# |-----------------------|
# |--- Satellite setup ---|
# |-----------------------|
if False:
    L1pt = [0, 90, 215.03*0.99] 
    L1xyz = geom.SPH2CART(L1pt)

    # make a fake orbit for now
    sep = 50 * Re2Rs 
    orbTheta = 10 * dtor

    fakets1 = np.linspace(0,np.pi,10)
    fakets2 = np.linspace(0,np.pi,10) + np.pi /2 
    fakets3 = np.linspace(0,np.pi,10) + np.pi 

    # Make fake orbits
    dx1 = sep * np.cos(fakets1) 
    dy1 = np.cos(orbTheta) * sep * np.sin(fakets1) 
    dz1 = np.sin(orbTheta) * sep * np.sin(fakets1) 
    sat1 = np.transpose(np.array([dx1, dy1, dz1])) + L1xyz

    dx2 = sep * np.cos(fakets2) 
    dy2 = np.cos(orbTheta) * sep * np.sin(fakets2) 
    dz2 = np.sin(orbTheta) * sep * np.sin(fakets2) 
    sat2 = np.transpose(np.array([dx2, dy2, dz2]))+ L1xyz

    dx3 = sep * np.cos(fakets3) 
    dy3 = np.cos(orbTheta) * sep * np.sin(fakets3) 
    dz3 = np.sin(orbTheta) * sep * np.sin(fakets3) 
    sat3 = np.transpose(np.array([dx3, dy3, dz3]))+ L1xyz

else:
    allSats = pd.read_excel('largeliss_demo.xlsx', sheet_name=None)
    satnames = list(allSats.keys())
    
    # sat Tags ['Sat 1', 'Sat 2', 'Sat 3', 'Sat 4', 'Sat 5']
    # value Tags ['Date (UTC)', 'X (ECLIPJ2000, km)', 'Y (ECLIPJ2000, km)', 'Z (ECLIPJ2000, km)', 'Xd (ECLIPJ2000, km/s)', 'Yd (ECLIPJ2000, km/s)', 'Zd (ECLIPJ2000, km/s)']
    
    
    
    # Proper calculation of coordinates accounting for Earth motion in lat
    sats = []
    for sat in ['Sat 1', 'Sat 2', 'Sat 3']:
        sats.append([])
        mySat = allSats[sat]
        myX, myY, myZ = mySat['X (ECLIPJ2000, km)'], mySat['Y (ECLIPJ2000, km)'], mySat['Z (ECLIPJ2000, km)']
        myTime = mySat['Date (UTC)']
    
        for j in range(240):
            idx = 0 + j*10
            x, y, z = myX[idx], myY[idx], myZ[idx]
            #print(sat, j, x,y,z)
            lon, lat, r = geom.CART2SPH([x,y,z]) # deg, deg, km
            #print (lon, lat, r)
    
            # format Apr 10 2030, 23:59:59
            t = datetime.datetime.strptime(myTime[idx], "%b %d %Y, %H:%M:%S" )
    
            eclip_frame = GeocentricTrueEcliptic(equinox="J2000", obstime=t)   
            geo_eclip_coord = SkyCoord(lon=lon*u.deg, lat=(90-lat)*u.deg, distance=r*u.km, frame=eclip_frame)
        
            # Assuming that GAMERA results are in Stonyhurst
            stonyhurst_coord = geo_eclip_coord.transform_to(HeliographicStonyhurst)
            stonyXYZ = stonyhurst_coord.transform_to(HeliographicStonyhurst(representation_type='cartesian')).cartesian.xyz.to(u.au).to_value()
            
            sats[-1].append(stonyXYZ)
            #print (sat, j, stonyXYZ)


# |----------------------------|
# |--- Get baselines in SPH ---|
# |----------------------------|

sat1 = np.array(sats[0])
sat2 = np.array(sats[1])
sat3 = np.array(sats[2])


if True:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sat1[:,0], sat1[:,1], sat1[:,2], 'r--')
    '''for i in range(len(sat1[:,0])):
        ax.plot(sat1[i,0],sat1[i,1],sat1[i,2], 'ro')
        ax.plot(sat2[i,0],sat2[i,1],sat2[i,2], 'bo')
        ax.plot(sat3[i,0],sat3[i,1],sat3[i,2], 'yo')
        ax.plot([sat1[i,0], sat2[i,0]], [sat1[i,1], sat2[i,1]], [sat1[i,2], sat2[i,2]], '--', c='purple')
        ax.plot([sat1[i,0], sat3[i,0]], [sat1[i,1], sat3[i,1]], [sat1[i,2], sat3[i,2]], '--', c='orange')
        ax.plot([sat3[i,0], sat2[i,0]], [sat3[i,1], sat2[i,1]], [sat3[i,2], sat2[i,2]], '--', c='green')'''
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# Get baselines

# Get ne, Bx, By, Bz
#filePath = '/Users/kaycd1/GAMERA/fromDevoj/256serial/wsaCR2261cme05092022.gam.h5'
#res = GAMERAres(filePath)

# Calc FR