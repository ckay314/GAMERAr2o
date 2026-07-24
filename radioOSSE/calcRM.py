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
import pickle


# Gamera coords in (lon, colat, r) = (phi, theta, r) = (deg, deg, rs)
# Geom funcs want in same order


Re2Rs = 0.009164023 # 1 rE to rSun
dtor  = np.pi / 180.


# |-----------------|
# |--- Sim Setup ---|
# |-----------------|
# Number of sats to include
nSats = 3

# Starting time within sat trajectories
tIdx = 400

# Option to fake/loadcalc/load
mode = 'load'

# Number of points in the baselines
nBase = 30

# |----------------------|
# |--- Fake Satellite ---|
# |----------------------|
if mode == 'fake':
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

# |--------------------|
# |--- Excel2Pickle ---|
# |--------------------|
if mode == 'loadcalc':
    allSats = pd.read_excel('largeliss_demo.xlsx', sheet_name=None)
    satnames = list(allSats.keys())
    
    # sat Tags ['Sat 1', 'Sat 2', 'Sat 3', 'Sat 4', 'Sat 5']
    # value Tags ['Date (UTC)', 'X (ECLIPJ2000, km)', 'Y (ECLIPJ2000, km)', 'Z (ECLIPJ2000, km)', 'Xd (ECLIPJ2000, km/s)', 'Yd (ECLIPJ2000, km/s)', 'Zd (ECLIPJ2000, km/s)']
    
    # Proper calculation of coordinates accounting for Earth motion in lat
    sats = []
    ts = []
    for sat in ['Sat 1', 'Sat 2', 'Sat 3', 'Sat 4', 'Sat 5']:
        sats.append([])
        ts.append([])
        mySat = allSats[sat]
        myX, myY, myZ = mySat['X (ECLIPJ2000, km)'], mySat['Y (ECLIPJ2000, km)'], mySat['Z (ECLIPJ2000, km)']
        myTime = mySat['Date (UTC)']
        
        for idx in range(len(myX)):
        #for j in range(240):
            #idx = 0 + j*10
            x, y, z = myX[idx], myY[idx], myZ[idx]
            #print(sat, j, x,y,z)
            lon, lat, r = geom.CART2SPH([x,y,z]) # deg, deg, km
            #print (lon, lat, r)
    
            # format Apr 10 2030, 23:59:59
            t = datetime.datetime.strptime(myTime[idx], "%b %d %Y, %H:%M:%S" )
            ts[-1].append(t)
            
            eclip_frame = GeocentricTrueEcliptic(equinox="J2000", obstime=t)   
            geo_eclip_coord = SkyCoord(lon=lon*u.deg, lat=(90-lat)*u.deg, distance=r*u.km, frame=eclip_frame)
        
            # Assuming that GAMERA results are in Stonyhurst
            stonyhurst_coord = geo_eclip_coord.transform_to(HeliographicStonyhurst)
            stonyXYZ = stonyhurst_coord.transform_to(HeliographicStonyhurst(representation_type='cartesian')).cartesian.xyz.to(u.au).to_value()
            stonyXYZ = stonyXYZ * 215.03
            
            sats[-1].append(stonyXYZ)
        sats[-1] = np.array(sats[-1])
        ts[-1] = np.array(ts[-1])
        
    # |--- Save the pickle ---|
    res = {}
    res['sats'] = sats
    # ts are all the same, just save one
    res['ts'] = ts[0]
    # Get time difference in hr
    dts = []
    for i in range(len(ts[0])):
        dts.append((ts[0][i] - ts[0][0]).total_seconds()/3600.)
    res['dts'] = np.array(dts)
    
    with open('largeliss_demo.pkl', 'wb') as file:
        pickle.dump(res, file)



# |-------------------|
# |--- Load Pickle ---|
# |-------------------|
if mode == 'load':
    with open('largeliss_demo.pkl', 'rb') as file:
        res = pickle.load(file)
        
    sats = res['sats'] # [sat1, sat2, ...] where sat is [nTimes, 3] with xyz in stony AU
    ts = res['ts']
    dts = res['dts']



# |-------------------|
# |--- Load Gamera ---|
# |-------------------|
filePath = '/Users/kaycd1/GAMERA/fromDevoj/256serial/wsaCR2261cme05092022.gam.h5'
gamRes = GAMERAres(filePath)




# |--------------------------------|
# |--- Get sat locs on Gam time ---|
# |--------------------------------|
# Gamera has 60 time steps (Step#0 -Step#59)
# One at very beginning then rest later separated
# by about one hour
gamDTs = gamRes.timeDeltas[1:] - gamRes.timeDeltas[1]
nTimes = len(gamDTs)
rwT0 = ts[tIdx]
shiftSatDTs = dts - dts[tIdx]
idxs = []
fracs   = []
for i in range(nTimes):
    b4idx = np.max(np.where(shiftSatDTs < gamDTs[i]))
    f1 = (gamDTs[i] - shiftSatDTs[b4idx]) / (shiftSatDTs[b4idx+1] - shiftSatDTs[b4idx]) 
    idxs.append(b4idx)
    fracs.append(f1)
idxs  = np.array(idxs)
fracs = np.array(fracs)

miniSats = []
for i in range(nSats):
    mySat = sats[i]
    myX = (1 - fracs) * mySat[idxs,0] + fracs * mySat[idxs+1, 0]
    myY = (1 - fracs) * mySat[idxs,1] + fracs * mySat[idxs+1, 1]
    myZ = (1 - fracs) * mySat[idxs,2] + fracs * mySat[idxs+1, 2]
    miniSat = np.transpose(np.array([myX, myY, myZ]))
    miniSats.append(miniSat)
    

# |----------------------|
# |--- Get baselines  ---|
# |----------------------|
pairs = list(combinations(range(len(miniSats)), 2))
baselines = [] # form [nPair, nTime, nBase, 3]
blXYZs  = [] # form [nPair, nTime, nBase, 3]
for aPair in pairs:
    satA = miniSats[aPair[0]]
    satB = miniSats[aPair[1]] # shape [nt, 3]
        
    blX = np.linspace(satA[:,0], satB[:,0], nBase) # shape [nBase, nT]
    blY = np.linspace(satA[:,1], satB[:,1], nBase)
    blZ = np.linspace(satA[:,2], satB[:,2], nBase)
    blXYZ = np.array([blX, blY, blZ]) # [3, nBase, nT]
    
    ogShape = blX.shape
    blX1d = blX.reshape([-1])
    blY1d = blY.reshape([-1])
    blZ1d = blZ.reshape([-1])

    bl = np.array(geom.CART2SPH([blX1d, blZ1d, blZ1d])) # [[pt1], [pt2], ...]
    bl1 = bl[0].reshape(ogShape)
    bl2 = bl[1].reshape(ogShape)
    bl3 = bl[2].reshape(ogShape)
    bl = np.array([bl1, bl2, bl3])
    bl = np.transpose(bl)
    
    blCart = np.transpose(blXYZ)    
    
    blXYZs.append(blCart)
    baselines.append(bl)
    
baselines = np.array(baselines)
blXYZs = np.array(blXYZs)

    
# |-----------------------|
# |--- Get Gamera data ---|
# |-----------------------|
if False:
    allParams = []
    for i in range(nTimes):
        print ('Interpolating time', i+1, 'of', nTimes)
        # Make 1d to do all pts for this time at once
        xvals = baselines[:,i,:,0] 
        ogShape = xvals.shape # [nPairs, nBase]
        xvals1d = xvals.reshape([-1]) # is sph not cart but xyz easier to type
        yvals1d = baselines[:,i,:,1].reshape([-1])
        zvals1d = baselines[:,i,:,2].reshape([-1])
        inVals = np.transpose([xvals1d, yvals1d, zvals1d])
    
        # input is [3, npts]
        myNums = gamRes.getInterpVal(inVals, ['D', 'Bx', 'By', 'Bz'], timestep=(i+1))
        # output is [nParams, npts]
        nowPs = []
        for j in range(4):
            val2d = myNums[j].reshape(ogShape)
            nowPs.append(np.transpose(val2d))
        allParams.append(np.transpose(np.array(nowPs)))
    allParams = np.array(allParams)    # [nT, nPair, nBase, nParams]

    with open('interpRes.pkl', 'wb') as file:
        pickle.dump(allParams, file)    

else:
    with open('interpRes.pkl', 'rb') as file:
        allParams = pickle.load(file)

# |----------------------------|
# |--- Get Faraday Rotation ---|
# |----------------------------|

# FR = sum [(A/f_0^2) ne s_hat dot B]  * len
f0 = 100e6 # frequency in Hz
A = 2.36e4 # rad m^2 /T /s
coeff = 2.64e-17 # rad / G / cm^2
nT2T = 1e-9 # nanotesla to tesla
nT2G = 1e-5
rsun2m = 6.957e8 # Rsun to m
c = 2.99e8
wave0 = c /f0

rms = np.zeros([len(pairs), nTimes])
for j in range(len(pairs)):
    sc1id = pairs[j][0]
    sc2id = pairs[j][1]
    for i in range(nTimes):
        myData = allParams[i,j,:,:] # [baseline, param]
        sc1xyz = miniSats[sc1id][ i, :]
        sc2xyz = miniSats[sc2id][ i, :]
        
        sc_dist = np.sqrt(np.sum((sc1xyz - sc2xyz)**2)) # in Rs
        unitS = (sc1xyz - sc2xyz) / sc_dist

        # Get dens, assuming Gamera cm-3
        dens = myData[:,0] * 1e6
        
        # Get B vector
        Bvec = myData[:,1:] * nT2T
        
        # Dot with ds and sum 
        intBds= np.mean(np.dot(Bvec, unitS)*dens) # mean to mult by full path len
        
        
        # Calc rotation measure
        rm = (A / f0**2) * intBds * sc_dist * rsun2m / wave0**2
        rms[j,i] = rm 
        

fig = plt.figure()
for j in range(len(pairs)):
    plt.plot(gamDTs, rms[j,:], label='Sat'+str(pairs[j][0])+'-Sat'+str(pairs[j][1]))
plt.xlabel('t (hr)')
plt.ylabel('RM (rad/m$^2$)')
plt.legend()
plt.show()
#plt.savefig('RM_t400.png')        
        
        


sat1 = miniSats[0]
sat2 = miniSats[1]
sat3 = miniSats[2]
#sat4 = sats[3]
#sat5 = sats[4]


if True:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sat1[:,0], sat1[:,1], sat1[:,2], 'r--')
    ax.plot(sat2[:,0], sat2[:,1], sat2[:,2], 'b--')
    #ax.plot(sat3[:,0], sat3[:,1], sat3[:,2], 'y--')
    
    for j in range(len(pairs)):
        for i in range(len(gamDTs)):
            ax.plot(blXYZs[j,i,:,0], blXYZs[j,i,:,1], blXYZs[j,i,:,2], 'k--')
    
    #ax.plot(sat4[:,0], sat4[:,1], sat4[:,2], 'g--')
    #ax.plot(sat5[:,0], sat5[:,1], sat5[:,2], 'm--')
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


# Get ne, Bx, By, Bz
#filePath = '/Users/kaycd1/GAMERA/fromDevoj/256serial/wsaCR2261cme05092022.gam.h5'
#res = GAMERAres(filePath)

# Calc FR