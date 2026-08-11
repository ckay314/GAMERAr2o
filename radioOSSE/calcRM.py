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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as colors

import geom
# Gamera coords in (lon, colat, r) = (phi, theta, r) = (deg, deg, rs)
# Geom funcs want in same order


Re2Rs = 0.009164023 # 1 rE to rSun
dtor  = np.pi / 180.


# |-----------------|
# |--- Sim Setup ---|
# |-----------------|
# Number of sats to include
nSats = 5

# Starting time within sat trajectories
tIdx = 400

# Option to fake/loadcalc/load/grid
fname = 'largeliss_grid.pkl'

#gridLon = [-30, -20,-10,0,10,20, 30]
#gridLat = [-10, 0, 10]

#gridLon = [-2,0,2]
#gridLat = [-2,0,2]
gridLon = np.linspace(-30,30,61, dtype=int)
gridLat = np.linspace(-10,10,21, dtype=int)

# Number of points in the baselines
nBase = 5

# Sat excel file
xlFile = 'largeliss_demo.xlsx'
# Sat path pickl
pathFile = 'largeliss_MegaGrid.pkl'

# Gamera file
gamFile = '/Users/kaycd1/GAMERA/fromDevoj/256serial/wsaCR2261cme05092022.gam.h5'

# Interp parameter pickle
paramPickle = 'interpResMegaGrid.pkl'    

# RM pickle (gridified and has RM)
RMfile = 'RMresMegaGrid.pkl'


def fakeSats():
    # Not made to interface with rest at this point
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

def getSatPickle(mode, xlFile, outFile, gridLon=None, gridLat=None):
    # |--------------------|
    # |--- Excel2Pickle ---|
    # |--------------------|
    if mode in ['single', 'grid']:
        allSats = pd.read_excel(xlFile, sheet_name=None)
        satnames = list(allSats.keys())
    
        # sat Tags ['Sat 1', 'Sat 2', 'Sat 3', 'Sat 4', 'Sat 5']
        # value Tags ['Date (UTC)', 'X (ECLIPJ2000, km)', 'Y (ECLIPJ2000, km)', 'Z (ECLIPJ2000, km)', 'Xd (ECLIPJ2000, km/s)', 'Yd (ECLIPJ2000, km/s)', 'Zd (ECLIPJ2000, km/s)']
        if mode == 'single':
            gridLon = [0]
            gridLat = [0]
    
        # Proper calculation of coordinates accounting for Earth motion in lat
    
        allRes = {}
        for dlon in gridLon:
            for dlat in gridLat:
                mykey = 'Lon'+str(dlon)+'Lat'+str(dlat)
                print ('Calculating locations for lon', dlon, 'lat', dlat, mykey)
                sats = []
                ts = []
                for sat in ['Sat 1', 'Sat 2', 'Sat 3', 'Sat 4', 'Sat 5']:
                    sats.append([])
                    ts.append([])
                    mySat = allSats[sat]
                    myX, myY, myZ = mySat['X (ECLIPJ2000, km)'], mySat['Y (ECLIPJ2000, km)'], mySat['Z (ECLIPJ2000, km)']
                    myTime = mySat['Date (UTC)']
                    
                    for idx in np.array(range(6))+tIdx-1:
                    #for idx in range(len(myX)):
                        x, y, z = myX[idx], myY[idx], myZ[idx]
                        lon, lat, r = geom.CART2SPH([x,y,z]) # deg, deg, km
                        
                        # format Apr 10 2030, 23:59:59
                        t = datetime.datetime.strptime(myTime[idx], "%b %d %Y, %H:%M:%S" )
                        ts[-1].append(t)
            
                        eclip_frame = GeocentricTrueEcliptic(equinox="J2000", obstime=t)   
                        geo_eclip_coord = SkyCoord(lon=lon*u.deg, lat=(90-lat)*u.deg, distance=r*u.km, frame=eclip_frame)
                        earth_gec = SkyCoord(lon=0*u.deg, lat=0*u.deg, distance=0*u.km, frame=eclip_frame)
        
                        # Assuming that GAMERA results are in Stonyhurst
                        sc0 = geo_eclip_coord.transform_to(HeliographicStonyhurst)
                        Esc0 = earth_gec.transform_to(HeliographicStonyhurst)
                        sclat = sc0.lat - Esc0.lat # Earth at 0 lat
                        sclon = sc0.lon - Esc0.lon # Earth at 0 lon
                        scrad = sc0.radius - Esc0.radius + 1.496e+8 * u.km # Earth at exact 1 AU
                        
                        #stonyhurst_coord = SkyCoord(lon=(sc0.lon + dlon*u.deg), lat=(sc0.lat + dlat*u.deg), radius=sc0.radius, frame="heliographic_stonyhurst")
                        stonyhurst_coord = SkyCoord(lon=(sclon + dlon*u.deg), lat=(sclat + dlat*u.deg), radius=scrad, frame="heliographic_stonyhurst")

                        stonyXYZ = stonyhurst_coord.transform_to(HeliographicStonyhurst(representation_type='cartesian')).cartesian.xyz.to(u.au).to_value()
                        stonyXYZ = stonyXYZ * 215.03
                        sats[-1].append(stonyXYZ)
                    sats[-1] = np.array(sats[-1])
                    ts[-1] = np.array(ts[-1])
        
                # |--- Package it ---|
                res = {}
                res['sats'] = sats
                # ts are all the same, just save one
                res['ts'] = ts[0]
                # Get time difference in hr
                dts = []
                for i in range(len(ts[0])):
                    dts.append((ts[0][i] - ts[0][0]).total_seconds()/3600.)
                res['dts'] = np.array(dts)
                allRes[mykey] = res
    
        #fname = 'largeliss.pkl'
        #if mode == 'grid':
        #    fname = 'largeliss_grid.pkl'
        with open(outFile, 'wb') as file:
            pickle.dump(allRes, file)

    #sats = res['sats'] # [sat1, sat2, ...] where sat is [nTimes, 3] with xyz in stony AU
    #ts = res['ts']
    #dts = res['dts']

def getGamParams(pathFile, gamFile, paramPickle):
    # |-------------------|
    # |--- Load Pickle ---|
    # |-------------------|
    with open(pathFile, 'rb') as file:
        allRes = pickle.load(file)
        
    # |-------------------|
    # |--- Load Gamera ---|
    # |-------------------|
    gamRes = GAMERAres(gamFile)


    collectParams = {}
    for key in allRes:
        print (key)
        res = allRes[key]
        sats = res['sats'] # [sat1, sat2, ...] where sat is [nTimes, 3] with xyz in stony AU
        ts = res['ts']
        dts = res['dts']

        # |--------------------------------|
        # |--- Get sat locs on Gam time ---|
        # |--------------------------------|
        # Gamera has 60 time steps (Step#0 -Step#59)
        # One at very beginning then rest later separated
        # by about one hour
        gamDTs = gamRes.timeDeltas[1:] - gamRes.timeDeltas[1]
        nTimes = len(gamDTs)
        rwT0 = ts[1]
        shiftSatDTs = dts - dts[1]
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
        allParams = []
        for i in range(nTimes):
            #print ('Interpolating time', i+1, 'of', nTimes)
            # Make 1d to do all pts for this time at once
            xvals = baselines[:,i,:,0] 
            ogShape = xvals.shape # [nPairs, nBase]
            xvals1d = xvals.reshape([-1]) # is sph not cart but xyz easier to type
            yvals1d = baselines[:,i,:,1].reshape([-1])
            zvals1d = baselines[:,i,:,2].reshape([-1])
            inVals = np.transpose([xvals1d, yvals1d, zvals1d])
            
            # input is [npts, 3]
            myNums = gamRes.getInterpVal(inVals, ['D', 'Bx', 'By', 'Bz'], timestep=(i+1))
            #print (inVals[0,:], myNums[:,0])
            # output is [nParams, npts]
            nowPs = []
            for j in range(4):
                val2d = myNums[j].reshape(ogShape)
                nowPs.append(np.transpose(val2d))
            allParams.append(np.transpose(np.array(nowPs)))
        allParams = np.array(allParams)    # [nT, nPair, nBase, nParams]
        collectParams[key] = allParams
    
    with open(paramPickle, 'wb') as file:
        pickle.dump(collectParams, file)    


def calcFaraday(satData, paramData, gamRes, saveIt=None):
    # |----------------------------|
    # |--- Get Faraday Rotation ---|
    # |----------------------------|

    # FR = sum [(A/f_0^2) ne s_hat dot B]  * len
    #f0 = 100e6 # frequency in Hz
    A = 2.36e4 # rad m^2 /T /s
    coeff = 2.64e-17 # rad / G / cm^2
    nT2T = 1e-9 # nanotesla to tesla
    rsun2m = 6.957e8 # Rsun to m
    c = 2.99e8 # in m
    
    nKeys = len(satData.keys())
    keys = np.array(list(satData.keys()))
    
    temp = paramData[keys[0]][0,0,:,0]
    nBase = len(list(combinations(temp, 2)))
    #nBase = len(paramData[keys[0]][0,0,:,0]) # should be same for all
    gamDTs = gamRes.timeDeltas[1:] - gamRes.timeDeltas[1]
    nTimes = len(gamDTs)
    
    
    # |--- Convert keys to lon and lat ---|
    lonlats = np.zeros([nKeys, 2])
    for i in range(nKeys):
        key = keys[i]
        splitIt = key.split('Lat')
        lonlats[i,0] = float(splitIt[0].replace('Lon',''))
        lonlats[i,1] = float(splitIt[1])
        
    # |--- Gridify ---|
    myLons = np.unique(lonlats[:,0])
    myLats = np.unique(lonlats[:,1])    
    lons, lats = np.meshgrid(myLons, myLats)
    myShape = lons.shape #[nLat, nLon]
    key2id = {}
    for i in range(nKeys):
        key = keys[i]
        i1 = np.where(myLons == lonlats[i,0])[0]
        i0 = np.where(myLats == lonlats[i,1])[0]
        key2id[key] = [i0, i1]
    
    allRes = np.zeros([9, myShape[0], myShape[1], nBase, nTimes]) # [nParams, lat, lon, baseline, t] params = RM, n1, bx1, by1, bz1, n2, bx2, by2, bz2
    allPairs = {}
    for key in satData.keys():
        
        mySat = satData[key]
        allParams = paramData[key]
        myLL = key2id[key]
        sats = mySat['sats'] # [sat1, sat2, ...] where sat is [nTimes, 3] with xyz in stony AU
        ts = mySat['ts']
        dts = mySat['dts']
    
        rwT0 = ts[1]
        shiftSatDTs = dts - dts[1]
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
        
        
        pairs = list(combinations(range(len(miniSats)), 2))
        

        for j in range(len(pairs)):
            sc1id = pairs[j][0]
            sc2id = pairs[j][1]
        
        
            for i in range(nTimes):
                myData = allParams[i,j,:,:] # [baseline, param]
                sc1xyz = miniSats[sc1id][ i, :]
                sc2xyz = miniSats[sc2id][ i, :]
        
                sc_dist = np.sqrt(np.sum((sc1xyz - sc2xyz)**2)) # in Rs
                unitS = (sc1xyz - sc2xyz) / sc_dist
                sang = geom.CART2SPH(unitS)
            
                # Get dens, assuming Gamera cm-3
                dens = myData[:,0] * 1e6
        
                # Get B vector
                Bvec = myData[:,1:] * nT2T
                Bmag = np.sqrt(np.sum(Bvec**2, axis=1))
                Bunit = Bvec[0,:] / Bmag[0]
                Bang = geom.CART2SPH(Bunit)
                ang = np.dot(Bunit, unitS)
                # Dot with ds and sum 
                intBds= np.mean(np.dot(Bvec, unitS)*dens) # mean to mult by full path len
                # Calc rotation measure
                rm = (A / c**2) * intBds * sc_dist * rsun2m 
                #srms[j,i] = rm
                # Rot measure
                
                allRes[0, myLL[0], myLL[1], j, i] = rm
                # N 
                allRes[1, myLL[0], myLL[1], j, i] = myData[:,0][0]
                allRes[5, myLL[0], myLL[1], j, i] = myData[:,0][-1]
                # Bx
                allRes[2, myLL[0], myLL[1], j, i] = myData[:,1][0]
                allRes[6, myLL[0], myLL[1], j, i] = myData[:,1][-1]
                # By
                allRes[3, myLL[0], myLL[1], j, i] = myData[:,2][0]
                allRes[7, myLL[0], myLL[1], j, i] = myData[:,2][-1]
                # Bz
                allRes[4, myLL[0], myLL[1], j, i] = myData[:,3][0]
                allRes[8, myLL[0], myLL[1], j, i] = myData[:,3][-1]
                
        allPairs[key] = pairs
    if type(saveIt) != type(None):
        outStuff = {}
        outStuff['params'] = allRes
        outStuff['pairs'] = allPairs
        with open(saveIt, 'wb') as file:
            pickle.dump(outStuff, file)
        
    return allRes, allPairs
    
def plotFaraday( myRM, pairs, time=None, figName='temp.png'):
    if type(time) == type(None):
        time = range(len(myRM[0,:]))
     
    fig = plt.figure()
    for j in range(len(pairs)):
        plt.plot(time, myRM[j,:]/1e-5, label='Sat'+str(pairs[j][0])+'-Sat'+str(pairs[j][1]))
    plt.xlabel('t (hr)')
    plt.ylabel('RM (10$^{-5}$ rad/m$^2$)')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(figName)      
    plt.close()
        

def plotProfiles(satData, paramData, gamRes, keys):
    # |----------------------------|
    # |--- Get Faraday Rotation ---|
    # |----------------------------|

    # FR = sum [(A/f_0^2) ne s_hat dot B]  * len
    #f0 = 100e6 # frequency in Hz
    A = 2.36e4 # rad m^2 /T /s
    coeff = 2.64e-17 # rad / G / cm^2
    nT2T = 1e-9 # nanotesla to tesla
    nT2G = 1e-5
    rsun2m = 6.957e8 # Rsun to m
    c = 2.99e8
    #wave0 = c /f0
    
    fig, axes = plt.subplots(6, 3, figsize=(10, 8), sharey='row')
    colMap = {'Lon0Lat0':'k', 'Lon0Lat-10':'b', 'Lon0Lat10':'r'}
    
    for key in keys:
        mySat = satData[key]
        allParams = paramData[key]
        myCol = colMap[key]
    
        sats = mySat['sats'] # [sat1, sat2, ...] where sat is [nTimes, 3] with xyz in stony AU
        ts = mySat['ts']
        dts = mySat['dts']
    
        gamDTs = gamRes.timeDeltas[1:] - gamRes.timeDeltas[1]
        nTimes = len(gamDTs)
        rwT0 = ts[1]
        shiftSatDTs = dts - dts[1]
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
    
    
        pairs = list(combinations(range(len(miniSats)), 2))
    
        nTimes = allParams.shape[0]
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
                sang = geom.CART2SPH(unitS)
            
                # Get dens, assuming Gamera cm-3
                dens = myData[:,0] * 1e6
        
                # Get B vector
                Bvec = myData[:,1:] * nT2T
                Bmag = np.sqrt(np.sum(Bvec**2, axis=1))
                Bunit = Bvec[0,:] / Bmag[0]
                Bang = geom.CART2SPH(Bunit)
                ang = np.dot(Bunit, unitS)
                # Dot with ds and sum 
                intBds= np.mean(np.dot(Bvec, unitS)*dens) # mean to mult by full path len
                # Calc rotation measure
                rm = (A / c**2) * intBds * sc_dist * rsun2m 
                rms[j,i] = rm 
            
                if i > 20:
                    #print (i, ang, dens[0]/1e6, Bmag[0], dens[0]*Bmag[0]*ang  )
                    axes[0,j].scatter(i, rm/1e-5, color=myCol, s=10)
                    axes[1,j].scatter(i, np.mean(dens)/1e6, color=myCol, s=10)
                    axes[2,j].scatter(i, np.mean(Bmag)/1e-9, color=myCol, s=10)
                    axes[3,j].scatter(i, np.mean(ang), color=myCol, s=10)
                    axes[4,j].scatter(i, sang[0], color=myCol, s=10)
                    axes[5,j].scatter(i, 90-sang[1], color=myCol, s=10)
                    axes[4,j].scatter(i, Bang[0], color=myCol, s=10, marker='x')
                    axes[5,j].scatter(i, 90-Bang[1], color=myCol, s=10, marker='x')
                
    axes[0,0].set_ylabel('RM (10$^{-5}$)')
    axes[1,0].set_ylabel('n (cm$^{-3}$)')
    axes[2,0].set_ylabel('B (nT)')
    axes[3,0].set_ylabel('$\\hat{B} \\cdot \\hat{s}$')
    axes[4,0].set_ylabel('Lon Angle ($^{\\circ}$)')
    axes[5,0].set_ylabel('Lat Angle ($^{\\circ}$)($^{\\circ}$)')
    axes[0,0].set_title('Sat 0 - Sat 1')
    axes[0,1].set_title('Sat 0 - Sat 2')
    axes[0,2].set_title('Sat 1 - Sat 2')
    axes[0,0].text(50, 8, '+10', c='r')
    axes[0,0].text(50, 6, 'Eq', c='k')
    axes[0,0].text(50, 4, '-10', c='b')
    for i in range(5):
        axes[i,-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('profileComp.png')

def RMmaps(rms, pairs):
    nKeys = len(rms.keys())
    keys = np.array(list(rms.keys()))
    nBase = len(pairs[keys[0]]) # should be same for all
    
    # |--- Convert keys to lon and lat ---|
    lonlats = np.zeros([nKeys, 2])
    for i in range(nKeys):
        key = keys[i]
        splitIt = key.split('Lat')
        lonlats[i,0] = float(splitIt[0].replace('Lon',''))
        lonlats[i,1] = float(splitIt[1])
        
    # |--- Gridify ---|
    myLons = np.unique(lonlats[:,0])
    myLats = np.unique(lonlats[:,1])    
    lons, lats = np.meshgrid(myLons, myLats)
    myShape = lons.shape #[nLat, nLon]
        
    # |--- Get max abs and range ---|
    maxs = np.zeros([myShape[0], myShape[1], nBase])
    rngs = np.zeros([myShape[0], myShape[1], nBase])
    for i in range(nKeys):
        key = keys[i]
        myRM = rms[key] # [nBase, nTime]
        i1 = np.where(myLons == lonlats[i,0])
        i0 = np.where(myLats == lonlats[i,1])
        maxs[i0,i1,:] =  np.max(np.abs(myRM), axis=1)
        
    
    fig, axes = plt.subplots(4, 3, figsize=(8, 6), sharex=True, sharey=True)
    faxes = axes.flatten()
    logMax =  np.log10(maxs)
    for i in range(nBase):
        faxes[i].contourf(lons,lats, logMax[:,:,i], vmin=-7, vmax=-3)
        faxes[i].set_aspect('equal')
        faxes[i].set_title('Baseline '+str(i))
    # Overall max
    ovMax = np.max(np.log10(maxs), axis=2)
    im = faxes[10].contourf(lons,lats, ovMax, vmin=-7, vmax=-3)
    faxes[10].set_aspect('equal')
    faxes[10].set_title('Baseline '+str(i))
    cbar = fig.colorbar(im, ax=axes, shrink=0.5, location='top')
    cbar.set_label('log$_{10}$ Max RM')
    
    # who max
    whoMax = np.zeros(myShape)
    for i in range(len(myLats)):
        for j in range(len(myLons)):
            whoMax[i,j] = np.where(logMax[i,j,:] == ovMax[i,j])[0][0]
            
    im2 = faxes[11].contourf(lons,lats, whoMax, levels=range(nBase), cmap='plasma')
    faxes[11].set_aspect('equal')
    faxes[11].set_title('Max Baseline')
    cbar2 = fig.colorbar(im2, ax=faxes[11], shrink=0.7)
    #plt.show()
    plt.savefig('RMmaps.png')
        
def getMask(rmRes, plotIt=False, nMax=50, bLims=[10,30] ):
    rms = rmRes['params']
    
    shape = rms.shape
    nLat = shape[1]
    nLon = shape[2]
    nBase = shape[3]
    nT = shape[4]
    #for j in range(9):
    #    print (j, np.max(rms[j,:,:,:]))
    nidx = np.array([1,5])
    mask = np.zeros([nLat,nLon])
    maxNs = np.zeros([nLat,nLon])
    maxBs = np.zeros([nLat,nLon])
    for i in range(nLat):
        for j in range(nLon):
            myB1 = np.sqrt(rms[2,i,j,:, :]**2 + rms[3,i,j,:, :]**2 + rms[4,i,j,:, :]**2)
            myB2 = np.sqrt(rms[6,i,j,:, :]**2 + rms[7,i,j,:, :]**2 + rms[8,i,j,:, :]**2)
            maxN = np.max(rms[nidx,i,j,:, :])
            maxB = np.max([myB1, myB2])
            incIt = 0
            if (maxB >= bLims[0]) and (maxB <= bLims[1]) and (maxN <=nMax):
                incIt = 1
            mask[i,j] = incIt
            maxNs[i,j] = maxN
            maxBs[i,j] = maxB
            
    
    if plotIt:        
        fig, axes = plt.subplots(3,1, sharex=True,sharey=True, figsize=(6,6))
        im1 = axes[0].imshow(mask, vmin= 0.5, vmax=0.5, origin='lower', cmap='magma')
        axes[0].contour(maxNs, levels=[50], colors = 'DodgerBlue', linewidths=2, linestyles='dashed')
        axes[0].contour(maxBs, levels=[10,30], colors = 'DarkCyan', linewidths=2, linestyles='dashed')
        ticks = [0.45, 0.55]
        tickLabs = ['Exc','Inc']
        cbar1 = plt.colorbar(im1, ticks = ticks)
        cbar1.set_ticklabels(tickLabs)
        cbar1.set_label('Mask')
        im2 = axes[1].imshow(maxNs, origin='lower', cmap='magma')
        axes[1].contour(maxNs, levels=[50], colors = 'DodgerBlue', linewidths=2, linestyles='dashed')
        cbar2 = plt.colorbar(im2)
        cbar2.set_label('Max n (cm$^{-3}$)')
        im3 = axes[2].imshow(maxBs, origin='lower', cmap='magma')
        axes[2].contour(maxBs, levels=[10,30], colors = 'DarkCyan', linewidths=2, linestyles='dashed')
        cbar3 = plt.colorbar(im3)
        cbar3.set_label('Max B (nT)')
        plt.tight_layout()
        #plt.show()
        plt.savefig('GameraMask.png')
    return mask

def maskedHistos(rmRes, mask):
    res = rmRes['params']
    shape = res.shape
    nLat = shape[1]
    nLon = shape[2]
    nBase = shape[3]
    nT = shape[4]
    
    # Collect good profiles
    goodRes = []
    whereGood = []
    for i in range(nLat):
        for j in range(nLon):
            if mask[i,j]:
                whereGood.append([i,j])
                goodRes.append(res[:,i,j,:,:])
    goodRes = np.array(goodRes) # [profile, param, base, time]
    gRshape = goodRes.shape
    earlyVals = goodRes[:,:,:,:20]
    medRMearly = np.median(goodRes[:,0,:,:])
    cutoff = 1e-6# 5 * medRMearly # testing showed 5x good to isolate CME -> 4.9e-7
    
    
    # Collect params from good profiles
    allrms = []
    allns = []
    allbs = []
    dupns = []
    dupbs = []
    maxrms = []
    maxns = []
    maxbs = []
    matchns = []
    matchbs = []
    for i in range(gRshape[0]):
        myRM = np.abs(goodRes[i,0,:,:])
        myNs = np.concatenate((goodRes[i,1,:,:], goodRes[i,5,:,:]), axis=0)
        b1 = np.sqrt(goodRes[i,2,:,:]**2 + goodRes[i,3,:,:]**2 + goodRes[i,4,:,:]**2)
        b2 = np.sqrt(goodRes[i,6,:,:]**2 + goodRes[i,7,:,:]**2 + goodRes[i,8,:,:]**2)
        myBs = np.concatenate((b1, b2), axis=0)
        
        isCME = np.where(np.abs(myRM[0,:]) > cutoff) # assume same for all baseline
        myMaxRMs = np.max(myRM[:,isCME[0]], axis=1)
        for j in range(nBase):
            matchIdx = np.where(myRM[j,:] == myMaxRMs[j])[0][0]
            matchn = 0.5*(goodRes[i,1,j,matchIdx] + goodRes[i,5,j,matchIdx])
            matchb = 0.5*(b1[j,matchIdx] + b2[j,matchIdx])
            matchns.append(matchn)
            matchbs.append(matchb)
        allrms.extend(myRM[:,isCME].flatten())
        maxrms.extend(myMaxRMs.flatten())
        
        
        
        # Have plasma vals at both ends of the baseline -> need to take
        # only the unique ones to avoid duplicates from same sat appearing
        # in multiple baselines
        meanN = 0.5 * (goodRes[i,1,:,:] + goodRes[i,5,:,:])
        uniqNs = np.unique(myNs, axis=0)
        myMaxNs = np.max(uniqNs[:,isCME[0]], axis=1)
        mydupmaxn = np.max(meanN[:,isCME[0]], axis=1)
        allns.extend(uniqNs[:,isCME].flatten())
        maxns.extend(myMaxNs.flatten())
        dupns.extend(meanN[:,isCME].flatten())
        #dupmaxns.extend(mydupmaxn.flatten())
        
        meanBs = 0.5 * (b1 + b2)
        uniqBs = np.unique(myBs, axis=0)
        myMaxBs = np.max(uniqBs[:,isCME[0]], axis=1)
        
        
        mydupmaxb = np.max(meanBs[:,isCME[0]], axis=1)
        allbs.extend(uniqBs[:,isCME].flatten())
        maxbs.extend(myMaxBs.flatten())
        dupbs.extend(meanBs[:,isCME].flatten())
        #dupmaxbs.extend(mydupmaxb.flatten())
        
    
    # Will have twice the number of baselines as sats
    # so num rm = 2 x num plasmas
    allrms = np.array(allrms)
    allns = np.array(allns)
    allbs = np.array(allbs)
    maxrms = np.array(maxrms)
    maxns = np.array(maxns)
    maxbs = np.array(maxbs)
    
    # Make some histos
    if True:
        fig, axes = plt.subplots(3,2, sharex = 'row')
        axes[0,0].hist(np.log10(allrms), range=(-6,-4))
        axes[1,0].hist(allns)
        axes[2,0].hist(allbs)
        axes[0,1].hist(np.log10(maxrms), range=(-6,-4))
        axes[1,1].hist(maxns)
        axes[2,1].hist(maxbs)
    
        for i in range(2):
            axes[0,i].set_xlabel('log$_{10}$ RM (rad/m$^2$)')
            axes[1,i].set_xlabel('n (cm$^{-3}$)')
            axes[2,i].set_xlabel('B (nT)')
            for j in range(3):
                axes[j,i].set_ylabel('Counts')
        axes[0,0].set_title('Full Time Profile')
        axes[0,1].set_title('Max of Time Profile')
    
        plt.tight_layout()
        plt.savefig('GameraHistos.png')
    
    # Make a scatter plot    
    if False:
        fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
        axes[0].scatter(dupns, dupbs, c=np.log10(allrms), s=10, vmin=-6, vmax=-4, cmap='plasma')
        im = axes[1].scatter(matchns, matchbs, c=np.log10(maxrms), s=10, vmin=-6, vmax=-4, cmap='plasma')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="7%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('log$_{10}$ RM (rad/m$^2$)')
        for i in range(2):
            axes[i].set_xlabel('n (cm$^{-3}$)')
            axes[i].set_ylabel('B (nT)')
        axes[0].set_title('Full Time Profile')
        axes[1].set_title('Max of Time Profile')
        plt.tight_layout()
        #plt.show()
        plt.savefig('GameraScatter.png')

def plotLLProfileSpread(rmRes, mask, mode='lon'):
    res = rmRes['params']
    shape = res.shape
    nLat = shape[1]
    nLon = shape[2]
    nBase = shape[3]
    nT = shape[4]
    ts = np.array(range(nT))
     
    cutoff = 1e-6 # ~ 10x med of early rm, from maskedHistos
    fig, axes = plt.subplots(5,2, sharex=True, sharey=True)
    axf = axes.flatten()
    
    if mode.lower() == 'lon':
        nNorm = nLon
        figName = 'GameraProfilesLon.png'
        labelTag = 'Longitude'
    elif mode.lower() == 'lat':
        nNorm = nLat
        figName = 'GameraProfilesLat.png'
        labelTag = 'Latitude'
    
    cmap = plt.colormaps['plasma'] 
    lw = 0.5
    if mode.lower() == 'lon':
        for i in range(nLat):
            for j in range(nLon):
                if mask[i,j]:
                    for k in range(nBase):
                        myRM = res[0,i,j,k,:]
                        isCME = np.where(np.abs(myRM) >=cutoff)[0]
                        axf[k].plot(np.log10(np.abs(res[0,i,j,k,isCME])), c=cmap(j/nNorm))
    else:
        for j in range(nLon):
            for i in range(nLat):
                if mask[i,j]:
                    for k in range(nBase):
                        myRM = res[0,i,j,k,:]
                        isCME = np.where(np.abs(myRM) >=cutoff)[0]
                        axf[k].plot(np.log10(np.abs(res[0,i,j,k,isCME])), c=cmap(i/nNorm), lw=lw)
    for i in [-2,-1]:
        axf[i].set_xlabel('Time (hr)')
    for j in [0,2,4,6,8]:
        axf[j].set_ylabel('log$_{10}$ RM\n(rad/m$^2$)')
    for ax in axf:
        ax.set_yticks([-6,-5,-4])
                    
    norm = colors.Normalize(vmin=-(nNorm-1)/2, vmax=(nNorm-1)/2)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(right=0.8, wspace=0.05, top=0.95)
    cbar_ax = fig.add_axes([0.83, 0.2, 0.05, 0.6])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(labelTag + '($^{\\circ}$)')
    #plt.tight_layout()
    #lt.show()
    plt.savefig(figName)

def plotAvgProf(rmRes,mask):
    res = rmRes['params']
    shape = res.shape
    nLat = shape[1]
    nLon = shape[2]
    nBase = shape[3]
    nT = shape[4]
    ts = np.array(range(nT))
     
    cutoff = 1e-6 # ~ 10x med of early rm, from maskedHistos  
    
    # Collect profiles into single 2d array with start aligned
    # pad back end with -9999
    nGood = int(np.sum(mask))
    subnT = 20
    allProfiles = np.zeros([nGood,nBase, subnT]) -9999# assume nobody has CME dur > 20
    allLens = []
    counter = 0
    for i in range(nLat):
        for j in range(nLon):
            if mask[i,j]: 
                for k in range(nBase):
                    myRM = np.abs(res[0,i,j,k,:])
                    isCME = np.where(np.abs(myRM) > cutoff)[0] 
                    allLens.append(len(isCME))
                    allProfiles[counter,k,:allLens[-1]] = myRM[isCME]
                counter +=1
    
    meanVals = np.zeros([nBase, subnT]) -9999
    stdVals = np.zeros([nBase, subnT]) -9999
    for i in range(nBase):
        for j in range(subnT):
            theseVals = allProfiles[:,i,j]
            isCME = np.where(theseVals != -9999)[0]
            if len(isCME) !=0:
                meanVals[i,j] = np.mean(np.abs(theseVals[isCME]))
                stdVals[i,j] = np.std(np.abs(theseVals[isCME]))
                
    cmap = plt.colormaps['plasma']
    fig, ax = plt.subplots(1,1)
    for i in range(nBase):
        mymeans = meanVals[i,:]
        mystds = stdVals[i,:]
        goodMeans = np.where(mymeans != -9999)[0]
        #plt.plot(mymeans[goodMeans], c=cmap(i/nBase))
        #plt.fill_between(range(len(goodMeans)),mymeans[goodMeans]-mystds[goodMeans],mymeans[goodMeans]+mystds[goodMeans],alpha=0.25, color=cmap(i/nBase))
        plt.plot(np.log10(mymeans[goodMeans]), c=cmap(i/nBase))
        plt.fill_between(range(len(goodMeans)),np.log10(mymeans[goodMeans]-mystds[goodMeans]),np.log10(mymeans[goodMeans]+mystds[goodMeans]),alpha=0.15, color=cmap(i/nBase))
    ax.set_ylabel('log$_{10}$ RM\n(rad/m$^2$)')
    ax.set_xlabel('Time (hr)')
    plt.tight_layout()
    plt.savefig('GameraMeanBaselines.png')
if False:
    sat1 = miniSats[0]
    sat2 = miniSats[1]
    sat3 = miniSats[2]
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


# |--------------------|
# |--- Main Process ---|
# |--------------------|


#|--- Function calls ---|
# Get satellite paths pickle
#getSatPickle('grid', xlFile, pathFile, gridLon=gridLon, gridLat=gridLat)

# Get interpolated data
#getGamParams(pathFile, gamFile, paramPickle)

# Calculate RM
if False:
    with open(pathFile, 'rb') as file:
        satData = pickle.load(file)
    with open(paramPickle, 'rb') as file:
        paramData = pickle.load(file)
    gamRes = GAMERAres(gamFile)
    print ('Done loading')
    allRMs, allPairs = calcFaraday(satData, paramData, gamRes, saveIt=RMfile)

with open(RMfile, 'rb') as file:
    RMres = pickle.load(file)
mask = getMask(RMres, plotIt=False)

#maskedHistos(RMres, mask)

plotLLProfileSpread(RMres,mask, mode='lon')

#plotAvgProf(RMres,mask)


# |-----------------|
# |--- Plot data ---|
# |-----------------|
'''with open(pathFile, 'rb') as file:
    satData = pickle.load(file)
with open(paramPickle, 'rb') as file:
    paramData = pickle.load(file)

gamRes = GAMERAres(gamFile)
#for i in range(59):
#    print (gamRes.getInterpVal([[0.1,110,213]],'D', timestep=i))

#plotProfiles(satData, paramData, gamRes, ['Lon0Lat0', 'Lon0Lat-10', 'Lon0Lat10'])'''

# |--- RM profiles for each lat/lon ---|
if False:
    gamDTs = gamRes.timeDeltas[1:] - gamRes.timeDeltas[1]
    for key in paramData:
        #mySat = satData[key]
        #myParams = paramData[key]
        myRM = allRMs[key]
        myPairs = allPairs[key]
        figName = 'RM_t400_'+key+'.png'
        plotFaraday( myRM, myPairs, time=gamDTs, figName=figName)


