import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import h5py

dtor = np.pi / 180.

class GAMERAres:
    def __init__(self, filename):
        '''
        Load the top level data with grid and time
        info but none of the simulation data. Because it
        is slow and I am impatient
        
        has vars Bx, By, Bz, Cs, D, Jx, Jy, Jz, P, Pb, 
                 vx, vy, vz stored
        
        prob need to calc Br, Bphi, Btheta, vr, vphi, vtheta
        
        indexing h5_data['X'][:] for top level
                 h5_data['Step#0']['Bx'][:] for lower levels
                    with shape (256, 64, 256) = (phi, theta, r)
                    phi = lon, 0-360 deg
                    theta = lat, 18-162 deg
                    r = r, 21.5 - 220 Rs
        
                top level are (257, 65, 257) so cell bounds
                whereas mhd vars are at mid points
        
        
        '''
        self.filename = filename
        self.h5_data = h5py.File(self.filename)
        self.keys = list(self.h5_data.keys())
        
        #|--- Load top level data ---|
        self.X = self.h5_data["X"][:]
        self.Y = self.h5_data["Y"][:]
        self.Z = self.h5_data["Z"][:]
 
        self.quantitites = list(self.h5_data["Step#0"].keys())
        self.steps = list(self.h5_data.keys())

        self.r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        self.theta = np.rad2deg(np.arccos(self.Z / self.r))
        self.phi = np.rad2deg(np.arctan2(self.Y, self.X)) % 360
        self.radial_scaling = self.r / 21.5
        
        # Assuming uniform grid spacing, get cell bounds
        self.rgrid = self.r[10,10,:]
        self.thetagrid = self.theta[0,:,10]
        self.phigrid = self.phi[:,10,10]
        # Get mid points
        self.rmid = 0.5*(self.rgrid[1:] + self.rgrid[:-1])
        self.thetamid = 0.5*(self.thetagrid[1:] + self.thetagrid[:-1])
        self.phimid = 0.5*(self.phigrid[1:] + self.phigrid[:-1])
       
        self.mjd_times = self.h5_data["timeAttributeCache"]["MJD"][:]
        self.ntimes = len(self.mjd_times)
        self.strTimes = []
        self.dateTimes = []
        self.timeDeltas = []
        for i in range(self.ntimes):
            dt = Time(self.mjd_times[i], format='mjd', scale='utc').to_datetime()
            strT = dt.strftime("%Y-%m-%d %H:%M")
            self.dateTimes.append(dt)
            self.strTimes.append(strT)
            self.timeDeltas.append((dt - self.dateTimes[0]).total_seconds()/3600.)
        self.strTimes = np.array(self.strTimes)
        self.dateTimes = np.array(self.dateTimes)
        self.timeDeltas = np.array(self.timeDeltas)
        
        # lower levels compressed with szip. pip h5py will not be happy about this. 
        # install via homebrew first (see notes h5syncloadFix.txt)
        #print("Compression filter used:", self.h5_data['Step#0']['Bx'].compression)
        #print("All filters:", self.h5_data['Step#0']['Bx']._filters)
        
            
        #self.Bx = np.array(self.h5_data['Step#0']['Bx'])
        
    def getInterpVal(self, pos, vals, timestep=0):
        '''
        Inputs:
            pos:    a list of points in the form [[lon1, lat1, r1], [lon2, lat2, r2]]
                    where lon in [0,360] deg, lat in [18-162] deg, and r in [21.5- 220] Rs
                    (order is [phi, theta, r])
            
            val:    the MHD variables of interest. from Bx, By, Bz, Cs, D, Jx, Jy, Jz, P, Pb, 
                     vx, vy, vz. should be an array/list [var1, var2, ..]
                    (add in bonus r, theta, phi of B/v tbd)
        
        Output:     an array with [var1, var2] where each var is an array with values of var 
                    for [pt1, pt2, ...]
        
            *** not allowing for the boundary cells on theta/r at the moment. could improve
        
        '''
        if isinstance(pos, list):
            pos = np.array(pos)

        # |--- Check the MHD vars ---|
        for val in vals:
            if val not in ['Bx', 'By', 'Bz', 'Cs', 'D', 'Jx', 'Jy', 'Jz', 'P', 'Pb', 'Vx', 'Vy', 'Vz']:
                sys.exit('Unknown MHD variable '+val+' passed to getInterpVal')
        
        # |--- Check the time step ---|
        if timestep > len(self.steps) - 1:
            sys.exit('Time step '+str(timestep)+' out of range')
        else:    
            myStep = 'Step#'+str(timestep)
        
        # |--- Find the lower indices ---|    
        pidx, tidx, ridx = [], [], []
        for apt in pos:
            # Account for loopiness of phi
            # Phi in midpoint range
            if (apt[0] >= self.phimid[0]) & (apt[0] <= self.phimid[-1]):
                pidx.append(np.max(np.where(self.phimid <= apt[0])))
            # Phi at the loop point
            elif (apt[0] <= self.phimid[0]) or (apt[0] >= self.phimid[-1] -360):
                pidx.append(-1)
            else:
                sys.exit('Cannot process phi/lon value '+str(apt[0])+'. Should be in 0-360 deg')
            
            # Theta
            if (apt[1] >= self.thetamid[0]) & (apt[1] <= self.thetamid[-1]):
                tidx.append(np.max(np.where(self.thetamid <= apt[1])))
            else:
                sys.exit('Cannot process theta/colat value '+str(apt[1])+'. Should be in 18-162 deg')             
            
            # R
            if (apt[2] >= self.rmid[0]) & (apt[2] <= self.rmid[-1]):
                ridx.append(np.max(np.where(self.rmid <= apt[2])))
            else:
                sys.exit('Cannot process r value '+str(apt[1])+'. Should be in 21.9 - 219.6 Rs')    
                
        # Arrayify
        pidx = np.array(pidx)         
        tidx = np.array(tidx)         
        ridx = np.array(ridx)  
        # |--- Get weighting coeffs for interp ---|    
        # Spacing at this point, seems to be uniform but safe to calc instead of assume
        pdel = self.phimid[pidx+1] - self.phimid[pidx]    
        tdel = self.thetamid[tidx+1] - self.thetamid[tidx]    
        rdel = self.rmid[ridx+1] - self.rmid[ridx]    
        
        # Slerp for phi
        fp1 = (pos[:,0] - self.phimid[pidx]) / pdel
        wp = np.sin((1-fp1)*pdel *dtor) / np.sin(pdel*dtor)
        # Slerp for theta
        ft1 = (pos[:,1] - self.thetamid[tidx]) / tdel
        wt = np.sin((1-ft1)*tdel *dtor) / np.sin(tdel*dtor)
        # Basic linterp for 
        wr = 1 -  (pos[:,2] - self.rmid[ridx]) / rdel
        
        # |--- Get vals at midpoints ---|
        # name indexing is ptr with 0 or 1 
        p000 = [[] for i in range(len(vals))]
        p001 = [[] for i in range(len(vals))]
        p010 = [[] for i in range(len(vals))]
        p011 = [[] for i in range(len(vals))]
        p100 = [[] for i in range(len(vals))]
        p101 = [[] for i in range(len(vals))]
        p110 = [[] for i in range(len(vals))]
        p111 = [[] for i in range(len(vals))]
        for j in range(len(vals)):
            for i in range(len(pidx)): # so much for trying to use arrays
                p000[j].append(self.h5_data[myStep][vals[j]][pidx[i],tidx[i],ridx[i]])
                p001[j].append(self.h5_data[myStep][vals[j]][pidx[i],tidx[i],ridx[i]+1])
                p010[j].append(self.h5_data[myStep][vals[j]][pidx[i],tidx[i],ridx[i]])
                p011[j].append(self.h5_data[myStep][vals[j]][pidx[i],tidx[i]+1,ridx[i]+1])
                p100[j].append(self.h5_data[myStep][vals[j]][pidx[i]+1,tidx[i],ridx[i]])
                p101[j].append(self.h5_data[myStep][vals[j]][pidx[i]+1,tidx[i],ridx[i]+1])
                p110[j].append(self.h5_data[myStep][vals[j]][pidx[i]+1,tidx[i],ridx[i]])
                p111[j].append(self.h5_data[myStep][vals[j]][pidx[i]+1,tidx[i]+1,ridx[i]+1])
        
        #|--- Interpolate ---|
        # Compress along p dim
        pm00 = wp * p000 + (1-wp) * p100
        pm01 = wp * p001 + (1-wp) * p101
        pm10 = wp * p010 + (1-wp) * p110
        pm11 = wp * p011 + (1-wp) * p111
        # Compress along t dim
        pmm0 = wt * pm00 + (1-wt) * pm10
        pmm1 = wt * pm01 + (1-wt) * pm11
        # Compress along r dim
        output = wr * pmm0 + (1-wr) * pmm1
        
                
        return output
        
        



#filePath = '/Users/kaycd1/GAMERA/fromDevoj/256serial/wsaCR2261cme05092022.gam.h5'
#res = GAMERAres(filePath)

#fakepts = [[0, 92, 200], [5, 88, 205 ]]
#fakepts = [[10, 92, 200]]
#nums = res.getInterpVal(fakepts, ['Bx', 'Vx'])
#print (nums)
#fig = plt.figure()
#plt.imshow( res.r[:,10,:])
#plt.show()