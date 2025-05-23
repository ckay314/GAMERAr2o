import numpy as np 
import matplotlib.pyplot as plt
import datetime 
import matplotlib.gridspec as gridspec
import h5py
from astropy.time import Time
#import cdflib



h5File = '/Users/kaycd1/GAMERA/GAMERA_ACE_time_profiles/ACE.sc.h5'


class ACEobsH5:
    "Container for ACE results from a h5 file"
    def __init__(self, fileIn):
            self.MJD    = None
            self.dts    = None
            self.Bx     = None
            self.By     = None
            self.Bz     = None
            self.B      = None
            self.vx     = None
            self.vy     = None
            self.vz     = None
            self.v      = None
            self.tMHD   = None
            self.r      = None
            self.Pt     = None
            self.Pb     = None
            self.Pram   = None
            self.Temp   = None
            self.beta   = None
            
            self.ddt_dts = None
            self.ddt_B   = None
            self.ddt_v   = None
            self.ddt_T   = None
            self.ddt_n   = None
            
            self.tShock   = None
            self.tFRfront = None
            self.tFRend   = None
            self.idx1     = None
            self.idx2     = None
            self.idx3     = None
            
            self.populate(fileIn)
            self.mjd2dts()
            self.calcDDTs()
            self.getBounds()
            
    def populate(self, fileIn):
            # Open the file in read mode
            with h5py.File(fileIn, 'r') as f:
                #print (f.keys())
                # Read MHD vars
                self.Bx = f['Bx'][:]
                self.By = f['By'][:]
                self.Bz = f['Bz'][:]
                self.B  = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
                self.n = f['D'][:] # is number density not mass
                self.vx = f['Vx'][:]
                self.vy = f['Vy'][:]
                self.vz = f['Vz'][:]
                self.v  = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
                # Time not temp 
                self.tMHD  = f['T'][:]
                # Calc temp from pressure
                self.Pt = f['P'][:]
                # think P is nano Pascal = 1e-8 barye
                # k is 1.38e-16 in cgs
                # adding in factor of 2 for Te & Tp, seems to match example better
                self.Temp = self.Pt / self.n / 1.38e-8 / 2 *2
                
                # Calc plasma Beta
                self.Pb = (self.B * 1e-5)**2 / 2
                #self.Beta = self.n * 1.38e-16 * self.Temp * 2 / (self.B * 1e-5)**2
                self.Beta = self.Pt *1e-8 / self.Pb
                
                # Calc ram pressure
                self.Pram = 0.5*(self.v * 1e5)**2 * self.n
                
                # modified Julian date
                self.MJD = f['MJDs'][:]
    
    
    def calcDDTs(self):
        # assuming uniform spacing here
        self.ddt_dts = self.dts[1:-1]
        self.ddt_B   = self.B[2:] - self.B[:-2]
        self.ddt_v   = self.v[2:] - self.v[:-2]
        self.ddt_n   = self.n[2:] - self.n[:-2]
        self.ddt_T   = self.Temp[2:] - self.Temp[:-2]
    
    def mjd2dts(self):
        dtArr = []
        for i in range(len(self.MJD)):
            myDT = Time(self.MJD[i], format='mjd').to_datetime()
            dtArr.append(myDT)
        self.dts = np.array(dtArr)

    def getBounds(self):
        self.ndB = self.ddt_B/np.max(self.ddt_B)
        self.ndv = self.ddt_v/np.max(self.ddt_v)
        self.ndn = self.ddt_n/np.max(self.ddt_n)
        self.ndT = self.ddt_T/np.max(self.ddt_T)
        self.ndCombo = self.ndB * self.ndv * self.ndn * self.ndT
        self.ndCombo = np.abs(self.ndCombo / np.max(self.ndCombo)) * np.sign(self.ndn)
        
        # Get the shock front where the derivative combo peaks
        idx = np.where(self.ndCombo == np.max(self.ndCombo))[0]
        self.idx1 = idx[0]+1 # add 1 to get back to non ddt indexing
        self.tShock = self.ddt_dts[self.idx1-1]
        
        # Get the flux rope front where beta < 1 first time after shock
        idxB = np.where(self.Beta < 1)[0]
        idxFR0 = np.min(idxB[np.where(idxB > (self.idx1))][0])
        self.idx2 = idxFR0
        self.tFRfront = self.dts[idxFR0]
        
        # Get flux rope end where beta > 1 first time after front
        idxE = np.where(self.Beta > 1)[0]
        idxFRe = np.min(idxE[np.where(idxE > idxFR0)][0])
        self.idx3 = idxFRe
        self.tFRend = self.dts[idxFRe]
        
        
    def getValues(self):
        # Print arrival times (already calculated presumably)
        print ('')
        print ('Shock:   ', self.tShock.strftime("%Y-%m-%d %H:%M:%S"))
        print ('FR Front:', self.tFRfront.strftime("%Y-%m-%d %H:%M:%S"))
        print ('FR End:  ', self.tFRend.strftime("%Y-%m-%d %H:%M:%S"))
        print ('')
        
        # Calculate durations
        shockDur = (self.tFRfront - self.tShock ).total_seconds() / 3600.
        FRDur    = ( self.tFRend - self.tFRfront).total_seconds() / 3600.
        print ('Shock Duration (hr): ', '{:4.2f}'.format(shockDur))
        print ('FR Duration (hr):    ', '{:4.2f}'.format(FRDur))
        print ('')

        sIdxs = [self.idx1 + i for i in range(self.idx2 - self.idx1)]
        FRIdxs = [self.idx2 + i for i in range(self.idx3 - self.idx2 +1)]
        
        sB, sBz, sn, sv, sT = self.B[sIdxs], self.Bz[sIdxs], self.n[sIdxs], self.v[sIdxs], np.log10(self.Temp[sIdxs])
        frB, frBz, frn, frv, frT = self.B[FRIdxs], self.Bz[FRIdxs], self.n[FRIdxs], self.v[FRIdxs], np.log10(self.Temp[FRIdxs])
        
        
        # Calculate bonus parameters 
        svxBz = np.abs(self.vx[sIdxs]) * self.Bz[sIdxs] # assume abs of vx bc -x = r
        frvxBz = np.abs(self.vx[FRIdxs]) * self.Bz[FRIdxs]
        
        Bt = np.sqrt(self.By**2 + self.Bz**2)
        thetaC = np.abs(np.atan2(self.By, self.Bz))
        dphidt = np.power(self.v, 4/3) * np.power(Bt, 2/3) * np.power(np.sin(thetaC/2), 8/3)
        Kp = 9.5 - np.exp(2.17676 - 5.2001e-5 * dphidt)
        sKp, frKp = Kp[sIdxs], Kp[FRIdxs]
        
        
        # Calculate upstream values to get shock properties
        # assume it's all in hourly resolution and we have sufficient profile
        # ahead of the shock
        if self.idx1 >= 18:
            upidxs = [self.idx1 - 17 + i for i in range(12)]
            upB, upv, upn, upT = np.mean(self.B[upidxs]), np.mean(self.v[upidxs]), np.mean(self.n[upidxs]), np.mean(self.Temp[upidxs])
            comp = np.mean(sn) / upn
        else:
            print ("error in getting upstream params")
            
        # Get duration of Bz neg for both sheath and FR
        # Assume uniform time resolution
        sBzIdxs = np.where(sBz < 0)[0]    
        frBzIdxs = np.where(frBz < 0)[0]  
        dt = 1 # hours
        # Check if contiguous 
        BzDurs = []
        for idxs in [sBzIdxs, frBzIdxs]:
            if len(idxs) == idxs[-1]-idxs[0]+1:
                BzDurs.append(len(idxs))
            else:
                theseDurs = []
                prevVal = idxs[0]
                startVal = idxs[0]
                for i in range(len(idxs)-1):
                    nextVal = idxs[i+1]
                    if nextVal - prevVal != 1:
                        theseDurs.append(prevVal-startVal+1)
                        startVal = nextVal
                    prevVal = nextVal
                theseDurs.append(prevVal-startVal+1)
                BzDurs.append(theseDurs)
        
        
        print ('|-------- Mean values --------|')
        print ('|--------------Sheath ----FR--|')
        print (' Btot [nT]: ', '{:8.2f}'.format(np.mean(sB)), '{:8.2f}'.format(np.mean(frB)))
        print (' Bz [nT]:   ', '{:8.2f}'.format(np.mean(sBz)), '{:8.2f}'.format(np.mean(frBz)))
        print (' v [km/s]:  ', '{:8.2f}'.format(np.mean(sv)), '{:8.2f}'.format(np.mean(frv)))
        print (' n [cm-3]:  ', '{:8.2f}'.format(np.mean(sn)), '{:8.2f}'.format(np.mean(frn)))
        print (' log(T) [K]:', '{:8.2f}'.format(np.mean(sT)), '{:8.2f}'.format(np.mean(frT)))
        print (' vx Bz:     ', '{:8.2f}'.format(np.mean(svxBz)), '{:8.2f}'.format(np.mean(frvxBz)))
        print (' Kp:        ', '{:8.2f}'.format(np.mean(sKp)), '{:8.2f}'.format(np.mean(frKp)))
        print ('')
        print ('|-------- Max values ---------|')
        print ('|--------------Sheath ----FR--|')
        print (' Btot [nT]: ', '{:8.2f}'.format(np.max(sB)), '{:8.2f}'.format(np.max(frB)))
        print (' Bz [nT]:   ', '{:8.2f}'.format(np.min(sBz)), '{:8.2f}'.format(np.min(frBz)))
        print (' v [km/s]:  ', '{:8.2f}'.format(np.max(sv)), '{:8.2f}'.format(np.max(frv)))
        print (' n [cm-3]:  ', '{:8.2f}'.format(np.max(sn)), '{:8.2f}'.format(np.max(frn)))
        print (' log(T) [K]:', '{:8.2f}'.format(np.max(sT)), '{:8.2f}'.format(np.max(frT)))
        print (' vx Bz:     ', '{:8.2f}'.format(np.min(svxBz)), '{:8.2f}'.format(np.min(frvxBz)))
        print (' Kp:        ', '{:8.2f}'.format(np.max(sKp)), '{:8.2f}'.format(np.max(frKp)))
        print ('')
        
        print ('')
        print ('FR expansion [km/s]:', '{:8.2f}'.format((frv[0]- frv[-1])/2))
        print ('Sheath Compression: ', '{:8.2f}'.format(comp))
        
        print ('')
        print ('Duration with negative Bz (hrs)')
        print ('Sheath:     ', BzDurs[0])
        print ('FR:         ', BzDurs[1])
        
        

    def plotIt(self, figName='temp_profileBounds.png'): 
        vCols = ['r', 'b', 'g', 'k']
        fig = plt.figure(constrained_layout=True, figsize=(8,8))
        gs = fig.add_gridspec(6, 1)
        f1 = fig.add_subplot(gs[0,:])                  
        f2 = fig.add_subplot(gs[1,:], sharex=f1)                  
        f3 = fig.add_subplot(gs[2,:], sharex=f1)                  
        f4 = fig.add_subplot(gs[3,:], sharex=f1)     
        f5 = fig.add_subplot(gs[4,:], sharex=f1)    
        f6 = fig.add_subplot(gs[5,:], sharex=f1)    
         
        f1.plot(self.dts, self.Bx, c=vCols[0], label='x')             
        f1.plot(self.dts, self.By, c=vCols[1], label='y')             
        f1.plot(self.dts, self.Bz, c=vCols[2], label='z')             
        f1.plot(self.dts, self.B, c=vCols[3], label='tot')  
        f1.legend(loc='lower right', ncols=4)
                   
        #f2.plot(self.dts, self.vx, c=vCols[0])             
        #f2.plot(self.dts, self.vy, c=vCols[1])             
        #f2.plot(self.dts, self.vz, c=vCols[2])             
        f2.plot(self.dts, self.v, c=vCols[3])  
        
        f3.plot(self.dts, self.n, c='k')
        f4.plot(self.dts, self.Temp, c='k')
        f5.plot(self.dts, self.Beta, c='k')
        f5.plot(self.dts, self.Beta*0+1, 'k--')
        
        
        
        f6.plot(self.ddt_dts, self.ndB, 'r', label='B', lw=1)
        f6.plot(self.ddt_dts, self.ndv, 'b', label='v', lw=1)
        f6.plot(self.ddt_dts, self.ndn, 'g', label='n', lw=1)
        f6.plot(self.ddt_dts, self.ndT, 'orange', label='T', lw=1)
        f6.plot(self.ddt_dts, self.ndCombo, 'k', label='Combo', lw=2)
        f6.legend(loc='upper left', ncols=2)
        
        
        allfs = [f1, f2, f3, f4, f5, f6]
        for i in range(6):
            myf   = allfs[i]
            ylims = myf.get_ylim()
            myf.plot([self.tShock, self.tShock], ylims, 'k--')
            myf.plot([self.tFRfront, self.tFRfront], ylims, 'k--')
            myf.plot([self.tFRend, self.tFRend], ylims, 'k--')
            if i != 4:
                myf.set_ylim(ylims)
        
        
        f5.set_yscale('log')      
        f1.set_ylabel('B [nT]')  
        f2.set_ylabel('v [km/s]')  
        f3.set_ylabel('n [cm$^{-3}$]')  
        f4.set_ylabel('T [k]')  
        f5.set_ylabel('$\\beta$')  
        f6.set_ylabel('Normalized Derivs')  
        
        plt.savefig(figName)        
            
obs = ACEobsH5(h5File)
obs.getValues()
obs.plotIt()
