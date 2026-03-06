import numpy as np 
import matplotlib.pyplot as plt
import datetime 
import matplotlib.gridspec as gridspec
import h5py
from astropy.time import Time
#import cdflib




class ACEobsH5:
    "Container for GAMERA results at ACE from a h5 file"
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
            
            # First derivatives
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
        try:
            idx = np.where(self.ndCombo == np.max(self.ndCombo))[0]
            self.idx1 = idx[0]+1 # add 1 to get back to non ddt indexing
            self.tShock = self.ddt_dts[self.idx1-1]
        except:
            print('Cannot determine shock front')
            self.tShock = None
        
        # Get the flux rope front, ideally where beta < 1 first time after shock
        # but some simulations are all low beta so that won't work
        useBeta = True
        if self.tShock:
            if self.Beta[self.idx1] < 1: useBeta = False
        # Beta version    
        if useBeta:
            try:
                idxB = np.where(self.Beta < 1)[0]
                idxFR0 = np.min(idxB[np.where(idxB > (self.idx1))][0])
                self.idx2 = idxFR0
                self.tFRfront = self.dts[idxFR0]
            except:
                print('Cannot determine FR front')
                self.tFRfront = None
        # Deriv of density as backup
        else:
            try:
                subddtn = self.ddt_n[self.idx1-1:]
                idxN = np.where(subddtn == np.min(subddtn))[0][0]
                self.tFRfront = self.ddt_dts[self.idx1-1:][idxN]
                self.idx2 = np.where(self.dts == self.tFRfront)[0][0]
            except:
                print('Cannot determine FR front')
                self.tFRfront = None
                
            
        # Get flux rope end where beta > 1 first time after front
        subBeta = self.Beta[self.idx2+2:] # add a couple points to rm front 
        subtimes = self.dts[self.idx2+2:]
        maxBeta = np.max(subBeta)
        if maxBeta > 1: 
            try:
                idxE = np.where(self.Beta > 1)[0]
                idxFRe = np.min(idxE[np.where(idxE > idxFR0)][0])
                self.idx3 = idxFRe
                self.tFRend = self.dts[idxFRe]
            except:
                print('Cannot determine FR end')
                self.tFRend = None
        else:
            try:
                idxE = np.where(subBeta == maxBeta)[0]
                self.tFRend = subtimes[idxE[0]]
                self.idx3 = np.where(self.dts == self.tFRend)[0][0]
            except:
                print('Cannot determine FR end')
                self.tFRend = None
        
    def getValues(self, fileName=None):
        outThing = ''
        
        # Determine if writing to file or printing
        if type(fileName) == type(None):
            printIt = True
        else:
            printIt = False
            f = open(fileName, 'w')

        if printIt:
            # Print arrival times (already calculated presumably)
            print ('')
            print ('Shock:    ', self.tShock.strftime("%Y-%m-%d %H:%M:%S"))
            print ('FR Front: ', self.tFRfront.strftime("%Y-%m-%d %H:%M:%S"))
            print ('FR End:   ', self.tFRend.strftime("%Y-%m-%d %H:%M:%S"))
            print ('')
        else:
            f.write('Shock:    ' +  self.tShock.strftime("%Y-%m-%d %H:%M:%S") + '\n')
            f.write('FR Front: ' + self.tFRfront.strftime("%Y-%m-%d %H:%M:%S") + '\n')
            f.write('FR End:   ' + self.tFRend.strftime("%Y-%m-%d %H:%M:%S") + '\n')
            f.write('\n')
        
        # Get fractional year for dates
        myYR = datetime.datetime(self.tFRfront.year,1,1,0,0)
        outThing += str(self.tFRfront.year) + ' '
        if self.tShock:
            outThing += '{:11.6f}'.format((self.tShock - myYR).total_seconds()/3600./24) + ' '
        else:
            outThing += '-999'.rjust(11) + ' '
            
        if self.tFRfront:
            outThing += '{:11.6f}'.format((self.tFRfront - myYR).total_seconds()/3600./24) + ' '
        else:
            outThing += '-999'.rjust(11) + ' '    

        if self.tFRend:
            outThing += '{:11.6f}'.format((self.tFRend - myYR).total_seconds()/3600./24) + ' '
        else:
            outThing += '-999'.rjust(11) + ' '    
             
        # Calculate durations
        shockDur = (self.tFRfront - self.tShock ).total_seconds() / 3600.
        FRDur    = ( self.tFRend - self.tFRfront).total_seconds() / 3600.
        if printIt:
            print ('Shock Duration (hr): '+ '{:6.2f}'.format(shockDur))
            print ('FR Duration (hr):    '+ '{:6.2f}'.format(FRDur))
            print ('')
        else:
            f.write('Shock Duration (hr): '+ '{:6.2f}'.format(shockDur)+ '\n')
            f.write('FR Duration (hr):    '+ '{:6.2f}'.format(FRDur)+ '\n')
            f.write('\n')
            
        outThing += '{:6.2f}'.format(shockDur)+' '+'{:6.2f}'.format(FRDur)+' '

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
            if len(idxs) != 0:
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
            else:
                BzDurs.append(0)
        
        if printIt:
            print ('|-------- Mean values --------|')
            print ('|--------------Sheath ----FR--|')
            print (' Btot [nT]: ', '{:10.2f}'.format(np.mean(sB)), '{:10.2f}'.format(np.mean(frB)))
            print (' Bz [nT]:   ', '{:10.2f}'.format(np.mean(sBz)), '{:10.2f}'.format(np.mean(frBz)))
            print (' v [km/s]:  ', '{:10.2f}'.format(np.mean(sv)), '{:10.2f}'.format(np.mean(frv)))
            print (' n [cm-3]:  ', '{:10.2f}'.format(np.mean(sn)), '{:10.2f}'.format(np.mean(frn)))
            print (' log(T) [K]:', '{:10.2f}'.format(np.mean(sT)), '{:10.2f}'.format(np.mean(frT)))
            print (' vx Bz:     ', '{:10.2f}'.format(np.mean(svxBz)), '{:10.2f}'.format(np.mean(frvxBz)))
            print (' Kp:        ', '{:10.2f}'.format(np.mean(sKp)), '{:10.2f}'.format(np.mean(frKp)))
            print ('')
            print ('|-------- Max values ---------|')
            print ('|--------------Sheath ----FR--|')
            print (' Btot [nT]: ', '{:10.2f}'.format(np.max(sB)), '{:10.2f}'.format(np.max(frB)))
            print (' Bz [nT]:   ', '{:10.2f}'.format(np.min(sBz)), '{:10.2f}'.format(np.min(frBz)))
            print (' v [km/s]:  ', '{:10.2f}'.format(np.max(sv)), '{:10.2f}'.format(np.max(frv)))
            print (' n [cm-3]:  ', '{:10.2f}'.format(np.max(sn)), '{:10.2f}'.format(np.max(frn)))
            print (' log(T) [K]:', '{:10.2f}'.format(np.max(sT)), '{:10.2f}'.format(np.max(frT)))
            print (' vx Bz:     ', '{:10.2f}'.format(np.min(svxBz)), '{:10.2f}'.format(np.min(frvxBz)))
            print (' Kp:        ', '{:10.2f}'.format(np.max(sKp)), '{:10.2f}'.format(np.max(frKp)))
            print ('')
        
            print ('')
            print ('FR expansion [km/s]:', '{:10.2f}'.format((frv[0]- frv[-1])/2))
            print ('Sheath Compression: ', '{:10.2f}'.format(comp))
        
            print ('')
            print ('Duration with negative Bz (hrs)')
            print ('Sheath:     ', BzDurs[0])
            print ('FR:         ', BzDurs[1])
            
        else:
            f.write('|-------- Mean values --------|'+ '\n')
            f.write('|--------------Sheath ----FR--|'+ '\n')
            f.write(' Btot [nT]: '+ '{:10.2f}'.format(np.mean(sB)) + '{:10.2f}'.format(np.mean(frB))+ '\n')
            f.write(' Bz [nT]:   '+ '{:10.2f}'.format(np.mean(sBz))+ '{:10.2f}'.format(np.mean(frBz))+ '\n')
            f.write(' v [km/s]:  '+ '{:10.2f}'.format(np.mean(sv))+ '{:10.2f}'.format(np.mean(frv))+ '\n')
            f.write(' n [cm-3]:  '+ '{:10.2f}'.format(np.mean(sn)) + '{:10.2f}'.format(np.mean(frn))+ '\n')
            f.write(' log(T) [K]:'+ '{:10.2f}'.format(np.mean(sT)) + '{:10.2f}'.format(np.mean(frT))+ '\n')
            f.write(' vx Bz:     '+ '{:10.2f}'.format(np.mean(svxBz)) + '{:10.2f}'.format(np.mean(frvxBz))+ '\n')
            f.write(' Kp:        '+ '{:10.2f}'.format(np.mean(sKp)) + '{:10.2f}'.format(np.mean(frKp))+ '\n')
            f.write('\n')
            f.write('|-------- Max values ---------|'+ '\n')
            f.write('|--------------Sheath ----FR--|'+ '\n')
            f.write(' Btot [nT]: '+ '{:10.2f}'.format(np.max(sB)) + '{:10.2f}'.format(np.max(frB))+ '\n')
            f.write(' Bz [nT]:   '+ '{:10.2f}'.format(np.min(sBz)) + '{:10.2f}'.format(np.min(frBz))+ '\n')
            f.write(' v [km/s]:  '+ '{:10.2f}'.format(np.max(sv)) + '{:10.2f}'.format(np.max(frv))+ '\n')
            f.write(' n [cm-3]:  '+ '{:10.2f}'.format(np.max(sn)) + '{:10.2f}'.format(np.max(frn))+ '\n')
            f.write(' log(T) [K]:'+ '{:10.2f}'.format(np.max(sT)) + '{:10.2f}'.format(np.max(frT))+ '\n')
            f.write(' vx Bz:     '+ '{:10.2f}'.format(np.min(svxBz)) + '{:10.2f}'.format(np.min(frvxBz))+ '\n')
            f.write(' Kp:        '+ '{:10.2f}'.format(np.max(sKp)) + '{:10.2f}'.format(np.max(frKp))+ '\n')
            f.write('\n')
           
            f.write('FR expansion [km/s]:'+ '{:10.2f}'.format((frv[0]- frv[-1])/2)+ '\n')
            f.write('Sheath Compression: '+ '{:10.2f}'.format(comp)+ '\n')
        
            f.write('\n')
            f.write('Duration with negative Bz (hrs)'+ '\n')
            f.write('Sheath:     '+ '{:6.1f}'.format(BzDurs[0])+ '\n')
            f.write('FR:         '+ '{:6.1f}'.format(BzDurs[1])+ '\n')
            f.close()
        
        outThing += '{:10.2f}'.format(np.mean(sB)) + ' '
        outThing += '{:10.2f}'.format(np.mean(sBz)) + ' '
        outThing += '{:10.2f}'.format(np.mean(sv)) + ' '
        outThing += '{:10.2f}'.format(np.mean(sn)) + ' '
        outThing += '{:10.2f}'.format(np.mean(sT)) + ' '
        outThing += '{:10.2f}'.format(np.mean(svxBz)) + ' '
        outThing += '{:10.2f}'.format(np.mean(sKp)) + ' '
        outThing += '{:10.2f}'.format(np.mean(frB)) + ' '
        outThing += '{:10.2f}'.format(np.mean(frBz)) + ' '
        outThing += '{:10.2f}'.format(np.mean(frv)) + ' '
        outThing += '{:10.2f}'.format(np.mean(frn)) + ' '
        outThing += '{:10.2f}'.format(np.mean(frT)) + ' '
        outThing += '{:10.2f}'.format(np.mean(frvxBz)) + ' '
        outThing += '{:10.2f}'.format(np.mean(frKp)) + ' '
        
        outThing += '{:10.2f}'.format(np.max(sB)) + ' '
        outThing += '{:10.2f}'.format(np.min(sBz)) + ' '
        outThing += '{:10.2f}'.format(np.max(sv)) + ' '
        outThing += '{:10.2f}'.format(np.max(sn)) + ' '
        outThing += '{:10.2f}'.format(np.max(sT)) + ' '
        outThing += '{:10.2f}'.format(np.min(svxBz)) + ' '
        outThing += '{:10.2f}'.format(np.max(sKp)) + ' '
        outThing += '{:10.2f}'.format(np.max(frB)) + ' '
        outThing += '{:10.2f}'.format(np.min(frBz)) + ' '
        outThing += '{:10.2f}'.format(np.max(frv)) + ' '
        outThing += '{:10.2f}'.format(np.max(frn)) + ' '
        outThing += '{:10.2f}'.format(np.max(frT)) + ' '
        outThing += '{:10.2f}'.format(np.min(frvxBz)) + ' '
        outThing += '{:10.2f}'.format(np.max(frKp)) + ' '
        
        outThing += '{:10.2f}'.format((frv[0]- frv[-1])/2) + ' '
        outThing += '{:10.2f}'.format(comp) + ' '
        outThing += '{:6.1f}'.format(BzDurs[0]) + ' '
        outThing += '{:6.1f}'.format(BzDurs[1]) + ' '
        
        return (outThing)
        
        

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
            if self.tShock:
                myf.plot([self.tShock, self.tShock], ylims, 'k--')
            if self.tFRfront:
                myf.plot([self.tFRfront, self.tFRfront], ylims, 'k--')
            if self.tFRend:
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
            
#h5File = '/Users/kaycd1/GAMERA/GAMERA_ACE_time_profiles/ACE.sc.h5'
h5File = '/Users/kaycd1/GAMERA/gamhelio_data_ACE_260304/ACE.sc-vel_202304211200R000_agong-' #900.h5'
vtags = ['900', '950', '1000', '1050', '1100']
for vtag in vtags:
    obs = ACEobsH5(h5File+vtag+'.h5')
    outThing = obs.getValues(fileName='results'+vtag+'.txt')
    print (outThing)
    obs.plotIt(figName='results'+vtag+'.png')
