import numpy as np 
import matplotlib.pyplot as plt
import datetime 
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import h5py
from astropy.time import Time
import cdflib
import pandas as pd

# |-----------------------| 
# |---- Profile Class ----| 
# |-----------------------| 
class ISprofile:
    # |------------------------------| 
    # |---- Structure Definition ----|
    # |------------------------------| 
    "Container for an in situ profile from either GAMERA or real obs"
    def __init__(self):
        # |---- Basic parameters (profiles) ----|
        self.MJD    = None # Modified Julian Date
        self.tMHD   = None # MHD time (hrs?)
        self.dts    = None # date time objects
        self.Bx     = None # GSE Bx (nT)
        self.By     = None # GSE By (nT)
        self.Bz     = None # GSE Bz (nT)
        self.B      = None # Total B (nT)
        self.vx     = None # GSE vx (km/s)
        self.vy     = None # GSE vx (km/s)
        self.vz     = None # GSE vx (km/s)
        self.v      = None # Total v (km/s)
        self.Ex     = None # GSE Ex (?)
        self.Ey     = None # GSE Ey (?)
        self.Ez     = None # GSE Ez (?)
        self.E      = None # Total E (?)
        self.Pt     = None # Thermal pressure
        self.Temp   = None # Temperature (K)
        
        # |---- Derived Profiles ----|
        self.Pb     = None # Magnetic pressure
        self.Pram   = None # Ram pressure
        self.beta   = None # Plasma beta
        self.vxBz   = None # related to cross cap potential (?)
        self.Kp     = None # Kp index
                    
        # |---- First derivatives ----|
        self.ddt_dts = None # times for derivs (one shorter than dts)
        self.ddt_B   = None
        self.ddt_v   = None
        self.ddt_T   = None
        self.ddt_n   = None
        
        # |---- Region things ----|   
        self.tShock    = None
        self.tFRfront  = None
        self.tFRend    = None
        self.idx1      = None
        self.idx2      = None
        self.idx3      = None
        self.DoYs      = [None, None, None] # fractional day of years
        self.hasSheath = False
        self.hasFR     = False
        
        # |---- Derived Values ----|
        self.comp       = None # Compression r
        self.sheathDur  = None # hr
        self.FRdur      = None # hr
        self.FRvexp     = None # km/s
        self.shNegBz    = None # hr
        self.frNegBz    = None # hr
        
        
        # |---- Mean/Max Values ----|
        self.mmBtot     = [[None, None], [None, None]] # [[meanSheath, meanFR], [maxSh, maxFR]]
        self.mmBz       = [[None, None], [None, None]] # min not max
        self.mmv        = [[None, None], [None, None]]
        self.mmn        = [[None, None], [None, None]]
        self.mmT        = [[None, None], [None, None]]
        self.mmvxBz     = [[None, None], [None, None]] # min not max
        self.mmKp       = [[None, None], [None, None]]

        # |---- Ensemble values ----|
        # Used to config Gamera simulation
        self.ensDict = {}
        # make a dictionary so we can add values by var name
        
        # |---- Flag for obs/sim ---|
        self.isSim = None
        
        # |---- Color for plots ---|
        self.color = 'k' # default to black instead of always checking
        
        
    # |--------------------------------| 
    # |---- Fill from H5 file type ----|
    # |--------------------------------| 
    def fillFromH5(self, fileIn):
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
            # This one, important for swx preds
            self.vxBz = np.abs(self.vx) * self.Bz
            # Kp
            Bt = np.sqrt(self.By**2 + self.Bz**2)
            thetaC = np.abs(np.atan2(self.By, self.Bz))
            dphidt = np.power(self.v, 4/3) * np.power(Bt, 2/3) * np.power(np.sin(thetaC/2), 8/3)
            self.Kp = 9.5 - np.exp(2.17676 - 5.2001e-5 * dphidt)
            
            
            # modified Julian date
            self.MJD = f['MJDs'][:]
            # Date time array
            self.dts = mjd2dts(self.MJD)
            
            # Calculate the derivatives
            self.calc_derivs()
                        
    # |---------------------------------| 
    # |---- Fill from csv file type ----|
    # |---------------------------------|
    def fillFromCDF(self, fileIn, tag):
        if tag.lower() not in ['obs', 'sim']:
            print('fillFromCSV needs either obs or sim tag to specify data set')
            return
            
        doObs = False
        if tag.lower() == 'obs':
            doObs =True
        
        cdf_file = cdflib.CDF(fileIn)
        # CDFInfo(CDF=PosixPath('/Users/kaycd1/GAMERA/gamhelio_data_ACE_260304/ACE.comp-vel_202304211200R000_agong-900.cdf'), Version='3.8.1', Encoding=6, Majority='Row_major', rVariables=[], zVariables=['
        # 'metavar0', 'Epoch_bin', 'XYZ_GSE', 'Vp', 'BGSEc', 'Np', 'Tpr', 'Ephemeris_time', 'X', 'Y', 'Z', 'Speed', 'Br', 'Density', 'Temperature', 'GAMHELIO_Speed', 'GAMHELIO_Br', 'GAMHELIO_Density', 'GAMHELIO_Temperature', 'GAMHELIO_inDom']
        
        if doObs:
            self.Bx = cdf_file.varget("BGSEc")[:,0] 
            self.By = cdf_file.varget("BGSEc")[:,1] 
            self.Bz = cdf_file.varget("BGSEc")[:,2] 
            self.B  = np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)
            self.n = cdf_file.varget("Np") # is number density not mass
            self.v = cdf_file.varget("Vp") 
            self.Temp = cdf_file.varget("Temperature") 
            # Need to double check on temp factors of 2!!!!
            self.Pt = self.Temp * self.n * 1.38e-8 #/ 2 *2
            
            # Calc mag pressure
            self.Pb = (self.B * 1e-5)**2 / 2
            # Calc plasma beta
            self.Beta = self.Pt *1e-8 / self.Pb 
            # Calc ram pressure
            self.Pram = 0.5*(self.v * 1e5)**2 * self.n
            
            # Kp
            Bt = np.sqrt(self.By**2 + self.Bz**2)
            thetaC = np.abs(np.atan2(self.By, self.Bz))
            dphidt = np.power(self.v, 4/3) * np.power(Bt, 2/3) * np.power(np.sin(thetaC/2), 8/3)
            self.Kp = 9.5 - np.exp(2.17676 - 5.2001e-5 * dphidt)
            
        else:
            self.Bx = cdf_file.varget("GAMHELIO_Br")
            self.v = cdf_file.varget("GAMHELIO_Speed") 
            self.n = cdf_file.varget("GAMHELIO_Density") 
            self.Temp = cdf_file.varget("GAMHELIO_Temperature") 
            self.B = self.Bx
            
            # Calc thermal pressure
            self.Pt = self.Temp * self.n * 1.38e-8 #/ 2 *2
            # Calc mag pressure
            self.Pb = (self.B * 1e-5)**2 / 2
            # Calc plasma beta
            self.Beta = self.Pt *1e-8 / self.Pb
            # Calc ram pressure
            self.Pram = 0.5*(self.v * 1e5)**2 * self.n
            
            # Cannot do Kp without vector B
                       
        
        # Convert epoch time to datetimes    
        raw_time = cdf_file.varget('Epoch_bin')
        tStrs = cdflib.cdfepoch.to_datetime(raw_time)
        dts = []
        for tstr in tStrs:
            dt_timestamp = pd.to_datetime(tstr)
            dts.append(dt_timestamp)
        self.dts = np.array(dts)
        
        # Calculate the derivatives
        self.calc_derivs()
            

            
    # |--------------------------| 
    # |---- Calc derivatives ----|
    # |--------------------------|
    def calc_derivs(self):
        if type(self.dts) == type(None):
            print ('Cannot calculate derivatives, missing dt entry')
        else:
            ddts = []
            # set the deriv time at the midpoint
            for i in range(len(self.dts)-1):     
                delta = (self.dts[i+1] - self.dts[i]).total_seconds()
                ddts.append(self.dts[i] + datetime.timedelta(seconds=0.5*delta))
            ddts = np.array(ddts)
            self.ddt_dts = ddts
            
            # Calc the derivs
            self.ddt_B   = self.B[2:] - self.B[:-2]
            self.ddt_v   = self.v[2:] - self.v[:-2]
            self.ddt_n   = self.n[2:] - self.n[:-2]
            self.ddt_T   = self.Temp[2:] - self.Temp[:-2]
                            

    # |------------------------------------| 
    # |---- Determine sheath/FR bounds ----|
    # |------------------------------------|
    def getBounds(self):
        # Make sure it has each deriv - necessary?
        hasB, hasv, hasn, hasT, hasCombo = True, True, True, True, False
        if type(self.ddt_B) == type(None): hasB = False
        if type(self.ddt_v) == type(None): hasv = False
        if type(self.ddt_n) == type(None): hasn = False
        if type(self.ddt_T) == type(None): hasT = False
        if hasB and hasv and hasn and hasT: hasCombo =True
        
        # Get normalized derivatives
        ndB, ndv, ndn, ndT, ndCombo = None, None, None, None, None
        if hasB: 
            dB_isReal = np.isfinite(self.ddt_B)
            ndB = self.ddt_B/np.max(self.ddt_B[dB_isReal])
        if hasv: 
            dv_isReal = np.isfinite(self.ddt_v)
            ndv = self.ddt_v/np.max(self.ddt_v[dv_isReal])
        if hasn: 
            dn_isReal = np.isfinite(self.ddt_n)
            ndn = self.ddt_n/np.max(self.ddt_n[dn_isReal])
        if hasT: 
            dT_isReal = np.isfinite(self.ddt_T)
            ndT = self.ddt_T/np.max(self.ddt_T[dT_isReal])
        if hasCombo:
            combo = ndB * ndv * ndn * ndT
            combo_isReal = np.isfinite(combo)
            ndCombo = np.abs(combo / np.max(combo[combo_isReal])) * np.sign(ndn)
        # Set up for DoY calc
        myYR = datetime.datetime(self.dts[0].year,1,1,0,0)    
        
        # |---------------------| 
        # |---- Shock Front ----| 
        # |---------------------|            
        if hasCombo:
            idx = np.where(ndCombo == np.max(ndCombo[combo_isReal]))[0]
            self.idx1 = idx[0] + 1 # set to latter of the times used in deriv
            self.tShock = self.ddt_dts[idx[0] -1] # this is midpoint time
        elif hasT:
            print('Using T derive for shock front time')
            idx = np.where(ndT == np.max(ndT))[0]
            self.idx1 = idx[0] + 1 # set to latter of the times used in deriv
            self.tShock = self.ddt_dts[idx[0] -1]
        else:
            print('Need alternate method for shock time')
        # Decimal DoY
        if self.tShock:
            self.DoYs[0] = (self.tShock - myYR).total_seconds()/3600./24

        # |------------------| 
        # |---- FR Front ----| 
        # |------------------| 
        # Check if we can use beta to find FR
        # MHD often all < 1 so need other method
        useBeta = True
        if self.tShock:
            if self.Beta[self.idx1] < 1: useBeta = False
        
        if useBeta:
            # set at first time beta goes < 1 post shock
            idxB = np.where(self.Beta < 1)[0]
            idxFR0 = np.min(idxB[np.where(idxB > (self.idx1))][0])
            self.idx2 = idxFR0
            self.tFRfront = self.dts[idxFR0]
        else:
            # Try using deriv of beta 
            ddBeta = self.Beta[1:] - self.Beta[:-1]
            subddBeta = ddBeta[self.idx1:]
            goodIdx = np.isfinite(subddBeta)
            minBeta  = np.min(subddBeta[goodIdx])
            minIdx = np.where(subddBeta == minBeta)[0][0]
            if minIdx > self.idx1:
                self.idx2 = minIdx -1
                self.tFRfront = self.dts[self.idx2]
            else:
                # use deriv of density as backup method
                subddtn = self.ddt_n[self.idx1-1:]
                idxN = np.where(subddtn == np.min(subddtn[dn_isReal[self.idx1-1:]]))[0][0]
                self.tFRfront = self.ddt_dts[self.idx1-1:][idxN]
                self.idx2 = np.min(np.where(self.dts >= self.tFRfront)[0])
        # Decimal DoY
        if self.tFRfront:
            self.DoYs[1] = (self.tFRfront - myYR).total_seconds()/3600./24
            
        # |----------------| 
        # |---- FR End ----| 
        # |----------------|
        # Ideally set to first time beta > 1 after FR front
        # but if doesnt do that then set to time of max beta
        subBeta = self.Beta[self.idx2+2:] # add a couple points to rm front 
        subtimes = self.dts[self.idx2+2:]
        beta_isReal = np.isfinite(subBeta)
        maxBeta = np.max(subBeta[beta_isReal])
        if maxBeta > 1:
            idxE = np.where(self.Beta > 1)[0]
            idxFRe = np.min(idxE[np.where(idxE > idxFR0)][0])
            self.idx3 = idxFRe
            self.tFRend = self.dts[idxFRe]
        else:
            # Check if we have enough beta to use
            if len(subBeta[beta_isReal]) > len(subBeta):
                idxE = np.where(subBeta == maxBeta)[0]
                self.tFRend = subtimes[idxE[0]]
                self.idx3 = np.where(self.dts == self.tFRend)[0][0]
            else:
                # use B getting close to upstream value?
                if self.idx1:
                    upIdx1 = self.idx1
                else:
                    upIdx1 = self.idx2
                upIdx0 = np.max([upIdx1-12, 0])
                upB = np.mean(self.B[upIdx0:upIdx1+1])
                below = np.where(self.B[upIdx1:] < upB)[0]
                if len(below) > 1:
                    self.idx3 = below[0] + upIdx1
                    self.tFRend = self.dts[self.idx3]
                else:
                    lowish = np.where(self.B[upIdx1:] < 1.25*upB)[0]
                    if len(lowish) > 1:
                        self.idx3 = below[0] + upIdx1
                        self.tFRend = self.dts[self.idx3]
                    else:
                        print('Cannot find FR back end')
        # Decimal DoY
        if self.tFRend:
            self.DoYs[2] = (self.tFRend - myYR).total_seconds()/3600./24
            
        # |---------------------| 
        # |---- Add regions ----| 
        # |---------------------|
        if self.tShock and self.tFRfront:
            self.hasSheath = True
            self.sheathDur = (self.tFRfront - self.tShock ).total_seconds() / 3600.
        if self.tFRend and self.tFRfront:
            self.hasFR = True    
            self.FRDur = (self.tFRend - self.tFRfront).total_seconds() / 3600.
            
            
    # |----------------------------------------| 
    # |---- Determine sheath/FR properties ----|
    # |----------------------------------------|
    def getRegionProperties(self):
        # |---- Sheath Calculation ----|
        if self.hasSheath:
            sIdxs = np.array([self.idx1 + i for i in range(self.idx2 - self.idx1)])
            sB, sn, sv, sT,  = self.B[sIdxs], self.n[sIdxs], self.v[sIdxs], np.log10(self.Temp[sIdxs])
            idx = np.isfinite(sB)[0]
            self.mmBtot[0] = [np.mean(sB[idx]), np.max(sB[idx])]  
            idx = np.isfinite(sv)[0]   
            self.mmv[0] = [np.mean(sv[idx]), np.max(sv[idx])]
            idx = np.isfinite(sn)[0]
            self.mmn[0] = [np.mean(sn[np.isfinite(sn)]), np.max(sn[np.isfinite(sn)])]
            idx = np.isfinite(sT)[0]
            self.mmT[0] = [np.mean(sT[idx]), np.max(sT[idx])]
            # Might not have these params
            if type(self.Bz) != type(None):
                sBz = self.Bz[sIdxs]
                idx = np.isfinite(sBz)[0]
                self.mmBz[0] = [np.mean(sBz[idx]), np.min(sBz[idx])]
            if type(self.vxBz) != type(None):
                svxBz = self.vxBz[sIdxs]
                idx = np.isfinite(svxBz)[0]
                self.mmvxBz[0] = [np.mean(svxBz[idx]), np.min(svxBz[idx])]
            if type(self.Kp) != type(None):
                sKp = self.Kp[sIdxs]
                idx = np.isfinite(sKp)[0]
                self.mmKp[0] = [np.mean(sKp[idx]), np.max(sKp[idx])]
            
                
            # Get longest continuous negative Bz duration
            if type(self.Bz) != type(None):
                sBzIdxs = np.where(sBz < 0)[0]  
                # Continuous
                if len(sBzIdxs) > 0:
                    if len(sBzIdxs) == sBzIdxs[-1]-sBzIdxs[0]+1:
                        self.shNegBz = (self.dts[sIdxs[sBzIdxs[-1]]] - self.dts[sIdxs[sBzIdxs[0]]]).total_seconds() / 3600.
                    # Otherwise need to check the pieces
                    else:
                        theseDurs = []
                        prevVal = sBzIdxs[0]
                        startVal = sBzIdxs[0]
                        for i in range(len(sBzIdxs)-1):
                            nextVal = sBzIdxs[i+1]
                            if nextVal - prevVal != 1:
                                theseDurs.append(prevVal-startVal+1)
                                startVal = nextVal
                            prevVal = nextVal
                        theseDurs.append(prevVal-startVal+1)
                        self.shNegBz = np.max(theseDurs)
                else:
                    self.shNegBz = 0
                
            # |---- Calculate compression ----|
            if self.idx1 >= 18:
                upidxs = [self.idx1 - 17 + i for i in range(12)]
                upn  =  np.mean(self.n[upidxs])
                self.comp = np.mean(sn) / upn    
        
        # |---- FR Calculation ----|    
        if self.hasFR:
            FRIdxs = [self.idx2 + i for i in range(self.idx3 - self.idx2 +1)]
            frB, frn, frv, frT = self.B[FRIdxs], self.n[FRIdxs], self.v[FRIdxs], np.log10(self.Temp[FRIdxs])
            idx = np.isfinite(frB)[0]
            self.mmBtot[1] = [np.mean(frB[idx]), np.max(frB[idx])]
            idx = np.isfinite(frv)[0]
            self.mmv[1] = [np.mean(frv[idx]), np.max(frv[idx])]
            # dunno why wasn't working in same format as others but this does
            self.mmn[1] = [np.mean(frn[np.isfinite(frn)]), np.max(frn[np.isfinite(frn)])]

            idx = np.isfinite(frT)[0]
            self.mmT[1] = [np.mean(frT[idx]), np.max(frT[idx])]
            
            # Might not have these params
            if type(self.Bz) != type(None):
                frBz = self.Bz[FRIdxs]
                idx = np.isfinite(frBz)[0]
                self.mmBz[1] = [np.mean(frBz[idx]), np.min(frBz[idx])]
            if type(self.vxBz) != type(None):
                frvxBz = self.vxBz[FRIdxs]
                idx = np.isfinite(frvxBz)[0]
                self.mmvxBz[1] = [np.mean(frvxBz[idx]), np.min(frvxBz[idx])]
            if type(self.Kp) != type(None):
                frKp = self.Kp[FRIdxs]
                idx = np.isfinite(frKp)[0]
                self.mmKp[1] = [np.mean(frKp[idx]), np.max(frKp[idx])]
            
            # Get longest continuous negative Bz duration
            if type(self.Bz) != type(None):
                frBzIdxs = np.where(frBz < 0)[0]  
                # Continuous
                if len(frBzIdxs) > 0:
                    if len(frBzIdxs) == frBzIdxs[-1]-frBzIdxs[0]+1:
                        self.frNegBz = (self.dts[FRIdxs[frBzIdxs[-1]]] - self.dts[FRIdxs[frBzIdxs[0]]]).total_seconds() / 3600.
                    # Otherwise need to check the pieces
                    else:
                        theseDurs = []
                        prevVal = frBzIdxs[0]
                        startVal = frBzIdxs[0]
                        for i in range(len(frBzIdxs)-1):
                            nextVal = frBzIdxs[i+1]
                            if nextVal - prevVal != 1:
                                theseDurs.append(prevVal-startVal+1)
                                startVal = nextVal
                            prevVal = nextVal
                        theseDurs.append(prevVal-startVal+1)
                        self.frNegBz = np.max(theseDurs)
                        
                else:
                    self.frNegBz = 0

    # |-----------------------| 
    # |---- Print Results ----|
    # |-----------------------|
    # TBD

# |-------------------------| 
# |---- MJD to Datetime ----| 
# |-------------------------| 
def mjd2dts(inArr):
    # inArr should be an array of mjds
    # returns an array of datetimes
    dtArr = []
    for i in range(len(inArr)):
        myDT = Time(inArr[i], format='mjd').to_datetime()
        dtArr.append(myDT)
    outArr = np.array(dtArr)
    return outArr     
            
            
# |-----------------------------| 
# |---- (Multi)profile Plot ----| 
# |-----------------------------| 
def plotProfile(profsIn, figName='temp.png'):
    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    gs  = fig.add_gridspec(8, 1)
    f1  = fig.add_subplot(gs[0,:])                  
    f1a = fig.add_subplot(gs[1,:], sharex=f1)    
    f1b = fig.add_subplot(gs[2,:], sharex=f1)    
    f1c = fig.add_subplot(gs[3,:], sharex=f1)    
    f2  = fig.add_subplot(gs[4,:], sharex=f1)                  
    f3  = fig.add_subplot(gs[5,:], sharex=f1)                  
    f4  = fig.add_subplot(gs[6,:], sharex=f1)     
    f5  = fig.add_subplot(gs[7,:], sharex=f1)    
    
    # Repackage a single profile 
    if type(profsIn) == type(ISprofile()):
        profsIn = [profsIn]
    
    # Main profiles
    for aProf in profsIn:
        f1.plot(aProf.dts, aProf.B, c=aProf.color)             
        f1a.plot(aProf.dts, aProf.By, c=aProf.color)             
        f1b.plot(aProf.dts, aProf.Bz, c=aProf.color)             
        f1c.plot(aProf.dts, aProf.B, c=aProf.color)  
        f2.plot(aProf.dts, aProf.v, c=aProf.color)  
    
        f3.plot(aProf.dts, aProf.n, c=aProf.color)
        f4.plot(aProf.dts, aProf.Temp, c=aProf.color)
        f5.plot(aProf.dts, aProf.Beta,c=aProf.color)
        f5.plot(aProf.dts, aProf.Beta*0+1, 'k--') 
        
    f5.set_yscale('log')      

    allfs = [f1, f1a, f1b, f1c, f2, f3, f4, f5]
    for i in range(len(allfs)):
        for aProf in profsIn:
            myf   = allfs[i]
            ylims = myf.get_ylim()
            if aProf.tShock:
                myf.plot([aProf.tShock, aProf.tShock], ylims, ':', color=aProf.color)
            if aProf.tFRfront:
                myf.plot([aProf.tFRfront, aProf.tFRfront], ylims, '--', color=aProf.color)
            if aProf.tFRend:
                myf.plot([aProf.tFRend, aProf.tFRend], ylims, '--', color=aProf.color)
            myf.set_ylim(ylims)
    
    
    f1.set_ylabel('B [nT]')  
    f1a.set_ylabel('B$_x$ [nT]')  
    f1b.set_ylabel('B$_y$ [nT]')  
    f1c.set_ylabel('B$_z$ [nT]')  
    f2.set_ylabel('v [km/s]')  
    f3.set_ylabel('n [cm$^{-3}$]')  
    f4.set_ylabel('T [k]')  
    f5.set_ylabel('$\\beta$') 
    
    # Get optimal plot range
    minT = aProf.tFRfront
    maxT = aProf.tFRend
    for aProf in profsIn:
        if aProf.hasSheath:
            if aProf.tShock < minT: minT = aProf.tShock
        else:
            if aProf.tFRfront < minT: minT = aProf.tFRfront
        if aProf.tFRend > maxT: maxT = aProf.tFRend
 
    pltStart = minT - datetime.timedelta(hours=12)    
    pltEnd   = maxT + datetime.timedelta(hours=12)    
    f1.set_xlim([pltStart,pltEnd])
    my_format = mdates.DateFormatter('%d %H:%M')
    f5.xaxis.set_major_formatter(my_format)
    f5.tick_params(axis='x', labelrotation=25)
    for aF in allfs[:-1]:
        plt.setp(aF.get_xticklabels(), visible=False)
    
    plt.savefig(figName)
    
# |-------------------------| 
# |---- Comparison Plot ----| 
# |-------------------------| 
def compPlot(profsIn, ensVar, figName='tempComp.png'):
    # Check if we have an obs profile for comparison
    haveObs = False
    obsProf = None
    simProf = []
    ensVal  = []
    compVals = [] # AT, dur, B, n, T, v, vexp
    obsVals  = [None]
    compCols = [] 
    # Sort things, pull out obs profile
    myYr = profsIn[0].tFRfront.year
    day0 = datetime.datetime(myYr,1,1)
    
    ylabs = ['AT (days)', 'Dur (hr)', 'B (nT)', 'v (km/s)', 'n (cm$^{-3}$)',  'log T (K)']

    for aProf in profsIn:
        if aProf.isSim: 
            simProf.append(aProf)
            ensVal.append(aProf.ensDict[ensVar])
            compCols.append(aProf.color)
            
            ATs = [None, None]
            durs = [None, None]
            
            if aProf.tShock:
                ATs[0] = (aProf.tShock - day0).total_seconds() / 3600. / 24.
                durs[0] = aProf.sheathDur
                comp = aProf.comp
            if aProf.tFRfront:
                ATs[1] = (aProf.tFRfront - day0).total_seconds() / 3600. / 24.
                durs[1] = aProf.FRDur            
            compVals.append([ATs, durs, aProf.mmBtot[0], aProf.mmv[0], aProf.mmn[0], aProf.mmT[0]])
            
        else:
            if haveObs:
                print ('Too many observation profiles passed to compPlot')
                return
            else:
                obsProf = aProf
                haveObs = True
                
                ATs = [None, None]
                durs = [None, None]
            
                if aProf.tShock:
                    ATs[0] = (aProf.tShock - day0).total_seconds() / 3600. / 24.
                    durs[0] = aProf.sheathDur
                    comp = aProf.comp
                if aProf.tFRfront:
                    ATs[1] = (aProf.tFRfront - day0).total_seconds() / 3600. / 24.
                    durs[1] = aProf.FRDur 
                obsVals = [ATs, durs, aProf.mmBtot[0], aProf.mmv[0], aProf.mmn[0], aProf.mmT[0], comp]

    # Set up the figure/axis
    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    gs  = fig.add_gridspec(3, 2)
    f1a  = fig.add_subplot(gs[0,0])                  
    f1b = fig.add_subplot(gs[0,1], sharex=f1a)
    f2a  = fig.add_subplot(gs[1,0], sharex=f1a)                  
    f2b = fig.add_subplot(gs[1,1], sharex=f1a)
    f3a  = fig.add_subplot(gs[2,0], sharex=f1a)                  
    f3b = fig.add_subplot(gs[2,1], sharex=f1a)
    allfs = [f1a, f1b, f2a, f2b, f3a, f3b]
    
    for i in range(len(profsIn)-1):
        myCompVals = compVals[i]
        for j in range(len(allfs)):
            if haveObs:
                obsShVal = obsVals[j][0]
                obsFRVal = obsVals[j][1]
            else:
                obsShVal, obsFRVal = 0, 0
            if type(myCompVals[j][0]) != type(None):
                allfs[j].plot(ensVal[i], myCompVals[j][0]-obsShVal, '+', color=compCols[i])
            if type(myCompVals[j][1]) != type(None):
                allfs[j].plot(ensVal[i], myCompVals[j][1]-obsFRVal, 'o', color=compCols[i])
        
    # Labels
    for j in range(len(allfs)):
        allfs[j].set_xlabel(ensVar)
        if haveObs:
            allfs[j].set_ylabel('$\\Delta$'+ylabs[j])
        else:
            allfs[j].set_ylabel(ylabs[j])
    plt.savefig(figName)
            
        
            


vtags = ['900', '950', '1000', '1050', '1100']
colors = ['#FF2056', '#E12AFB', '#AD46FF', '#615FFF', '#432DD7']
allProfs = []
counter = 0
cdfFile = '/Users/kaycd1/GAMERA/gamhelio_data_ACE_260304/ACE.comp-vel_202304211200R000_agong-900.cdf'
prof = ISprofile()
prof.fillFromCDF(cdfFile, 'obs')
prof.isSim = False
prof.getBounds()
prof.getRegionProperties()
allProfs.append(prof)
#plotProfile(allProfs)

for vtag in vtags:
    counter += 1
    prof = ISprofile()
    h5File = '/Users/kaycd1/GAMERA/gamhelio_data_ACE_260304/ACE.sc-vel_202304211200R000_agong-'+vtag+'.h5'
    prof.fillFromH5(h5File)
    prof.isSim = True
    prof.ensDict['v [km/s]'] = int(vtag)
    prof.color = colors[-counter]
    prof.getBounds()
    prof.getRegionProperties()
    allProfs.append(prof)
plotProfile(allProfs)
compPlot(allProfs, 'v [km/s]')
    