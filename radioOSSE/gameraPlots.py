import numpy as np
import matplotlib.pyplot as plt
import geom 
from extractGAMERA import GAMERAres

def plotEcliptic(res, tidx):
    #stuff = res.h5_data[myStep][vals[j]][pidx[i],tidx[i],ridx[i]]
    stepName = 'Step#'+str(tidx)
    ecIdx = np.where(res.theta[0,:,0] == 90)[0]
    myShape = res.h5_data[stepName]['Vx'].shape # all vars same shape
    xCoords = res.X[:myShape[0],ecIdx[0],:myShape[2]]
    yCoords = res.Y[:myShape[0],ecIdx[0],:myShape[2]]
    zCoords = res.Z[:myShape[0],ecIdx[0],:myShape[2]]
    rvals   = np.sqrt(xCoords**2 + yCoords**2 + zCoords**2) 
    myShape = xCoords.shape
    theVal  = np.zeros([myShape[0], myShape[1]])
    vecVals = np.zeros([myShape[0], myShape[1],3])
    for i in range(myShape[0]):
        temp = res.h5_data[stepName]['D'][i,ecIdx,:]
        theVal[i,:]  = temp
        # vector option
        tempx = res.h5_data[stepName]['Bx'][i,ecIdx,:]
        vecVals[i,:,0] = tempx
        tempy = res.h5_data[stepName]['By'][i,ecIdx,:]
        vecVals[i,:,1] = tempy
        tempz = res.h5_data[stepName]['Bz'][i,ecIdx,:]
        vecVals[i,:,2] = tempz
    
    # add on edge
    fullX = np.zeros([myShape[0]+1, myShape[1]])
    fullX[:-1,:] = xCoords
    fullX[-1] = xCoords[0,:]
    fullY = np.zeros([myShape[0]+1, myShape[1]])
    fullY[:-1,:] = yCoords
    fullY[-1] = yCoords[0,:]
    fullV = np.zeros([myShape[0]+1, myShape[1]])
    fullV[:-1,:] = theVal
    fullV[-1] = theVal[0,:]
    fullR = np.zeros([myShape[0]+1, myShape[1]])
    fullR[:-1,:] = rvals
    fullR[-1] = rvals[0,:]
    fullVec = np.zeros([myShape[0]+1, myShape[1], 3])
    for i in range(3):
        fullVec[:-1,:,i] = vecVals[:,:,i]
        fullVec[-1,:,i] = vecVals[0,:,i]
    
    bTot = np.sqrt(fullVec[:,:,0]**2 + fullVec[:,:,1]**2 + fullVec[:,:,2]**2 )
    fig = plt.figure()
    levels = [0,5,10,15,20,25,30,35,40,45,50]
    #levels = 10
    #levels = np.linspace(0,30,10)
    #im = plt.contourf(fullX, fullY, np.abs(bTot)*fullR**2 /215**2, levels=levels, cmap='plasma')
    im = plt.contourf(fullX, fullY, fullV*fullR**2 /215**2, levels=levels, cmap='plasma')
    plt.axis('equal')
    plt.colorbar(im, label = 'n  (R/1 au)$^2$ (cm$^{-3}$)')
    #plt.colorbar(im, label = 'B  (R/1 au)$^2$ (nT)')
    plt.axis('off')
    
    # Add sats
    dtor = 3.14159/180.
    gridLon = [0,10,20, 30]
    for dLon in gridLon:
        myR = 215
        myX = np.cos(dLon*dtor) * myR
        myY = np.sin(dLon*dtor) * myR
        plt.scatter(myX, myY, c='c', s=20)
        plt.scatter(myX, -myY, c='c', s=20)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('gameraEclipticDensity_t40.png')

def plotMerid(res, tidx):
    #stuff = res.h5_data[myStep][vals[j]][pidx[i],tidx[i],ridx[i]]
    stepName = 'Step#'+str(tidx)
    merIdx = np.where(res.phi[:,10,0] == 0)[0][0]
    myShape = res.h5_data[stepName]['Vx'].shape # all vars same shape
    xCoords = res.X[merIdx,:myShape[1],:myShape[2]]
    yCoords = res.Y[merIdx,:myShape[1],:myShape[2]]
    zCoords = res.Z[merIdx,:myShape[1],:myShape[2]]
    rvals   = np.sqrt(xCoords**2 + yCoords**2 + zCoords**2) 
    myShape = xCoords.shape
    theVal  = np.zeros([myShape[0], myShape[1]])
    vecVals = np.zeros([myShape[0], myShape[1],3])
    for i in range(myShape[0]):
        temp = res.h5_data[stepName]['D'][merIdx,i,:]
        theVal[i,:]  = temp
        # vector option
        tempx = res.h5_data[stepName]['Bx'][merIdx,i,:]
        vecVals[i,:,0] = tempx
        tempy = res.h5_data[stepName]['By'][merIdx,i,:]
        vecVals[i,:,1] = tempy
        tempz = res.h5_data[stepName]['Bz'][merIdx,i,:]
        vecVals[i,:,2] = tempz
    
    # add on edge
    '''fullX = np.zeros([myShape[0]+1, myShape[1]])
    fullX[:-1,:] = xCoords
    fullX[-1] = xCoords[0,:]
    fullY = np.zeros([myShape[0]+1, myShape[1]])
    fullY[:-1,:] = yCoords
    fullY[-1] = yCoords[0,:]
    fullV = np.zeros([myShape[0]+1, myShape[1]])
    fullV[:-1,:] = theVal
    fullV[-1] = theVal[0,:]
    fullR = np.zeros([myShape[0]+1, myShape[1]])
    fullR[:-1,:] = rvals
    fullR[-1] = rvals[0,:]
    fullVec = np.zeros([myShape[0]+1, myShape[1], 3])
    for i in range(3):
        fullVec[:-1,:,i] = vecVals[:,:,i]
        fullVec[-1,:,i] = vecVals[0,:,i]'''
    
    bTot = np.sqrt(vecVals[:,:,0]**2 + vecVals[:,:,1]**2 + vecVals[:,:,2]**2 )
    fig = plt.figure()
    #levels = [0,5,10,15,20,25,30,35,40,45,50]
    #levels = 10
    levels = np.linspace(0,30,10)
    im = plt.contourf(xCoords, zCoords, np.abs(bTot)*rvals**2 /215**2, levels=levels, cmap='plasma')
    #im = plt.contourf(xCoords, zCoords, theVal*rvals**2 /215**2, levels=levels, cmap='plasma')
    plt.axis('equal')
    #plt.colorbar(im, label = 'n  (R/1 au)$^2$ (cm$^{-3}$)')
    plt.colorbar(im, label = 'B  (R/1 au)$^2$ (nT)')
    plt.axis('off')
    
    # Add sats
    dtor = 3.14159/180.
    gridLon = [0,10]
    for dLon in gridLon:
        myR = 215
        myX = np.cos(dLon*dtor) * myR
        myY = np.sin(dLon*dtor) * myR
        plt.scatter(myX, myY, c='c', s=20)
        plt.scatter(myX, -myY, c='c', s=20)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('gameraMeridB_t40.png')

    
filePath = '/Users/kaycd1/GAMERA/fromDevoj/256serial/wsaCR2261cme05092022.gam.h5'
gamRes = GAMERAres(filePath)
#plotEcliptic(gamRes, 40)
plotMerid(gamRes, 40)