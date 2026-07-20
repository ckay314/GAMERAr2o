import numpy as np
import matplotlib.pyplot as plt
import geom 
from extractGAMERA import GAMERAres
from itertools import combinations
from matplotlib import cm
from astropy.time import Time

filePath = '/Users/kaycd1/GAMERA/fromDevoj/256serial/wsaCR2261cme05092022.gam.h5'
res = GAMERAres(filePath)

Re2Rs = 0.009164023 # 1 rE to rSun
L1pt = [0, 90, 215.03*0.99] 
# Gamera coords in (lon, colat, r) = (phi, theta, r) = (deg, deg, rs)
L1xyz = geom.SPH2CART(L1pt)


# |---- Set up satellites ---|
sat1xyz = np.copy(L1xyz)
sat1xyz[1] += 40 * Re2Rs
sat1 = geom.CART2SPH(sat1xyz)

sat2xyz = np.copy(L1xyz)
sat2xyz[1] += 100 * Re2Rs
sat2 = geom.CART2SPH(sat2xyz)

allSats = [L1pt, sat2]

# |--- Get baselines ---|
npts = 30
pairs = list(combinations(range(len(allSats)), 2))
baselines = []
for aPair in pairs:
    satA = allSats[aPair[0]]
    satB = allSats[aPair[1]]
    # Get cartesian
    satAxyz = geom.SPH2CART(satA)
    satBxyz = geom.SPH2CART(satB)
    
    blX = np.linspace(satAxyz[0], satBxyz[0], npts)
    blY = np.linspace(satAxyz[1], satBxyz[1], npts)
    blZ = np.linspace(satAxyz[2], satBxyz[2], npts)
    
    bl = np.transpose(np.array(geom.CART2SPH([blX, blY, blZ]))) # [[pt1], [pt2], ...]
    baselines.append(bl)
    
#|--- Extract along baseline ---\
allBx = []
dt=40
nt = 10
for aBase in baselines:
    for t in range(nt):
        print (t+dt)
        myNums = res.getInterpVal(aBase, ['D'], timestep=dt+t)
        print (myNums)
        allBx.append(myNums[0])

fig, ax = plt.subplots(layout="constrained")

for t in range(nt)[::-1]:
    mjd = Time(res.mjd_times[t+42], format='mjd', scale='utc').to_datetime().strftime("%Y-%m-%d %H:%M")
    ax.plot(np.linspace(0,1,npts), allBx[t], c=cm.hot(t/nt), label=mjd)
ax.set_xlabel('Fractional baseline')
ax.set_ylabel("Number density (cm$^{-3}$)")
plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
#qfig.legend(loc="outside right upper")
plt.tight_layout()
#plt.show()
plt.savefig('baselineTest.png')
