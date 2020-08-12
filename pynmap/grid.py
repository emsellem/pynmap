# -*- coding: utf-8 -*-

"""This module allows to compute histograms in an efficient way
"""

# Importing general modules
import numpy as np

# importing local modules
from .rotation import xy_to_polar
from .losvd import LOSVD

__version__ = '0.0.1 (4 July 2019)'
# 04/06/2019 - Version 0.0.1: - Creation, gathering from other modules

def point_2vprofiles(x, y, z, Vx, Vy, Vz, nbins=100, dz=np.array([-0.2, 0.2]),
                     Rmin=1.e-2, Rmax=None, Vmax=None, plot=False,
                     saveplot=False, **kwargs):
    """Compute the dispersions in the three directions
    selecting paticles in the Equatorial plane (or close to it)

    Parameters
    ----------
    nbins : number of bins (default = 20)
    dz : vertical range over which to select (default = 200 pc)
    Rmax : maximum radius for this profile (default = None)

    Returns
    -------
    Rbin: array
        Bins of radii
    v: 3xnbins array
        Velocities in cylindrical coordinates (R,T,Z)
    s: 3xnbins array
        Dispersion in cylindrical coordinates (R,T,Z)
    """
    s = np.zeros((3, nbins), dtype=np.float32)
    v = np.zeros_like(s)

    # Selection in Z
    selectZ = np.where((dz[0] < z) & (z < dz[1]))

    x = x[selectZ]
    y = y[selectZ]
    R, theta = xy_to_polar(x, y)

    VR = Vx[selectZ] * np.cos(theta) + Vy[selectZ] * np.sin(theta)
    VT = -Vx[selectZ] * np.sin(theta) + Vy[selectZ] * np.cos(theta)
    Vz = Vz[selectZ]

    if Rmax == None : Rmax = R.max()
    Rbin = np.logspace(np.log10(Rmin),np.log10(Rmax),nbins)

    Rdigit = np.digitize(R, Rbin)
    for i in range(nbins) :
        Rdigit_temp = np.where(Rdigit == i)[0]
        s[0,i] = VR[Rdigit_temp].std()
        s[1,i] = VT[Rdigit_temp].std()
        s[2,i] = Vz[Rdigit_temp].std()
        v[0,i] = VR[Rdigit_temp].mean()
        v[1,i] = VT[Rdigit_temp].mean()
        v[2,i] = Vz[Rdigit_temp].mean()

    if plot:
        plot_sigma(Rbin, v, s, Rmin=Rmin, Rmax=Rmax, Vmax=Vmax, 
                   save=saveplot, **kwargs)

    return Rbin, v, s

def points_2vmaps(x, y, v, weights=None, limXY=[-1,1,-1,1], nXY=11, mask=None):
    """
    Calculate the first 2 velocity moments

    Parameters
    ----------
    x : array
        x coordinate
    y : array 
        y coordinate
    v : array
        Value of velocity to project 
    w : array
        Weight (e.g., mass) if None weights are set to 1.
    limXY : Tuple or Array
          (xmin, xmax, ymin, ymax)
    nXY : int or tupl of int
          Number of bins in X and Y. If nXY is a single integer
          nX and nY will be set both to nXY
          Otherwise, nXY can be a set of 2 integers (tuple/array)
          which will then translate into nX=nXY[0] and nY=nXY[1]

    Returns
    -------
    X,Y : arrays
        projected x's and y's
    M : array
        Projected Weights
    V : array
        Projected velocity
    S : array
        Projected dispersion
    """
    if np.size(nXY) == 1 :
        nXY = np.zeros(2, dtype=np.int) + nXY
    elif np.size(nXY) != 2 :
        print("ERROR: dimension of n should be 1 or 2")
        return 0,0,0,0,0


    if weights is None : weights = np.ones_like(x)
    if mask is None:
        mask = np.zeros_like(weights, dtype=bool)

    nogood = ~np.isfinite(v)
    mask = mask & nogood
    xm = x[~mask]
    ym = y[~mask]
    vm = v[~mask]
    wm = weights[~mask]

    stepx2 = (limXY[1] - limXY[0]) / (nXY[0] - 1) / 2.  
    stepy2 = (limXY[3] - limXY[2]) / (nXY[1] - 1) / 2.  
    binX = np.linspace(limXY[0]-stepx2, limXY[1]+stepx2, nXY[0]+1)
    binY = np.linspace(limXY[2]-stepy2, limXY[3]+stepy2, nXY[1]+1)
    # moment 0
    mat_weight = np.histogram2d(xm, ym, [binX, binY], weights=wm)[0].T
    # moment 1
    mat_wvel = np.histogram2d(xm, ym, [binX, binY], weights=wm * vm)[0].T
    # moment 2
    mat_wvsquare = np.histogram2d(xm, ym, [binX, binY],
                                  weights=wm * vm**2)[0].T

    mask = (mat_weight != 0)
    mat_vel = np.zeros_like(mat_weight)
    mat_sig = np.zeros_like(mat_weight)
    mat_vel[mask] = mat_wvel[mask] / mat_weight[mask]
    mat_sig[mask] = np.sqrt(mat_wvsquare[mask] / mat_weight[mask] \
                    - mat_vel[mask] * mat_vel[mask])

    # Return : M, M*V, V, M*(V*V+S*S), S
    gridX, gridY = np.meshgrid(np.linspace(limXY[0], limXY[1], nXY[0]),
                               np.linspace(limXY[2], limXY[3], nXY[1]))
    return gridX, gridY, mat_weight, mat_vel, mat_sig

def points_2map(x, y, data=None, weights=None, limXY=[-1,1,-1,1],
                nXY=10, mask=None):
    """
    Calculate the first 2 velocity moments

    Parameters
    ----------
    x : array
        x coordinate
    y : array 
        y coordinate
    data : array
        Value to project 
    w : array
        Weight (e.g., mass) if None weights are set to 1.
    limXY : Tuple or Array
          (xmin, xmax, ymin, ymax)
    nXY : int or tupl of int
          Number of bins in X and Y. If nXY is a single integer
          nX and nY will be set both to nXY
          Otherwise, nXY can be a set of 2 integers (tuple/array)
          which will then translate into nX=nXY[0] and nY=nXY[1]

    Returns
    -------
    X,Y : arrays
        projected x's and y's
    M : array
        Projected mass
    V : array
        Projected velocity
    S : array
        Projected dispersion
    """
    if np.size(nXY) == 1 :
        nXY = np.zeros(2, dtype=np.int) + nXY
    elif np.size(nXY) != 2 :
        print("ERROR: dimension of n should be 1 or 2")
        return 0,0,0,0,0

    if mask is None: mask = np.zeros_like(x, dtype=bool)
    if weights is None : weights = np.ones_like(x)
    if data is None : data = np.ones_like(x)

    nogood = ~np.isfinite(data)
    mask = mask & nogood
    xm = x[~mask]
    ym = y[~mask]
    wm = weights[~mask]
    dm = data[~mask]

    stepx2 = (limXY[1] - limXY[0]) / (nXY[0] - 1) / 2.  
    stepy2 = (limXY[3] - limXY[2]) / (nXY[1] - 1) / 2.  
    binX = np.linspace(limXY[0]-stepx2, limXY[1]+stepx2, nXY[0]+1)
    binY = np.linspace(limXY[2]-stepy2, limXY[3]+stepy2, nXY[1]+1)
    mat_weight = (np.histogram2d(xm, ym, [binX, binY], weights=wm)[0]).T
    mat_wdata = (np.histogram2d(xm, ym, [binX, binY], weights=wm * dm)[0]).T

    mask = (mat_weight != 0)
    mat_data = np.zeros_like(mat_weight)
    mat_data[mask] = mat_wdata[mask] / mat_weight[mask]

    # Return : X, Y, Weights, Data
    gridX, gridY = np.meshgrid(np.linspace(limXY[0], limXY[1], nXY[0]),
                               np.linspace(limXY[2], limXY[3], nXY[1]))
    return gridX, gridY, mat_weight, mat_data, mat_wdata

def points_2losvds(x, y, data, weights=None, limXY=[-1,1,-1,1], nXY=10,
                      limV=[-1000,1000], nV=10, mask=None):
    """
    Calculate the first 2 velocity moments
    x : x coordinate
    y : y coordinate
    data : data to project
    weights weight (e.g., mass)

    limXY : (xmin, xmax, ymin, ymax)
    nXY : number of bins in X and Y. If nXY is a single integer
          nX and nY will be set both to nXY
          Otherwise, nXY can be a set of 2 integers (tuple/array)
          which will then translate into nX=nXY[0] and nY=nXY[1]
    limXY: [xmin, xmax, ymin, ymax]
    nXY: number of bins in X, Y
    limV : [vmin, vmax]
    nV : number of bins for V

    Return a 3D array with grid of X, Y and V
    """
    if np.size(nXY) == 1:
        nXY = np.zeros(2, dtype=np.int) + nXY
    elif np.size(nXY) != 2:
        print("ERROR: dimension of n should be 1 or 2")
        return 0,0,0,0,0

    # Defining the bins
    stepx2 = (limXY[1] - limXY[0]) / (nXY[0] - 1) / 2.  
    stepy2 = (limXY[3] - limXY[2]) / (nXY[1] - 1) / 2.  
    stepv2 = (limV[1] - limV[0]) / (nV - 1) / 2.  
    binX = np.linspace(limXY[0]-stepx2, limXY[1]+stepx2, nXY[0]+1)
    binY = np.linspace(limXY[2]-stepy2, limXY[3]+stepy2, nXY[1]+1)
    binV = np.linspace(limV[0]-stepv2, limV[1]+stepv2, nV+1)
    # Defining the centres of each pixel
    pixX = np.linspace(limXY[0], limXY[1], nXY[0])
    pixY = np.linspace(limXY[2], limXY[3], nXY[1])
    pixV = np.linspace(limV[0], limV[1], nV)

    if mask is None: mask = np.zeros_like(x, dtype=bool)
    sizeX = np.size(x[~mask])
    sample = np.hstack((x[~mask].reshape(sizeX,1), y[~mask].reshape(sizeX,1),
                        data[~mask].reshape(sizeX,1)))

    # Deriving the LOSVDs via histograms
    if weights is None:
        weights = np.ones_like(sample[:,0])
    locLOSVD = np.histogramdd(sample, bins=(binX, binY, binV), weights=weights)[0]

    # Return coordinates and LOSVD via the LOSVD class
    return LOSVD(pixX, pixY, pixV, locLOSVD)
