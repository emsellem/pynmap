"""
This module allows to compute histograms in an efficient way
"""

"""
Importing the most important modules
"""
import numpy as np
from .rotation import xy_to_polar

__version__ = '0.0.1 (4 July 2019)'
# 04/06/2019 - Version 0.0.1: - Creation, gathering from other modules

def comp_eq_profiles(x, y, z, Vx, Vy, Vz, nbins=100, 
                   dz=np.array([-0.2, 0.2]), 
                   Rmin=1.e-2, Rmax=None, Vmax=None,
                   plot=False, saveplot=False, **kwargs):
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

def plot_sigma(Rbin, v, s, Rmin=0., Rmax=None, Vmax=None,
               snapname="Snapshot", suffix="", 
               figure_folder="Figures/", save=False) :
    """Plot the 3 sigma profiles (cylindrical coordinates)
    """
    import matplotlib
    from matplotlib import pyplot as plt

    if Vmax is None: 
        Vmax = np.maximum(np.max(v), np.max(s)) * 1.1

    plt.clf()
    plt.plot(Rbin, s[0], 'r-', label=r'$\sigma_R$')
    plt.plot(Rbin, s[1], 'b-', label=r'$\sigma_{\theta}$')
    plt.plot(Rbin, s[2], 'k-', label=r'$\sigma_z$')
    plt.plot(Rbin, v[0], 'r--', label=r'$V_R$')
    plt.plot(Rbin, v[1], 'b--', label=r'$V_{\theta}$')
    plt.plot(Rbin, v[2], 'k--', label=r'$V_z$')
    plt.legend()
    plt.xlabel("R [kpc]", fontsize=20)
    plt.ylabel("$\sigma$ [km/s]", fontsize=20)
    plt.xlim(Rmin, Rmax)
    plt.ylim(0, Vmax)
    plt.title("Model = %s - %s"%(snapname, suffix))
    if save:
        plt.savefig(figure_folder+"Fig_%s_%s.png"%(snapname, suffix))

## ====== Create a map of the velocity moments =========== ##
def create_moment_maps(x, y, v, weights=None, 
                       lim=[-1,1,-1,1], nXY=10):
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
    lim : Tuple or Array
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
    binX = np.linspace(lim[0], lim[1], nXY[0]+1)
    binY = np.linspace(lim[2], lim[3], nXY[1]+1)
    # moment 0
    mat_weight = np.histogram2d(x, y, [binX, binY], weights=weights)[0]
    # moment 1
    mat_wvel = np.histogram2d(x, y, [binX, binY], weights=weights * v)[0]
    # moment 2
    mat_wvsquare = np.histogram2d(x, y, [binX, binY], 
                                  weights=weights * v**2)[0]

    mask = (mat_weight != 0)
    mat_vel = np.zeros_like(mat_weight)
    mat_sig = np.zeros_like(mat_weight)
    mat_vel[mask] = mat_wvel[mask] / mat_weight[mask]
    mat_sig[mask] = np.sqrt(mat_wvsquare[mask] / mat_weight[mask] \
                    - mat_vel[mask] * mat_vel[mask])

    # Return : M, M*V, V, M*(V*V+S*S), S
    gridX, gridY = np.meshgrid(np.linspace(lim[0], lim[1], nXY[0]), 
                               np.linspace(lim[2], lim[3], nXY[1]))
    return gridX, gridY, mat_weight, mat_vel, mat_sig
##----------------------------------------------------------
############################################################
def project_data(x, y, data, weights=None, lim=[-1,1,-1,1], nXY=10):
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
    lim : Tuple or Array
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

    if weights is None : weights = np.ones_like(x)
    binX = np.linspace(lim[0], lim[1], nXY[0]+1)
    binY = np.linspace(lim[2], lim[3], nXY[1]+1)
    mat_weight = np.histogram2d(x, y, [binX, binY], weights=weights)[0]
    mat_wdata = np.histogram2d(x, y, [binX, binY], weights=weights * data)[0]

    mask = (mat_weight != 0)
    mat_data = np.zeros_like(mat_weight)
    mat_data[mask] = mat_wdata[mask] / mat_weight[mask]

    # Return : M, M*V, V, M*(V*V+S*S), S
    gridX, gridY = np.meshgrid(np.linspace(lim[0], lim[1], nXY[0]), 
                               np.linspace(lim[2], lim[3], nXY[1]))
    return gridX, gridY, mat_weight, mat_data
##----------------------------------------------------------
############################################################
def create_losvds(x, y, data, weights=None, limXY=[-1,1,-1,1], 
               nXY=10, limV=[-1000,1000], nV=10):
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
    limV : (vmin, vmax)

    nV : number of bins for V

    Return a 3D array with grid of X, Y and V
    """
    if np.size(nXY) == 1:
        nXY = np.zeros(2, dtype=np.int) + nXY
    elif np.size(nXY) != 2:
        print("ERROR: dimension of n should be 1 or 2")
        return 0,0,0,0,0

    ## Defining the bins
    binX = np.linspace(limXY[0], limXY[1], nXY[0]+1)
    binY = np.linspace(limXY[2], limXY[3], nXY[1]+1)
    binV = np.linspace(limV[0], limV[1], nV+1)
    ## Defining the centres of each pixel
    pixX = (binX[1:] + binX[0:-1]) / 2.
    pixY = (binY[1:] + binY[0:-1]) / 2.
    pixV = (binV[1:] + binV[0:-1]) / 2.
    
    sizeX = np.size(x)
    sample = np.hstack((x.reshape(sizeX,1), y.reshape(sizeX,1), data.reshape(sizeX,1)))
    ## Coordinates of the bins
    #    coords = np.asarray(np.meshgrid(pixX, pixY, pixV, indexing='ij'))

    ## Deriving the LOSVDs via histograms
    if weights is None:
        weights = np.ones_like(sample[:,0])
    locLOSVD = np.histogramdd(sample, bins=(binX, binY, binV), weights=weights)[0]

    # Return coordinates and LOSVD via the LOSVD class
    return LOSVD(pixX, pixY, pixV, locLOSVD)
##----------------------------------------------------------
############################################################
# Small class for LOSVDs
# Including the sampling in X, Y and V
############################################################
class LOSVD(object) :
    """ Provide a class for LOSVDs
    which includes the coordinates (X, Y, V)
    """
    def __init__(self, binx=None, biny=None, binv=None, losvd=None) :
        """Class for Line of Sight Velocity distributions
        """
        self.binx = binx
        self.biny = biny
        self.binv = binv
        self.losvd = losvd

