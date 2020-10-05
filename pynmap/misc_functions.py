# -*- coding: utf-8 -*-
"""This module has a few useful functions
"""

# General modules
import numpy as np
import os
from os.path import join as joinpath


# Specific modules
from astropy.table import Table
from scipy.interpolate  import griddata as gdata, interp1d 

def gausshermite(vbin=None, gh=None, default_nv=101) :
    """ Returns the Gauss-Hermite function given a set of parameters
    given as an array GH (first three moments are flux, velocity and dispersion)
    and the input sampling (velocities) Vbin

    Parameters
    ---------
    vcoord: array
        Velocity coordinates (centers)
    gh: array
        Set of Gauss-Hermite moments, starting with
        gh[0]: amplitude
        gh[1]: mean
        gh[2]: dispersion
        gh[3]: third moment
        ...

    Returns
    -------
    prof: array
        Profile given the set of velocity coordinates (same size as vcoord)
    """
    if vbin is None :
        vbin = np.linspace(-gh[2]*5. + gh[1], gh[2]*5. + gh[1], default_nv)

    # Degree is just length of gh minus 1
    degree = len(gh) - 1
    if degree < 2 :
        print("Error: no enough parameters here")
        return vbin * 0.

    # If second moment is 0, cannot compute the Gauss-Hermite...
    if gh[2] == 0. :
        print("Error: Sigma is 0!")
        return vbin * 0.

    VbinN = (vbin - gh[1]) / gh[2]
    VbinN2 = VbinN * VbinN

    # Initialisation of the profile using the first moments (0,1,2)
    GH0 = (2. * VbinN2 - 1.0) / np.sqrt(2.)
    GH1 = (2. * VbinN2 - 3.) * VbinN / np.sqrt(3.)
    GH2 = GH1

    # Recursive definition
    var = 1.0
    for i in range(3, degree+1) :
        var += gh[i] * GH2
        GH2 = (np.sqrt(2.) * GH1 * VbinN - GH0) / np.sqrt(i + 1.0);
        GH0 = GH1;
        GH1 = GH2;

    # Finally return profile with the right amplitude
    return gh[0] * var * np.exp(- VbinN2 / 2.)

def moment012(x, data) :
    """ Returns the first two moments of a 1D distribution
    e.g. a Line-of-Sight-Velocity-Distribution with arrays of V and amplitude

    Input:
    :param  xax : the input coordinates (1d array)
    :param data :  the input data (1d array, same size as xax)

    :returns m0, m1, m2 : the maximum amplitude and first two moments

    """
    if np.all(data == 0):
        m0 = 0.
        m1 = 0.
        m2 = np.min(np.diff(x)) / 3.0
    else:
        m0 = np.sum(data)
        m1 = np.average(x, weights=data)
        m2 = np.sqrt(np.average(x**2, weights=data) - m1**2)

    return m0, m1, m2

def compute_ml(mass, age=None, ZH=None, method="SSPs"):
    """Return the M/L ration given a set of input
    age and metallicity

    Parameters
    ----------
    mass: array
         Input mass array in Msun
    age: array
         Input age array in Gyr (same size as mass)
    ZH: array
         Input metallicity array (same size as mass)

    Returns
    -------
    ML: array
        M/L array - same shape as input mass
    """
    if method == "SSPs":
        return compute_ml_ssps(mass, age, ZH)
    else:
        return np.ones_like(mass)

def compute_ml_ssps(mass, age=None, MH=None,
                    recipe='EMILES', IMF='KB',
                    slope=1.30, iso='BaSTI', band='V',
                    fill_nan=True):
    """Return the M/L ratio given a set of input
    Parameters
    ----------
    mass: array [dtype=float]
        The stellar mass data, of shape (N,)
    age: array [dtype=float]
        The stellar age data, of shape (N,)
    MH: array [dtype=float]
        The stellar metallicity data, of shape (N,)
    recipe: str
        The recipe by which to compute the stellar mass-to-light
        ratio. Choose from `['EMILES']`
    IMF: str
        The assumed IMF. Choose from `['UN', 'BI', 'CH', 'KB']`,
        corresponding to the canonical single power-law, double broken
        power-law ("bimodal"), Chabrier, and revised Kroupa, respectively
    slope: float
        The slope of the high-mass end of the IMF
    iso: str
        The isochrone set to be used. Choose from `['BaSTI',
        'pad']`, corresponding to the BaSTI (Pietrinferni et al., 2004)
            and Padova (Girardi et al., 2000) isochrones, respectively
    band: str
        The photometric band in which to compute the stellar mass-
        to-light ratio. Choose from `['U', 'B', 'V', 'R', 'I', 'J', 'H',
        'K']` Johnson-Cousins filters

    Returns
    -------
    ML: array [dtype=float]
        The stellar mass-to-light ratio in `band`, of shape (N,)
    """
    # check that the age and metallicity have been specified
    if 'EMILES' in recipe and age is not None and MH is not None:
        # choice of iso
        dic_iso = {'padova': 'PADOVA00', 'basti': 'BASTI'}
        try:
            iso = dic_iso[iso.lower()]
        except:
            print("ERROR: iso not found in list ({0})".format(dic_iso.keys()))

        pynDir = os.path.split(os.path.realpath(__file__))[0]  # get the pynmap directory
        lfn = joinpath(pynDir, 'SSPLibraries',
                       "out_phot_{0}_{1}.txt".format(IMF, iso))
        mfn = joinpath(pynDir, 'SSPLibraries',
                       "out_mass_{0}_{1}.txt".format(IMF, iso))

        vegaBands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K']
        assert band in vegaBands, ("Photometric band '{}' "
                                   "is not in the EMILES "
                                   "SSP predictions".format(band))
        vegaSunMag = [5.600, 5.441, 4.820, 4.459,
                      4.148, 3.711, 3.392, 3.334]
        # Solar magnitude in various bands
        k = vegaBands.index(band)
        sunMag = vegaSunMag[k]

        mData = SSPMass(mfn)
        # create a mask, but allow for a tolerance in the keyword `slope` value
        mw = np.isclose(slope, mData['slope'], atol=1e-2)
        mAges = mData['age'][mw]
        mMetals = mData['MH'][mw]
        # get the mass in stars + remnants
        mMass = mData['mSRem'][mw]

        lData = SSPPhot(lfn)
        # create a mask, but allow for a tolerance in the keyword `slope` value
        lw = np.isclose(slope, lData['slope'], atol=1e-2)
        # lAges = lData['age'][lw]
        # lMetals = lData['MH'][lw]
        lFlux = 10 ** (-0.4 * (lData[band][lw] - sunMag))

        sspML = mMass / lFlux

        # interpolate the SSP predictions, and extract the M/L
        # for the given (`age`, `MH`) pairs
        ML = gdata((mAges, mMetals), sspML, (age, MH), method='cubic')

        if fill_nan:
            # Replacing the Nan by the nearest values
            whichNan = np.isnan(ML)
            ML[whichNan] = gdata((mAges, mMetals), sspML,
                                 (age[whichNan], MH[whichNan]),
                                 method='nearest')

        return ML

    else:
        return np.ones_like(mass)

def SSPMass(afile):
    """
    Reads in the mass predictions from SSP models

    Parameters
    ----------
    afile: str
        The path to the mass predictions

    Returns
    -------
    bData: dict
        A dictionary of all columns of the mass predictions
    """
    names = ['IMF', 'slope', 'MH', 'age', 'mTot', 'mSRem',
            'mS', 'mRem', 'mGas', 'mlSRemV', 'mlSV', 'mV', 'unk']
    bData = Table.read(afile, names=names, format='ascii')
    return bData

def SSPPhot(afile):
    """
    Reads in the photometric predictions from SSP models

    Parameters
    ----------
    afile: str
        The path to the photometric predictions

    Returns
    -------
    bData: dict
        A dictionary of all columns of the photometric predictions
    """
    names = ['IMF', 'slope', 'MH', 'age',
             'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K',
             'UV', 'BV', 'VR', 'VI', 'VJ', 'VH', 'VK',
             'mlU', 'mlB', 'mlV', 'mlR', 'mlI', 'mlJ', 'mlH',
             'mlK', 'F439W', 'F555W', 'F675W', 'F814W',
             'F439WF555W', 'F555WF675W', 'F555WF814W']
    bData = Table.read(afile, names=names, format='ascii')
    return bData

def find_centerodd(X, Y, Z, Radius=3.0) :
    """ Find central value for an odd sided field
        X : x coordinates
        Y : y coordinates
        Z : values
        Radius: radius within which to derive the central value
        Return the central value, and some amplitude value
    """
    # First select the points which are non-zero and compress
    ravZ = np.ravel(Z)
    sel = (ravZ != 0.)
    selz = np.compress(sel,ravZ)

    ## Select the central points for the central value
    selxy = np.ravel((np.abs(X) < Radius) & (np.abs(Y) < Radius)) & sel
    selzxy = np.compress(selxy, ravZ)
    cval = np.median(selzxy)

    sig = np.std(selz)
    sselz = np.compress(np.abs(selz - cval) < 3 * sig, selz - cval)
    ampl = np.max(np.abs(sselz)) / 1.1
    return cval, cval - ampl, cval + ampl

def find_centereven(X, Y, Z, Radius=3.0, sel0=True) :
    """ Find the central value for an even sided field
        X : x coordinates
        Y : y coordinates
        Z : data values
        Radius: radius within which to derive the central value
        Returns the min and max
    """
    ravZ = np.ravel(Z)
    if sel0 : sel = (ravZ != 0.)
    else : sel = (np.abs(ravZ) >= 0.)
    sel = np.ravel((np.abs(X) < Radius) & (np.abs(Y) < Radius)) & sel
    selz = np.compress(sel, ravZ)
    maxz = np.max(selz)
    minz = np.min(selz)

    return minz, maxz

def xy_cov_matrix2d(x, y, weights):
    """Given a set of coordinates and weights
    compute the covariance matrix for the coordinates
    """
    momI = weights.sum()
    if momI == 0. :
        return np.array([[1., 0.], [0., 1.]])

    momIX = (weights * x).sum() / momI
    momIY = (weights * y).sum() / momI
    a = (weights * x * x).sum() / momI - momIX**2
    c = (weights * y * y).sum() / momI - momIY**2
    b = (weights * x * y).sum() / momI - momIX * momIY
    return np.array([[a, b], [b, c]])

def characterise_ima2d(x, y, flux, nbins=30, minfrac=1.e-5, maxfrac=0.99, facn=3, mask=None):
    """Characterise a 2d flux image
    """

    rad = np.zeros(nbins*facn)
    prof = np.zeros_like(rad)

    eps = np.zeros(nbins)
    pa = np.zeros_like(eps)
    flux_prof = np.zeros_like(eps)
    if mask is None:
        mask = (np.abs(flux) < 0)
    tflux = flux[~mask].sum()

    flux_samp = (np.logspace(np.log10(minfrac), np.log10(maxfrac), 
                             nbins * facn) * np.max(flux[~mask]))[::-1]
    # Going from centre to outer parts - R increasing
    for i, g in enumerate(flux_samp):
        xd, yd, fd, [radi, l1, l2, epsd, pad] = morph_ima2d(x, y, flux, ground=g, mask=mask)
        prof[i] = fd.sum()
        rad[i] = radi

    # Select the ante penultieme value
    ind_rad = np.argwhere(rad == np.max(rad))
    if len(ind_rad) < 0:
        print("Problem selecting valid pixels")
        return [0], [0], [0], [0], 0.
    # If found
    ind_rad = ind_rad[0][0]

    sel_rad = (rad[:ind_rad] > 0.)
    inv_cumf = interp1d(prof[:ind_rad][sel_rad], rad[:ind_rad][sel_rad])
    ground_func = interp1d(rad[:ind_rad][sel_rad], flux_samp[:ind_rad][sel_rad])
    re_50 = inv_cumf(tflux / 2.)

    # reinterpolating the profiles
    rsample = np.linspace(np.min(rad[:ind_rad][sel_rad]), 
                          np.max(rad[:ind_rad][sel_rad]), nbins)
    for i, r in enumerate(rsample):
        flux_prof[i] = ground_func(r)
        xd, yd, fd, [radi, l1, l2, eps[i], pa[i]] = morph_ima2d(x, y, flux, ground=flux_prof[i], mask=mask)

    inv_pa = interp1d(rsample, pa)
    inv_eps = interp1d(rsample, eps)
    eps_50 = inv_eps(re_50)
    pa_50 = inv_pa(re_50)

    return rsample, flux_prof, eps, pa, re_50, eps_50, pa_50

def guess_stepx(xarray):
    pot_step = np.array([np.min(np.abs(np.diff(xarray, axis=i))) for i in range(xarray.ndim)])
    return np.min(pot_step[pot_step > 0.])

def guess_stepxy(Xin, Yin, index_range=[0,100], verbose=False) :
    """Guess the step from a 1 or 2D grid
    Using the distance between points for the range of points given by
    index_range

    Parameters:
    -----------
    Xin, Yin: input (float) arrays
    index_range : tuple or array of 2 integers providing the min and max indices = [min, max]
            default is [0,100] to make it faster
    verbose: default is False

    Returns
    -------
    step : guessed step (float)
    """
    # Stacking the first 100 points of the grid and determining the distance
    stackXY = np.vstack((Xin.ravel()[index_range[0]: index_range[1]], Yin.ravel()[index_range[0]: index_range[1]]))
    diffXY = np.unique(distance.pdist(stackXY.T))
    step = np.min(diffXY[diffXY > 0])
    if verbose:
        print("New step will be %s"%(step))

    return step

def morph_ima2d(x, y, flux, ground=0., ceiling=np.inf, mask=None):
    """Derive the morphology of an image after selecting
    the given glux

    Input
    -----
    x, y (arrays): coordinates
    flux (array): flux array

    Returns
    -------
    xs, ys, fs, [r, l1, l2, eps, pa]
        which are:
        selected Xs, Ys, Fs arrays 
        and effective area radius, eps, pa
        l1 and l2 being the major and minor axes
    """
    # selecting pixels which are above the threshold
    if mask is None:
        mask = (np.abs(flux) < 0)

    selp = (flux > ground) & (flux < ceiling) & (~mask)
    xs = x[selp]
    ys = y[selp]
    dx = guess_stepx(x)
    dy = guess_stepx(y)
    fs = flux[selp]

    covmat = xy_cov_matrix2d(xs, ys, fs)
    l1, l2, eps, pa = comp_pa_eps(covmat)
    rad = np.sqrt(xs.size * dx * dy / np.pi)
    return xs, ys, fs, [rad, l1, l2, eps, pa]
     
def comp_pa_eps(covmat, threshold=1.e-10):
    """Return the PA and Eps for a given set of covariance 
    parameters

    Input
    -----
    covmat: covariance matrix

    Returns
    -------
    major, minor axes, eps, pa
    """
    m00, m11, m01 = covmat[0,0], covmat[1,1], covmat[1,0]

    if m00 < threshold : m00 = 0.
    if m11 < threshold : m11 = 0.

    if m01 == 0 :
       if m00 == 0. :
          return m11, m00, 0., 90.
       if m11 > m00 :
          return m11, m00, 1. - np.sqrt(m00 / m11), 0.
       else :
          if m11 == 0.:
             return m00, m11, 0., 90.
          else :
             return m00, m11, 1. - np.sqrt(m11 / m00), 90.

    delta = (m00 - m11)**2. + 4 * m01**2
    lambda0 = ((-m00 + m11) + np.sqrt(delta)) / 2.
    lambda1 = ((m00 + m11) + np.sqrt(delta)) / 2.
    lambda2 = ((m00 + m11) - np.sqrt(delta)) / 2.
    lp = np.sqrt(lambda1)
    lm = np.sqrt(np.maximum(lambda2,0.))
    if lp == 0. :
        eps = 1.
    else:
        eps = 1. - lm / lp

    theta = np.rad2deg(np.arctan2(lambda0, m01)) + 90.0
    return lp, lm, eps, theta
