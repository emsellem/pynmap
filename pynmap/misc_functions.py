# -*- coding: utf-8 -*-
"""This module has a few useful functions
"""

# General modules
import numpy as np
import os
from os.path import join as joinpath

# Specific modules
from astropy.table import Table
from scipy.interpolate  import griddata as gdata

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
    m0 = np.max(data)
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
