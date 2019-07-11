"""Module misc_io containing a set of useful I/O routines 
and classes. This includes reading the inputs files from
the simulations.
"""
import os
from os.path import join as joinpath

import numpy as np
from numpy import float64 as floatF
from numpy import float32 as floatsF

from scipy.interpolate  import griddata as gdata

#####################################################
#------ Reading Magneticum files --------#
def read_ascii_files(filename):
    """Reading ascii formatted files
    File is formatted with first the 3 configuration spatial coordinates
    Then the velocities, then mass, then possibly Age and Metallicity

    Parameters
    ----------
    filename: str
        Name of the input ascii file. Should contain one particle per line with
        x y z Vx Vy Vz mass age Z/H

    Returns
    -------
    pos: array [3,N]
        Configuration cartesian positions [kpc] for N particles. 
    vel: array [3,N]
        Cartesian velocities [kpc] for N particles. 
    mass: array [N]
        Masses
    age: array [N]
        Ages 
    zh: array [N]
        Z/H
    """

    # Reading the input file
    if os.path.isfile(filename):
        try:
            data = np.loadtxt(filename).T
            if data.shape[0] == 7:
                x, y, z, vx, vy, vz, mass = data
                age = np.zeros_like(mass)
                ZH = np.zeros_like(mass)
            elif data.shape[0] == 8:
                x, y, z, vx, vy, vz, mass, age = data
                ZH = np.zeros_like(mass)
            elif data.shape[0] == 9:
                x, y, z, vx, vy, vz, mass, age, ZH = data
            else:
                print("ERROR: the input file should at least "
                      "contain 7 columns or no more than 9")
                return [None]*3, [None]*3, [None], [None], [None]

            pos = np.vstack((x, y, z)).T
            vel = np.vstack((vx, vy, vz)).T
            return pos, vel, mass, age, ZH

        except ValueError:
            print("ERROR: could not read content of the input file")

    else:
        print("Input file not found: {0}".format(filename))
        return [None]*3, [None]*3, [None], [None], [None]

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
             'UV', 'BV', 'VR', 'VI', 'VH', 'VK', 'mlU', 
             'mlB', 'mlV', 'mlR', 'mlI', 'mlJ', 'mlH', 
             'mlK', 'F439W', 'F555W', 'F675W', 'F814W', 
             'F439WF555W', 'F555WF675W', 'F555WF814W']
    bData = Table.read(afile, names=names, format='ascii')
    return bData

# Function to return the value of M/L following a recipe
# depending on the given age and metallicity
def compute_ML(mass, age=None, ZH=None, recipe="Adriano"):
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
    if recipe == "SSPs":
        return compute_ML_SSPs(mass, age, ZH)
    else:
        return np.ones_like(mass)

def compute_ML_SSPs(mass, age=None, ZH=None, 
                    recipe='EMILES', IMF='KB', 
                    slope=1.30, iso='BaSTI', band='V'):
    """Return the M/L ratio given a set of input
    Parameters
    ----------
    mass: array [dtype=float]
        The stellar mass data, of shape (N,)
    age: array [dtype=float]
        The stellar age data, of shape (N,)
    ZH: array [dtype=float]
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
    if 'EMILES' in recipe and age is not None and ZH is not None:
        # choice of iso
        dic_ico = {'padova': 'PADOVA00', 'basti': 'BASTI'}
        try: 
            iso = dic_iso[iso.lower()]
        except:
            print("ERROR: iso not found in list ({0})".format(dic_iso.keys()))
        
        pynDir = os.path.split(os.path.realpath(__file__))[0] # get the pynmap directory
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
        lAges = lData['age'][lw]
        lMetals = lData['MH'][lw]
        lFlux = 10**(-0.4*(lData[band][lw] - sunMag))
        
        sspML = mMass / lFlux
        
        # fudge the abundances until they can be implemented 
        # correctly based on what the simulations actually measure
        MH = np.log10(ZH) - np.log10(0.19) 
        # interpolate the SSP predictions, and extract the M/L 
        # for the given (`age`, `ZH`) pairs
        ML = gdata((mAges, mMetals), sspML, (age,MH), method='cubic') 
        
        return ML
        
    else:
        return np.ones_like(mass)

######### MAP class #################################
class snapmap(object):
    """Snapshot map class which contains just a name
    and some data array. Has functionalities to save
    as graphics, ascii or pickle

    """
    def __init__(self, name="dummy", data=np.zeros((2,2))):
        """Initialise snapmap class
        Parameters
        ----------
        name: str
            Name of the input map
        data: array
            Data array
        """
        self.name = name
        self.data = data

    def write_ascii(self, filename):
        """Saving the map into an ascii file
        """
        np.savetxt(filename, self.data)

    def read_ascii(self, filename):
        """Reading map from ascii
        """
        self.data = np.loadtxt(filename)

    def write_pickle(self, filename):
        """Saving the map via pickle
        """
        f = open(filename, 'wb')
        pickle.dump(self.data, f)
        f.close()

    def read_pickle(self, filename):
        """Saving the map via pickle
        """
        f = open(filename, 'rb')
        self.data = pickle.load(f)
        f.close()

    def write_pic(self, filename):
        """Write a png or jpg
        """
        import matplotlib
        from matplotlib import pyplot as plt
        fig = plt.figure(1, figsize=(8,7))
        plt.imshow(self.data)
        plt.savefig(filename)

#####################################################
#----------------------- General functions ---------#
def return_dummy(n=1, default=0):
    return [default] * n

def convert_to_list(a):
    """Convert a float into a list
    """
    try:
        a = [floatsF(a)]
    except ValueError:
        if not isinstance(a):
            print("ERROR: The number provided is not a proper number")
            return None
    return a
    

