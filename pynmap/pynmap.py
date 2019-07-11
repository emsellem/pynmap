#!/usr/bin/python

"""
Program for computing projected quantities for snapshots
ESO / CRAL: @ 2019
Authors: Eric Emsellem
"""

__version__ = '0.0.1 (25-06, 2019)'

## Changes -- 
##   04/06/2019- EE - v0.0.1: creation gathering tools from other packages
"""
Import of the required modules
"""

# general modules
import sys
import os 
from os.path import join as joinpath
import copy

# Astropy and fits
import astropy
from astropy.io import fits as pyfits
from astropy.constants import G as Ggrav # m^3 *kg^-1 * s^-2

# std packages
import numpy as np

## Importing all function from the pytorque module
from .moments import *
from .rotation import *
from .misc_io import *

##############################################################
#----- Default formats ---------------------------------------
##############################################################
dic_io = {"Magneticum": read_ascii_files,
          "Ramses": read_ascii_files,
          "ascii": read_ascii_files}

default_scale = 1000.0
default_Vscale = 1000.0
default_lim = [-default_scale, default_scale, 
               -default_scale, default_scale]
default_Vlim = [-default_Vscale, default_Vscale]
default_npoint = 100
default_nVpoint = 100

######### Maps CLASS #################################
class SnapMap(object) :
    """New class to store the maps
    """
    def __init__(self, name="", info=None, input_dic={}) :
        """Initialise the nearly empty class
        Add _info for a description if needed
        """
        self._info = info
        self.name = name
        for item in input_dic.keys():
            setattr(self, item, input_dic[item])

    def info(self):
        print("Name {0}: {1}".format(self.name, self._info))

######### Maps CLASS #################################
class SnapMapList(list) :
    """Set of spectra
    """
    def __new__(self, *args, **kwargs):
        return super(SnapMapList, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)
        self.update(**kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.update(**kwargs)
        return self

    def update(self, **kwargs) :
        ## 1 required attribute
        if not hasattr(self, 'info') :
            if 'info' in kwargs :
                upipe.print_warning("Overiding info")
            self._info = kwargs.get('info', "")

    def info(self):
        print(self._info)

    def print(self):
        for item in self:
            print(item.name)

##########################################################
######### SNAPSHOT CLASS #################################
##########################################################
class snapshot(object):
    """Snapshot model class
    Attributes
    ----------
    folder : str
        file folder where the input images are
    filename : str
        Name of snapshot file
    verbose : bool
        Verbose option [False]
    """
    def __init__(self, folder="", snap_name=None, 
            snap_type="ascii",
            distance=10.0, MLrecipe="Adriano",
            inclination=0., PA=0.,
            verbose=True,
            info="Maps"):
        """Initialisation of the snapshot class
        Mainly setting up a few parameters like the distance
        and reading the data from the given input file

        Parameters
        ----------
        PA: float
            Input Position Angle in degrees of the snapshot
        inclination: float
            Input inclination in degrees of the snapshot
        distance: float
            Distance in Mpc [10.0]
        verbose: bool
            Verbose option [True]
        """

        ##=== Checking if the directory and file name exist.
        ##=== For this we use the os python module
        self.verbose = verbose

        ##=== Some useful number
        self.distance = distance # Galaxy distance in Mpc
        self.pc_per_arcsec = self.distance * np.pi / 0.648  # Conversion arcsec => pc (pc per arcsec)
        self.MLrecipe = "Adriano"

        ##=== angles
        self._PA_orig = PA  # PA at origin
        self._inclination_orig = inclination  # Inclination at origin

        ##== Checking the existence of the two images ===========================================
        if not os.path.exists(folder):
            print(('ERROR: Path %s' %(folder), ' does not exists, sorry!'))
            return
        self.folder = folder 

        if snap_name is None :
            print('ERROR: no filename for the main input snapshot file provided')
            return

        ##== We now save these information into the class
        self.snap_name = snap_name
        self.snap_type = snap_type
        in_name = joinpath(self.folder, self.snap_name)

        # Reading the input file
        self.read_snapshot(in_name)

        # initialise the maps
        self.reset_angles()

    def reset_angles(self):
        """Reset the Angles
        """
        self.PAs = []
        self.inclinations = []

    def read_snapshot(self, filename):
        """Reading the input file according to the filetype
        """
        ##== Reading the positions, velocities and Ages/metallicities
        if self.verbose:
            print("Opening Input file {0} ".format(
                   filename))
        self.pos, self.vel, self.mass, self.age, self.ZH = dic_io[self.snap_type](filename)
        self.npart = len(self.mass)
        # the original position, velocities
        self._pos_orig = copy.copy(self.pos)
        self._vel_orig = copy.copy(self.vel)
        # Compute the luminosity
        self.ML = compute_ML(self.mass, self.age, self.ZH, recipe=self.MLrecipe)
        self.flux = self.mass * self.ML
        # Get the copy for the original pos, vel
        self.reset_rotations()

    def reset_rotations(self):
        """Reseting the position and velocities to the original values
        """
        # the original position, velocities
        self.pos = copy.copy(self._pos_orig)
        self.vel = copy.copy(self._vel_orig)

    def rotastro(self, PA=0., inclination=90., reset=True, direct=True):
        """
        Performs a counter-clockwised rotation of PA and i with angles given in degrees
        for the positions and velocities of particles.
        The rotation is always done starting with the initial configuration
        (galaxy is seen face-on). So rotastro(0,0) returns to the initial configuration.

        Parameters
        ----------
        PA: float
            Position Angle (in degrees). Rotation around Z axis
        inclination: float
            Inclination (in degrees). Rotation around X axis
        reset: bool
            [False]. If True, resetting to original positions, velocities
            If False, accumulating the rotations
        direct: bool
            [True]. Sense of rotation. If True, direct sense.
        """
        if reset:
            # Going back to original configuration
            self.reset_rotations()

        # Convert to list in case these are just single values
        LPA = convert_to_list(PA)
        Linclination = convert_to_list(inclination)
        # Now do the requested rotations
        for pa, inclin in zip(LPA, Linclination):
            self.pos, self.vel = rotmat(self.pos, self.vel, 
                                    pa, inclin, 
                                    direct=direct)
            self.PAs.append(pa)
            self.inclinations.append(inclin)

        if self.verbose:
            print('Rotations done!')

    def info(self):
        """Printing some information
        """
        print("Input file name: {0} with type {1}".format(
                self.snap_name, self.snap_type))
        print("Number of particles: {0}".format(self.npart))
        print("Original PA={0} and inclination={1}".format(
               self._PA_orig, self._inclination_orig))
        print("Applied rotations: PAs        Inclinations")
        if len(self.PAs) > 0:
            k = 1
            for pa, inclin in zip(self.PAs, self.inclinations):
                print("{0}          {1:8.2f}       {2:8.2f}".format(
                        k, pa, inclin))
                k += 1
        else :
            print("No rotations applied (yet)")

    def comp_vel_moments(self, data=None, weight=None, 
                         lim=default_lim, nXY=default_npoint):
        """Compute velocity moments up to sigma
        """
        weights = self._select_weight(weight)
        if data is None:
            data = self.vel[:,1]

        X, Y, M, V, S = create_moment_maps(self.pos[:,0], self.pos[:,2], 
                                           data, weights=weights,
                                           lim=lim, nXY=nXY)
        mydict = {"X": X, "Y": Y, "V": V, "M": M, "S": S}
        mydict["PAs"] = self.PAs
        mydict["inclinations"] = self.inclinations
        newmap = SnapMap(name="Velocity moments", input_dic=mydict)
        return newmap

    def create_map(self, data=None, weight=None, 
                   lim=default_lim, nXY=default_npoint):
        """Create a map of some quantity
        """
        weights = self._select_weight(weight)

        if data is None:
            data = self.mass
        X, Y, W, D = project_data(self.pos[:,0], self.pos[:,2], data, 
                                   weights=weights, lim=lim, nXY=nXY)

        mydict = {"X": X, "Y": Y, "W": W, "D": D}
        mydict["PAs"] = self.PAs
        mydict["inclinations"] = self.inclinations
        newmap = SnapMap(name="map", input_dic=mydict)
        return newmap

    def _select_weight(self, weight=None):
        """Return the expected weights
        """
        if weight is None:
            return weight
        elif weight == "mass":
            weights = self.mass
        elif weight == "lum":
            weights = self.flux
        else:
            print("ERROR: please specify a weight function as 'mass' or 'lum'")
            return None

        return weights

    def create_losvds(self, data=None, weight=None, limXY=default_lim,
            nXY=default_npoint, limV=default_Vlim, nV=default_nVpoint):
        """Return the line of sight velocity distributions
        over a given grid
        """
        weights = self._select_weight(weight)
        if data is None:
            data = self.vel[:,1]
        mylosvd = create_losvds(self.pos[:,0], self.pos[:,2], data, 
                              weights=weights, limXY=limXY, nXY=nXY,
                              limV=limV, nV=nV)

        
        mylosvd.PAs = self.PAs
        mylosvd.inclinations = self.inclinations
        return mylosvd

    def get_eq_profiles(self, nbins=100, dz=np.array([-0.2, 0.2]), 
                       Rmin=1.e-2, Rmax=None, Vmax=None,
                       plot=False, saveplot=False, **kwargs):
        """Compute the profiles of the 3 dispersions in 
        cylindrical coordinates
        """

        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = (self.pos[:,0]**2 > -1.)

        Rbin, v, s = comp_eq_profiles(self.pos[:,0][mask], self.pos[:,1][mask], 
                                      self.pos[:,2][mask], self.vel[:,0][mask], 
                                      self.vel[:,1][mask], self.vel[:,2][mask],
                                      nbins=nbins, dz=dz, Rmin=Rmin, Rmax=Rmax,
                                      Vmax=Vmax, plot=plot, saveplot=saveplot, 
                                      **kwargs)
        return Rbin, v, s
