# -*- coding: utf-8 -*-
"""
This module is the main module for pynmap. It provides input snapshot
class to be used together with I/O

ESO / CRAL: @ 2019
Authors: Eric Emsellem
"""
# general modules
import copy
import os
from os.path import join as joinpath

# numpy
import numpy as np
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve_fft

# Importing all functions from the i/o and moments + rotation
from .misc_io import read_ascii_files, convert_to_list, spread_amr
from .grid import (points_2losvds, points_2map, points_2vmaps,
                   point_2vprofiles)
from .misc_functions import compute_ml
from .rotation import rotate_mat

# Default formats for the I/O files
dic_io = {"Magneticum": read_ascii_files,
          "ascii": read_ascii_files}

# Default scales and limits
default_scale = 1000.0
default_Vscale = 1000.0
default_limXY = [-default_scale, default_scale,
                 -default_scale, default_scale]
default_limV = [-default_Vscale, default_Vscale]
default_npoint = 100
default_nVpoint = 100


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
        self.input_keys = list(input_dic.keys())

    def info(self):
        print("Name {0}: {1}".format(self.name, self._info))
        print("List of keys: {}".format(self.input_keys))

    @property
    def extent(self):
        if hasattr(self, 'X') & hasattr(self, 'Y'):
            return [np.min(self.X), np.max(self.X), 
                    np.min(self.Y), np.max(self.Y)]
        else:
            return None

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
        # 1 required attribute
        if not hasattr(self, 'info') :
            if 'info' in kwargs :
                upipe.print_warning("Overiding info")
            self._info = kwargs.get('info', "")

    def info(self):
        print(self._info)

    def print(self):
        for item in self:
            print(item.name)


class snapshot(object):
    """Snapshot model class

    Attributes
        folder : str
            file folder where the input images are
        filename : str
            Name of snapshot file
        verbose : bool
            Verbose option [False]
    """
    def __init__(self, folder="", snap_name=None, 
            snap_type="ascii",
            distance=10.0, MLrecipe="SSPs",
            inclination=0., PA=0.,
            verbose=True,
            extra_labels=[]):
        """Initialisation of the snapshot class
        Mainly setting up a few parameters like the distance
        and reading the data from the given input file

        Args:
            folder (str): input folder name
            snap_name (str): name of the snapshot file
            snap_type (str): type of the snapshot
            distance (float): distance of the galaxy (Mpc)
            MLrecipe (str): 'SSP' or none
            inclination (float): initial inclination value (degrees)
            PA (float): input position angle (degrees)
            verbose (bool):
            info:
            extra_labels:
        """
        # Checking if the directory and file name exist.
        # For this we use the os python module
        self.verbose = verbose

        # Some useful number
        self.distance = distance # Galaxy distance in Mpc
        self.pc_per_arcsec = self.distance * np.pi / 0.648  # Conversion arcsec => pc (pc per arcsec)
        self.MLrecipe = MLrecipe

        # angles
        self._PA_orig = PA  # PA at origin
        self._inclination_orig = inclination  # Inclination at origin

        # We now save these information into the class
        self.snap_type = snap_type
        self.extra_labels = extra_labels

        self.folder = folder 
        if snap_name is None:
            print('ERROR: no filename for the main input snapshot file provided')
            return
        self.snap_name = snap_name
        in_name = joinpath(self.folder, self.snap_name)
        if not os.path.isfile(in_name):
            print('ERROR: filename {} does not exists'.format(in_name))
            return

        # Reading the input file
        self.read(in_name)

        # initialise the maps
        self._reset_rotations()

    def _check_attrs(self, list_attr=['']):
        """Check if all attributes are there

        Args:
            list_attr:

        Returns:

        """
        return all(hasattr(self, attr) for attr in list_attr)

    def _reset_rotations(self):
        """Reset the Angles
        """
        self.PAs = []
        self.inclinations = []

    def read(self, filename):
        """Reading the input file according to the filetype
        """
        # Reading the positions, velocities and Ages/metallicities
        if self.verbose:
            print("Opening Input file {0} ".format( filename))
            print("By default, will read all 6 positions/velocities: "
                  "x, y, z, Vx, Vy, Vz")
        self.pos, self.vel, data = dic_io[self.snap_type](filename)

        if self.pos is None:
            print("ERROR: file may be empty. \n"
                  "       Did not succeed to read positions and velocities.\n"
                  "       - Aborting")
            return

        if data.shape[0] != len(self.extra_labels):
            print("WARNING: cannot convert data beyond positions/velocities/masses")
        else:
            for i, lab in enumerate(self.extra_labels):
                setattr(self, lab, data[i])

        if not hasattr(self, 'mass'):
            self.mass = np.ones_like(self.pos[:,0])

        if not hasattr(self, 'age'):
            if hasattr(self, 'logage'):
                # Going into linear age and MH
                self.age = 10**(self.logage)
            else:
                self.age = np.zeros_like(self.mass)

        if not hasattr(self, 'MH'):
            if hasattr(self, 'ZH'):
                self.MH = np.log10(self.ZH) - np.log10(0.19)
            else:
                self.MH = np.zeros_like(self.mass)

        self.npart = len(self.mass)
        # the original position, velocities
        self._pos_orig = copy.copy(self.pos)
        self._vel_orig = copy.copy(self.vel)

        # Compute the luminosity
        self.ML = self.compute_ml()

        # mask
        self.mask = (self.mass == 0)

        # Get the copy for the original pos, vel
        self.reset()

    def compute_ml(self):
        self.ML = compute_ml(self.mass, self.age, self.MH, method=self.MLrecipe)
        self.flux = self.mass / self.ML

    def reset(self):
        """Reseting the position and velocities to the original values
        """
        # the original position, velocities
        self.pos = copy.copy(self._pos_orig)
        self.vel = copy.copy(self._vel_orig)
        self._reset_rotations()

    def rotate(self, PA=0., inclination=90., reset=True, direct=True):
        """
        Performs a counter-clockwised rotation of PA and i with angles given in degrees
        for the positions and velocities of particles.
        The rotation is always done starting with the initial configuration
        (galaxy is seen face-on). So rotate(0,0) returns to the initial configuration.

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
        if self.pos is None:
            print('WARNING: no positions - Aborting rotation')
            return

        if reset:
            # Going back to original configuration
            self.reset()

        # Convert to list in case these are just single values
        LPA = convert_to_list(PA)
        Linclination = convert_to_list(inclination)
        # Now do the requested rotations
        for pa, inclin in zip(LPA, Linclination):
            self.pos, self.vel = rotate_mat(self.pos, self.vel,
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
    def velocity_moments(self, data=None, weights=None, weight_name=None,
                         limXY=default_limXY, nXY=default_npoint, **kwargs):
        """Compute velocity moments up to sigma
        """
        if self.pos is None:
            print("WARNING: no positions - Aborting")
            return SnapMap(name="EmptyMap")

        if weight_name is not None:
            weights = self._select_weight(weight_name)
        if data is None:
            data = self.vel[:,2]
            print("WARNING: using z component of V as data")

        X, Y, M, V, S = points_2vmaps(self.pos[:,0], self.pos[:,1], data,
                                      weights=weights, limXY=limXY,
                                      nXY=nXY, **kwargs)
        mydict = {"X": X, "Y": Y, "V": V, "M": M, "S": S}
        mydict["PAs"] = self.PAs
        mydict["inclinations"] = self.inclinations
        newmap = SnapMap(name="Velocity moments", input_dic=mydict)
        return newmap

    def amr_velocity_moments(self, data=None, weights=None,
                             weight_name=None, box=150.,
                             limXY=default_limXY, nXY=default_npoint,
                             sigfac=10, smear=False, spread=True, **kwargs):
        """Compute velocity moments up to sigma for amr gas
        """
        if not self._check_attrs(['level']):
            print("ERROR: missing some input info in class")
            return None

        if weight_name is not None:
            weights = self._select_weight(weight_name)
        if data is None:
            data = self.vel[:,2]

        # Getting the levels and scales
        list_levels = np.unique(self.level)
        if np.size(nXY) == 2: nX, nY = nXY[0], nXY[1]
        else: nX = nY = nXY
        scalex =(limXY[1] - limXY[0]) / (nX-1)
        scaley =(limXY[3] - limXY[2]) / (nY-1)

        inv_sigfac = 1./ sigfac
        # Deriving the missing info and the spread if needed
        if weights is None: weights = np.ones_like(self.mass)
        if spread:
            [x, y], [weightr], [datar, levelr] = spread_amr([self.pos[:,0],
                                                          self.pos[:,1]],
                                                          self.level,
                                                          list_val=[weights],
                                                          list_const = [data, self.level],
                                                          nsample=10, box=box,
                                                          stretch=[1.,1.])
        else:
            x, y, datar, weightr, levelr = self.pos[:,0], self.pos[:,1], \
                                           data, weights, self.level

        if smear:
            M = np.zeros((nY, nX), dtype=np.float32)
            V = np.zeros((nY, nX), dtype=np.float32)
            S = np.zeros((nY, nX), dtype=np.float32)

            for l in list_levels:
                cells = box / 2**(l)
                selgas = (levelr == l)
                X, Y, Mf, Vt, St = points_2vmaps(x[selgas], y[selgas],
                                                 datar[selgas],
                                                 weights=weightr[selgas],
                                                 limXY=limXY, nXY=nXY, **kwargs)
                Vf = Mf * Vt
                Muf = Mf * (Vt ** 2 + St ** 2)

                # Convolving each one in turn with a Gaussian kernel
                stdx = 2. * cells * gaussian_fwhm_to_sigma / scalex
                stdy = 2. * cells * gaussian_fwhm_to_sigma / scaley
                if (stdx > inv_sigfac) & (stdy > inv_sigfac) & smear:
                    kernel = Gaussian2DKernel(stdx, stdy, x_size=np.int(sigfac*stdx),
                                              y_size=np.int(sigfac*stdy))
                    Mf = convolve_fft(Mf, kernel, fft_pad=False, fill_value=0)
                    Vf = convolve_fft(Vf, kernel, fft_pad=False, fill_value=0)
                    Muf = convolve_fft(Muf, kernel, fft_pad=False, fill_value=0)

                M += Mf
                V += Vf
                S += Muf

            V /= M
            S = np.sqrt(Muf / M - V**2)

        else:
            X, Y, M, V, S = points_2vmaps(x, y, datar, weights=weightr,
                                          limXY=limXY, nXY=nXY)

        mydict = {"X": X, "Y": Y, "V": V, "M": M, "S": S}
        mydict["PAs"] = self.PAs
        mydict["inclinations"] = self.inclinations
        newmap = SnapMap(name="Velocity moments", input_dic=mydict)
        return newmap

    def create_map(self, data=None, weights=None, weight_name=None,
                   limXY=default_limXY, nXY=default_npoint, **kwargs):
        """Create a map of some quantity
        """
        if weight_name is not None:
            weights = self._select_weight(weight_name)

        if data is None:
            data = self.mass
        X, Y, W, D, DW = points_2map(self.pos[:,0], self.pos[:,1], data,
                                 weights=weights, limXY=limXY, nXY=nXY, **kwargs)

        mydict = {"X": X, "Y": Y, "W": W, "D": D, "DW": DW}
        mydict["PAs"] = self.PAs
        mydict["inclinations"] = self.inclinations
        newmap = SnapMap(name="map", input_dic=mydict)
        return newmap

    def create_losvds(self, data=None, weights=None, weight_name=None,
                      limXY=default_limXY, nXY=default_npoint,
                      limV=default_limV, nV=default_nVpoint, **kwargs):
        """Return the line of sight velocity distributions
        over a given grid
        """
        if weight_name is not None:
            weights = self._select_weight(weight_name)
        if data is None:
            data = self.vel[:,2]
        mylosvd = points_2losvds(self.pos[:,0], self.pos[:,1], data,
                                 weights=weights, limXY=limXY, nXY=nXY,
                                 limV=limV, nV=nV, **kwargs)

        mylosvd.PAs = self.PAs
        mylosvd.inclinations = self.inclinations
        return mylosvd

    def get_tensor_profiles(self, nbins=100, dz=np.array([-0.2, 0.2]),
                       Rmin=1.e-2, Rmax=None, Vmax=None,
                       plot=False, saveplot=False, **kwargs):
        """Compute the profiles of the 3 dispersions in 
        cylindrical coordinates
        """

        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = (self.pos[:,0]**2 > -1.)

        Rbin, v, s = point_2vprofiles(self.pos[:,0][mask], self.pos[:,1][mask],
                                      self.pos[:,2][mask], self.vel[:,0][mask],
                                      self.vel[:,1][mask], self.vel[:,2][mask],
                                      nbins=nbins, dz=dz, Rmin=Rmin, Rmax=Rmax,
                                      Vmax=Vmax, plot=plot, saveplot=saveplot,
                                      **kwargs)
        return Rbin, v, s


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
        from matplotlib import pyplot as plt
        fig = plt.figure(1, figsize=(8,7))
        plt.imshow(self.data)
        plt.savefig(filename)


