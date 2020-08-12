# -*- coding: utf-8 -*-

"""Module misc_io containing a set of useful I/O routines
and classes. This includes reading the inputs files from
the simulations.
"""

import os

import numpy as np
from numpy import float32 as floatsF

def read_rdramses_part(filename):
    """Reading RDRAMSES files from hydro or particles
    File is formatted with first the 3 configuration spatial coordinates
    Then the velocities, then mass, then possibly Age and Metallicity

    Parameters
    ----------
    filename: str
        Name of the input ascii file. Should contain one particle per line with
        x y z Vx Vy Vz mass age Z/H and possibly Fe/H or O/H

    Returns
    -------
    pos: array [3,N]
        Configuration cartesian positions [kpc] for N particles. 
    vel: array [3,N]
        Cartesian velocities [kpc] for N particles. 
    mass: array [N]
        Masses
    age: array [N]
        Ages (Myr)
    mh: array [N]
        M/H
    feh: array [N]
        Fe/H
    oh: array [N]
        O/H
    """
    # Reading the input file
    if os.path.isfile(filename):
        try:
            data = np.loadtxt(filename).T
            if data.shape[0] == 7:
                print("Found 7 columns, will not read age and Z/H")
                x, y, z, vx, vy, vz, mass = data
                age = np.zeros_like(mass)
                ZH = np.zeros_like(mass)
                OH = np.zeros_like(mass)
                FeH = np.zeros_like(mass)
            elif data.shape[0] == 8:
                print("Found 8 columns, will not read Z/H")
                x, y, z, vx, vy, vz, mass, age = data
                ZH = np.zeros_like(mass)
                OH = np.zeros_like(mass)
                FeH = np.zeros_like(mass)
            elif data.shape[0] == 9:
                print("Found 9 columns, will read all including age and Z/H")
                x, y, z, vx, vy, vz, mass, age, ZH = data
                OH = np.zeros_like(mass)
                FeH = np.zeros_like(mass)
            elif data.shape[0] == 11:
                print("Found 11 columns, will read all including age and Z/H + Fe/H and O/H")
                x, y, z, vx, vy, vz, mass, age, ZH, FeH, OH = data
            else:
                print("ERROR: the input file should at least "
                      "contain 7 columns or no more than 9")
                return None, None, None, None, None

            pos = np.vstack((x, y, z)).T
            vel = np.vstack((vx, vy, vz)).T

            return pos, vel, mass, age, ZH, FeH, OH

        except ValueError:
            print("ERROR: could not read content of the input file")

    else:
        print("Input file not found: {0}".format(filename))
        return None, None, None, None, None

def read_rdramses_hydro(filename):
    """Reading RDRAMSES files from hydro or particles
    File is formatted with first the 3 configuration spatial coordinates
    Then the velocities, then mass, then possibly Age and Metallicity

    Parameters
    ----------
    filename: str
        Name of the input ascii file. Should contain one particle per line with
        x y z Vx Vy Vz mass age Z/H and possibly Fe/H or O/H

    Returns
    -------
    pos: array [3,N]
        Configuration cartesian positions [kpc] for N particles. 
    vel: array [3,N]
        Cartesian velocities [kpc] for N particles. 
    rho: array [N]
        density cm-3
    mass: array [N]
        Masses Msun
    T: array [N]
        Temperature in K
    mh: array [N]
        M/H
    feh: array [N]
        Fe/H
    oh: array [N]
        O/H
    """
    # Reading the input file
    if os.path.isfile(filename):
        try:
            data = np.loadtxt(filename).T
            if data.shape[0] == 10:
                print("Found 10 columns, will not read age and Z/H")
                x, y, z, vx, vy, vz, rho, level, mass, T = data
                ZH = np.zeros_like(mass)
                OH = np.zeros_like(mass)
                FeH = np.zeros_like(mass)
            elif data.shape[0] == 13:
                print("Found 13 columns")
                x, y, z, vx, vy, vz, rho, level, mass, T, ZH, FeH, OH = data
            else:
                print("ERROR: the input file should at least "
                      "contain 10 columns or no more than 13")
                return None

            pos = np.vstack((x, y, z)).T
            vel = np.vstack((vx, vy, vz)).T

            return pos, vel, rho, level, mass, T, ZH, FeH, OH

        except ValueError:
            print("ERROR: could not read content of the input file")

    else:
        print("Input file not found: {0}".format(filename))
        return None

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
    """
    # Reading the input file
    if os.path.isfile(filename):
        try:
            data = np.loadtxt(filename).T
            print("Found {} columns in file".format(data.shape[0]))
            x, y, z, vx, vy, vz = data[:6]
            rdata = data[6:]

            pos = np.vstack((x, y, z)).T
            vel = np.vstack((vx, vy, vz)).T

            return pos, vel, rdata

        except ValueError:
            print("ERROR: could not read content of the input file")

    else:
        print("Input file not found: {0}".format(filename))

    return None, None, None

def spread_amr(list_coord, level, list_val=[], list_const=[], nsample=10,
               box=10, stretch=[]) :
    """Function to reproduce a set of x, y, z and data
    n times to spread within a pixel

    Args:
        list_coord:
        list_val:
        list_const:
        level:
        nsample:
        box:

    Returns:
        list of coord, valt, val

    """
    scales = box / 2.0**level
    scalesr = np.repeat(scales, nsample)
    sizex = len(list_coord[0])
    if stretch == []:
        stretch = [1.] * len(list_coord)

    # Offseting the coordinates
    list_cr = []
    for cx, s in zip(list_coord, stretch):
        rx = (np.random.random_sample((nsample, sizex)).ravel() - 0.5) \
             * s * scalesr
        list_cr.append(np.repeat(cx, nsample) + rx)

    # Repeating the values themselves and spread
    list_valr = []
    for val in list_val:
        list_valr.append(np.repeat(val / nsample, nsample))

    # Repeating the values themselves
    list_constr = []
    for const in list_const:
        list_constr.append(np.repeat(const, nsample))

    return list_cr, list_valr, list_constr

def gas_to_cube(x, y, z, mass, level, rangein=None, debug=-1, data=None, box=100.):

    opdata = (data != None)

    # Levels and list of these
    list_levels = np.unique(level)
    Nlevels = len(list_levels)
    max_level = max(list_levels)

    # Scales
    scales = box / 2.0**level
    scale_min = min(scales)
    scale_max = max(scales)
    print("Levels are: ", list_levels)

    indminx = np.argmin(x - scales / 2.)
    indminy = np.argmin(y - scales / 2.)
    indminz = np.argmin(z - scales / 2.)
    indmaxx = np.argmax(x + scales / 2.)
    indmaxy = np.argmax(y + scales / 2.)
    indmaxz = np.argmax(z + scales / 2.)
    # Scales in kpc
    scale_minx = scales[indminx]
    scale_miny = scales[indminy]
    scale_minz = scales[indminz]
    scale_maxx = scales[indmaxx]
    scale_maxy = scales[indmaxy]
    scale_maxz = scales[indmaxz]

    # Minimum start for the grid
    sc = scale_max
    if rangein == None :
       rangein = [x[indminx] - scale_minx / 2., x[indmaxx] + scale_maxx /2.,
                  y[indminy] - scale_miny / 2., y[indmaxy] + scale_maxy /2.,
                  z[indminz] - scale_minz / 2., z[indmaxz] + scale_maxz /2.]

    # Get an integer number of pixels with the largest scale
    minmaxXYZ = ((np.asarray(rangein) + np.array([0., sc, 0., sc, 0., sc])) // scale_max) * scale_max

    # Add the 1/2 pixel size to get the real boundaries
    minx, maxx = minmaxXYZ[0], minmaxXYZ[1]
    miny, maxy = minmaxXYZ[2], minmaxXYZ[3]
    minz, maxz = minmaxXYZ[4], minmaxXYZ[5]
    # We have no central pixel
    # sc2 = sc / 2.
    # newrange = minmaxXYZ + np.array([sc2, -sc2, sc2, -sc2, sc2, -sc2])
    # Lengths of the largest x,y,z
    Dx = maxx - minx
    Dy = maxy - miny
    Dz = maxz - minz
    # Get the real number of pixels as divided by the minimum scale
    Nx = np.int(Dx / scale_min + 0.5)
    Ny = np.int(Dy / scale_min + 0.5)
    Nz = np.int(Dz / scale_min + 0.5)

    # Get the newrange
    newrange = [minx + scale_min / 2., maxx - scale_min / 2., miny + scale_min / 2., maxy - scale_min / 2., minz + scale_min / 2., maxz - scale_min / 2.]
    # Select the right particles
    sel = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy) & (z >= minz) & (z < maxz)

    print("Range will be (X,Y,Z) : ", newrange)
    print("Number of Pixels: ", Nx, Ny, Nz)

    cube_Mass = np.zeros((Nx,Ny,Nz,Nlevels), dtype=np.float32)
    if opdata:
        cube_Data = np.zeros((Nx,Ny,Nz,Nlevels), dtype=np.float32)
#    lim = [Nx * scale_min / 2., Ny * scale_min / 2., Nz * scale_min / 2.]
#    X, Y, Z = numpy.mgrid[rangein[0]:rangein[1]:1j*Nx, rangein[2]:rangein[3]:1j*Ny, rangein[4]:rangein[5]:1j*Nz]

    for i, l, in enumerate(list_levels) :
        selectL = (level == l) * (sel)
        scale = box / 2.0**l
        xs = x[selectL]
        ys = y[selectL]
        zs = z[selectL]
        ms = mass[selectL]
        if opdata :
            datas = data[selectL]
        mx, Mx = min(xs) - scale / 2., max(xs) + scale / 2.
        my, My = min(ys) - scale / 2., max(ys) + scale / 2.
        mz, Mz = min(zs) - scale / 2., max(zs) + scale / 2.
        Dx = Mx - mx
        Dy = My - my
        Dz = Mz - mz
        Nx = np.int(Dx / scale + 0.5)
        Ny = np.int(Dy / scale + 0.5)
        Nz = np.int(Dz / scale + 0.5)
        binX = np.linspace(mx, Mx, Nx+1)
        binY = np.linspace(my, My, Ny+1)
        binZ = np.linspace(mz, Mz, Nz+1)
        nx = np.size(xs)
        cube = np.zeros((nx,3), dtype=np.float32)
        cube[:,0] = xs
        cube[:,1] = ys
        cube[:,2] = zs
        hist3 = np.histogramdd(cube, bins=[binX,binY,binZ], weights=ms)
        if opdata :
            hist3D = np.histogramdd(cube, bins=[binX,binY,binZ], weights=datas)
        if l < max_level :
            # histzoom = scipy.ndimage.interpolation.zoom(hist3[0], 2**(max_level - l),
            #                                mode='constant', prefilter=True, order=0)
            histzoom = resample(hist3[0], 2**(max_level - l))
            if opdata :
                histzoomD = resample(hist3D[0], 2**(max_level - l))
        else :
            histzoom = hist3[0]
            if opdata :
                histzoomD = hist3D[0]
        if debug == l : return hist3[0], histzoom
        # Deriving the offset and applying it
        ox = mx - minx
        oy = my - miny
        oz = mz - minz
        ix = np.int(ox / scale_min + 0.5)
        iy = np.int(oy / scale_min + 0.5)
        iz = np.int(oz / scale_min + 0.5)
        sh = histzoom.shape
        cube_Mass[ix:ix+sh[0],iy:iy+sh[1],iz:iz+sh[2],i] = histzoom / 8**(max_level - l)
        if opdata :
            cube_Data[ix:ix+sh[0],iy:iy+sh[1],iz:iz+sh[2],i] = histzoomD

    X, Y, Z = numpy.mgrid[newrange[0]:newrange[1]:1j*Nx, newrange[2]:newrange[3]:1j*Ny, newrange[4]:newrange[5]:1j*Nz]
    if opdata :
        return X, Y, Z, cube_Mass, cube_Data, list_levels
    else :
        return X, Y, Z, cube_Mass, 0., list_levels



# General Functions
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

def rebin_1darray(a, shape, function='sum'):
    """Rebin an array into a new shape by making the sum or mean
    """
    sh = shape,a.shape[0]//shape
    if function == 'mean':
        return a.reshape(sh).mean(-1)
    elif function == 'sum':
        return a.reshape(sh).sum(-1)
    else:
        print("WARNING: doing the sum as input function {} "
              " not recognised".format(function))
        return a.reshape(sh).sum(-1)
    
def rebin_2darray(a, shape, function='sum'):
    """Rebin an array into a new shape by making the sum or mean
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    if function == 'mean':
        return a.reshape(sh).mean(-1).mean(1)
    elif function == 'sum':
        return a.reshape(sh).sum(-1).sum(1)
    else:
        print("WARNING: doing the sum as input function {} "
              " not recognised".format(function))
        return a.reshape(sh).sum(-1).sum(1)
