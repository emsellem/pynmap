# Licensed under a 3-clause BSD style license - see LICENSE

"""MUSE-PHANGS preparation recipe module
"""

__authors__   = "Eric Emsellem"
__copyright__ = "(c) 2019, ESO + CRAL"
__license__   = "3-clause BSD License"
__contact__   = " <eric.emsellem@eso.org>"

# Importing modules
import numpy as np

def rotateXYZ_mat(pos, vel=None, alpha=0.0, beta=0.0, gamma=0.0,
              rotorder=[0,1,2], direct=True):   # Angles in degrees
    """
    Performs a rotation around the three axes in order given by rotorder
    for the position and velocities of particles.

    Parameters
    ----------
    alpha: float
        Angle around X in degrees
    beta: float
        Angle around Y in degrees
    gamma: float
        Angle around Z in degrees
    direct: bool
        Direct rotation [True] or go back to 
        initial configuration [False]
    """
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    calpha, salpha = np.cos(alpha_rad), np.sin(alpha_rad)
    cbeta, sbeta = np.cos(beta_rad), np.sin(beta_rad)
    cgamma, sgamma = np.cos(gamma_rad), np.sin(gamma_rad)

    if direct :
        matX=np.array([[1, 0, 0], [0, calpha, salpha],
                       [0, -salpha, calpha]], dtype=np.float32)
        matY=np.array([[cbeta, 0, -sbeta], [0, 1, 0],
                       [sbeta, 0, cbeta]], dtype=np.float32)
        matZ=np.array([[cgamma, sgamma, 0],
                       [-sgamma, cgamma, 0],
                       [0, 0, 1]],dtype=np.float32)
    else :
        matX=np.transpose(np.array([[1, 0, 0],
                                    [0, calpha, salpha],
                                    [0, -salpha, calpha]],
                                    dtype=np.float32))
        matY=np.transpose(np.array([[cbeta, 0, -sbeta],
                                    [0, 1, 0],
                                    [sbeta, 0, cbeta]],
                                    dtype=np.float32))
        matZ=np.transpose(np.array([[cgamma, sgamma, 0],
                                    [-sgamma, cgamma, 0],
                                    [0, 0, 1]],
                                    dtype=np.float32))

    list_mat = [matX, matY, matZ]
    mat = np.dot(np.dot(list_mat[rotorder[0]], 
                        list_mat[rotorder[1]]), 
                        list_mat[rotorder[2]])

    pos = np.transpose(np.dot(mat,np.transpose(pos)))
    if vel is not None:
        vel = np.transpose(np.dot(mat,np.transpose(vel)))

    return pos, vel

def rotate_mat(pos, vel=None, PA=0.0, inclination=0.0, direct=True):
    """
    Performs a rotation of PA and i with angles given in degrees
    for the position and velocities of particles.

    Rotation with PA rotates around z
    And rotation with inclination rotates around X

    Parameters
    ---------
    PA: float
        Position Angle (in degrees)
    inclination : float
        Inclination (in degrees)
    direct: bool
        Type of rotation. If True, return to initial configuration
        If False do the rotation

    Returns
    ------
    pos: array
        Array of positions (x,y,z)
    vel: array
        Array of velocities (Vx, Vy, Vz)
        If None [default], ignored
    """
    PA = np.deg2rad(PA)
    inclination = np.deg2rad(inclination)
    cPA = np.cos(PA)
    sPA = np.sin(PA)
    ci = np.cos(inclination)
    si = np.sin(inclination)

    if direct :
        mat=np.array([[cPA, -sPA, 0], [ci*sPA, ci*cPA, -si],
                      [si*sPA, cPA*si, ci]], dtype=np.float32)
    else :
        mat=np.transpose(np.array([[cPA, -sPA, 0], 
                                   [ci*sPA, ci*cPA, -si],
                                   [si*sPA, cPA*si, ci]], 
                                   dtype=np.float32))

    pos = np.transpose(np.dot(mat,np.transpose(pos)))
    if vel is not None:
        vel = np.transpose(np.dot(mat,np.transpose(vel)))

    return pos, vel

def xy_to_polar(x, y, cx=0.0, cy=0.0, PA=None) :
    """
    Convert x and y coordinates into polar coordinates

    cx and cy: Center in X, and Y. 0 by default.
    PA : position angle in radians
         (Counter-clockwise from vertical)
         This allows to take into account some rotation
         and place X along the abscissa
         Default is None and would be then set for no rotation

    Return : R, theta (in radians)
    """
    if PA is None : PA = -np.pi / 2.
    # If the PA does not have X along the abscissa, rotate
    if np.mod(PA + np.pi / 2., np.pi) != 0.0 : 
        x, y = rotxyC(x, y, cx=cx, cy=cy, angle=PA + np.pi / 2.)
    else : x, y = x - cx, y - cy

    # Polar coordinates
    R = np.sqrt(x**2 + y**2)
    # Now computing the true theta
    theta = np.arctan2(y, x)
    return R, theta

def polar_to_xy(r, theta) :
    """
    Convert x and y coordinates into polar coordinates Theta in Radians
    Return :x, y
    """

    # cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def rotxC(x, y, cx=0.0, cy=0.0, angle=0.0) :
    """ Rotate by an angle (in radians) 
        the x axis with a center cx, cy

        Return rotated(x)
    """
    return (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)

def rotyC(x, y, cx=0.0, cy=0.0, angle=0.0) :
    """ Rotate by an angle (in radians) 
        the y axis with a center cx, cy

        Return rotated(y)
    """
    return (cx - x) * np.sin(angle) + (y - cy) * np.cos(angle)

def rotxyC(x, y, cx=0.0, cy=0.0, angle=0.0) :
    """ Rotate both x, y by an angle (in radians) 
        the x axis with a center cx, cy

        Return rotated(x), rotated(y)
    """
    # First centring
    xt = x - cx
    yt = y - cy
    # Then only rotation
    return rotxC(xt, yt, angle=angle), rotyC(xt, yt, angle=angle)

def az_average(data, center=None, scale=1.0):
    """
    Calculate the azimuthally averaged radial profile.
    data - The 2D data
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(data.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, 
                           (y.max()-y.min())/2.0])
        print("Center found at : {0}, {1} "
              "[pixels]".format(center[0], center[1]))

    r = np.hypot(np.abs(x - center[0]), np.abs(y - center[1]))

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = np.nan_to_num(data.flat[ind])

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar==1)[0]       # location of changed radius
    nr = np.zeros_like(rind, dtype=float)
    nr[1:] = np.diff(rind)
    nr[0] = rind[0]
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = np.zeros_like(rind, dtype=float)
    tbin[1:] = csim[rind[1:]] - csim[rind[:-1]]
    tbin[0] = csim[rind[0]]

    radial_prof = tbin / nr
    radii = np.arange(np.max(r_int)) * scale

    return radii, radial_prof

def vector_xy_to_rtheta(x, y, Vx, Vy):
    """
    Convert Vxy into VR, Vtheta

    Parameters
    ----------
    x, y:
    Vx, Vy:

    Return : R, theta, VR, Vtheta
    """
    R, theta = xy_to_polar(x, y)
    # Polar coordinates
    VR = Vx * np.cos(theta) + Vy * np.sin(theta)
    Vtheta = -Vx * np.sin(theta) + Vy * np.cos(theta)
    return R, theta, VR, Vtheta

# Set of functions to transform into Sun-view - 2021/Apr
# Adapted from Mattia Sormani's input formulae in python
def xyzsunhat(xsun, ysun, zsun):
    """Return the x, y, z hat components in the xyz frame

    Input
    -----
    xsun, ysun, zsun: x, y, z location for the Sun

    Returns
    -------
    xhat, yhat, zhat
    """
    rsun = np.sqrt(xsun**2 + ysun**2 + zsun**2)
    Rsun = np.sqrt(xsun**2 + ysun**2)
    sintheta = Rsun / rsun
    costheta = zsun / rsun
    sinphi   = ysun / Rsun
    cosphi   = xsun / Rsun 
    xhat = np.array([-cosphi * sintheta, 
                     -sinphi * sintheta, -costheta])
    yhat = np.array([+sinphi, -cosphi, +0.0])
    zhat = np.array([-cosphi * costheta, 
                     -sinphi * costheta, +sintheta])
    return xhat, yhat, zhat

def rblhat(l, b, r):
    """Return r, b and l hat components in the xyz frame

    Input
    -----
    l, b, r: input galactic coordinates

    Returns
    -------
    rhat, bhat, lhat    
    """
    theta = np.pi / 2. - b
    rhat  = [+np.sin(theta) * np.cos(l), 
             +np.sin(theta) * np.sin(l), 
             + np.cos(theta)]
    bhat  = [-np.cos(theta) * np.cos(l), 
            -np.cos(theta) * np.sin(l), 
            + np.sin(theta)] 
    lhat  = [-np.sin(l), np.cos(l), 0.]
    return rhat, bhat, lhat

def xyz2xyzsun(x, y, z, vx, vy, vz,
            xsun=-8.0, ysun=0.0, zsun=0.0,
            vxsun=0.0, vysun=2.2, vzsun=0.0):
    xhat, yhat, zhat = xyzsunhat(xsun, ysun, zsun)
    deltaxyz = np.array([x - xsun, y - ysun, z - zsun])
    deltavxyz = np.array([vx - vxsun, vy - vysun, vz - vzsun])
    xs, ys, zs = np.array([xhat, yhat, zhat]) @ deltaxyz
    vxs, vys, vzs = np.array([xhat, yhat, zhat]) @ deltavxyz
    return xs, ys, zs, vxs, vys, vzs

def xyzsun2xyz(xs, ys, zs, vxs, vys, vzs, 
               xsun=-8.0, ysun=0.0, zsun=0.0,
               vxsun=0.0, vysun=2.2, vzsun=0.0):
    xhat, yhat, zhat = xyzsunhat(xsun, ysun, zsun)
    deltaxyz = np.array([xs, ys, zs]) @ np.array([xhat, yhat, zhat])
    deltavxyz = np.array([vxs, vys, vzs]) @ np.array([xhat, yhat, zhat])
    x, y, z  = deltaxyz + np.array([xsun, ysun, zsun])
    vx, vy, vz  = deltavxyz + np.array([vxsun, vysun, vzsun])
    return x, y, z, vx, vy, vz

def xyzsun2lbr(xs, ys, zs, vxs, vys, vzs):
    r     = np.sqrt(xs**2 + ys**2 + zs**2)
    l     = np.arctan2(ys, xs)
    theta = np.arccos(zs / r)
    b     = np.pi / 2. - theta
    rhat, bhat, lhat = rblhat(l, b, r)
    vr, vb, vl = np.array([rhat, bhat, lhat]) @ np.array([vxs, vys, vzs])
    return l, b, r, vl, vb, vr

def lbr2xyzsun(l, b, r, vl, vb, vr):
    theta = np.pi / 2. - b
    rhat, bhat, lhat = rblhat(l, b, r)
    xs  = r * np.sin(theta) * np.cos(l)
    ys  = r * np.sin(theta) * np.sin(l)
    zs  = r * np.cos(theta)
    vxs, vys, vzs = np.array([vr, vl, vb]) @ np.array([rhat, lhat, bhat])
    return xs, ys, zs, vxs, vys, vzs

def xyz2lbr(x, y, z, vx, vy, vz,
            xsun=-8.0, ysun=0.0, zsun=0.0,
            vxsun=0.0, vysun=2.2, vzsun=0.0):
    xs, ys, zs, vxs, vys, vzs = xyz2xyzsun(x, y, z, vx, vy, vz,
                                xsun,ysun,zsun,
                                vxsun,vysun,vzsun)
    l, b, r, vl, vb, vr = xyzsun2lbr(xs, ys, zs, vxs, vys, vzs)
    return l, b, r, vl, vb, vr

def lbr2xyz(l, b, r, vl, vb, vr, 
            xsun=-8.0, ysun=0.0, zsun=0.0,
            vxsun=0.0, vysun=2.2, vzsun=0.0):
    xs, ys, zs, vxs, vys, vzs = lbr2xyzsun(l, b, r, vl, vb, vr)
    x, y, z, vx, vy, vz = xyzsun2xyz(xs, ys, zs, vxs, vys, vzs, 
                                     xsun, ysun, zsun, 
                                     vxsun, vysun, vzsun)
    return x, y, z, vx, vy, vz
