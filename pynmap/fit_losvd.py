# -*- coding: utf-8 -*-
"""This module is used to fit LOSVDs
Authors: Eric Emsellem
"""
import numpy as np
from .misc_functions import moment012, gausshermite
from .utils.mpfit import mpfit

# Tring to import lmfit
try :
    # If it works, can use lmfit version
    # for the non-linear least squares
    import lmfit
    from lmfit import minimize, Parameters, Minimizer
    exist_lmfit = True
except ImportError :
    exist_lmfit = False
    print("WARNING: Only mpfit is available an optimiser")
    print("WARNING: you may want to install lmfit")
    print("WARNING: The routine will therefore use mpfit")
    print("         Available from pylos.utils")


# Return the Amplitude which optimises the Chi2 for a simple linear sum
def _solve_amplitude(data, ufit, error=None):
    """ Compute the amplitude needed to normalise the 1d profile which minimises
    the Chi2, given a x array, data and an error array

    The calculation follows a simple linear optimisation using

          Ioptimal = (dn x dn / dn x fn)

          where dn is the normalised data array = data / error
          and   fn is the normalised fitted array = fit / error
    Input:

    data     -   the input data (1D array, all input arrays should have the same 1d size)
    ufit     -   the un-normalised fitting profile
    error    -   the error array - Default is None (would use constant errors then)

    Output:

    Iamp     -   The normalisation needed to minimise Chi2
    """

    if error is None: error = np.ones_like(data)
    error = np.where(error == 0., 1., error)

    dn = data / error
    fn = ufit / error
    try:
        Iamp = np.sum(dn * dn) / np.sum(dn * fn)
    except RuntimeWarning:
        Iamp = 1.
    return Iamp

def fitgh(xax, data, deg_gh=2, err=None, params=None,
        fixed=None, limitedmin=None, limitedmax=None,
        minpars=None, maxpars=None,
        verbose=True, veryverbose=False, **kwargs):

    optimiser = kwargs.pop("optimiser", "mpfit")
    if optimiser == "mpfit":
        return fitgh_mpfit(xax=xax, data=data, deg_gh=deg_gh, err=err, params=params,
                fixed=fixed, limitedmin=limitedmin, limitedmax=limitedmax,
                minpars=minpars, maxpars=maxpars,
                verbose=verbose, veryverbose=veryverbose, **kwargs)
    elif optimise == "lmfit":
        if not exist_lmfit:
            print("ERROR: lmfit not installed, so please either install it or"
                  " use mpfit as optimiser option")
            return None, None, None
        else:
            return fitgh_lmfit(xax=xax, data=data, deg_gh=deg_gh, err=err, params=params,
                    fixed=fixed, limitedmin=limitedmin, limitedmax=limitedmax,
                    minpars=minpars, maxpars=maxpars,
                    verbose=verbose, veryverbose=veryverbose, **kwargs)
    else:
        print("ERROR: optimiser must be lmfit or mpfit")
        return None, None, None


# MPFIT version of the 1D Gauss-Hermite fitting routine
def fitgh_mpfit(xax, data, deg_gh=2, err=None, params=None,
                fixed=None, limitedmin=None, limitedmax=None,
                minpars=None, maxpars=None,
                verbose=True, veryverbose=False, **kwargs):
    """ Fitting routine for a Gauss-Hermite function using mpfit

    :param xax: x axis
    :param data: count axis
    :param err: error corresponding to data
    :param deg_gh: order of the Gauss-Hermite moments to fit. Must be >=2
        If =2 (default), the programme will only fit a single Gaussian
        If > 2, it will add as many h_n as needed.
    :param params: Input fit parameters: Vgauss, Sgauss, h3, h4, h5...
        Length should correspond to deg_gh. If not, a guess will be used.
    :param fixed: Is parameter fixed?
    :param limitedmin/minpars: set lower limits on each parameter
        (default: width>0)
    :param limitedmax/maxpars: set upper limits on each parameter
    :param iprint: if > 0 and verbose, print every iprint iterations of lmfit.
        default is 50
    :param lmfit_method: method to pass on to lmfit
        ('leastsq', 'lbfgsb', 'anneal').
        Default is leastsq (most efficient for the problem)
    :param **kwargs: Will be passed to MPFIT, you can for example
        use xtol, gtol, ftol, quiet
    :param verbose: self-explanatory
    :param veryverbose: self-explanatory

    :returns   Fit parameters:
    :returns   Model:
    :returns   Fit errors:
    :returns   chi2:

    """
    # Set up some default parameters for mpfit
    if "xtol" not in list(kwargs.keys()): kwargs["xtol"] = 1.e-10
    if "gtol" not in list(kwargs.keys()): kwargs["gtol"] = 1.e-10
    if "ftol" not in list(kwargs.keys()): kwargs["ftol"] = 1.e-10
    if "quiet" not in list(kwargs.keys()): kwargs["quiet"] = True

    # If no coordinates is given, create them and use pixel as a unit
    # The x axis will be from 0 to N-1, N being the number of data points
    # which will be considered as "ordered" on a regular 1d grid
    if xax is None:
        xax = np.arange(len(data))
    else:
        # Make sure that x is an array (if given)
        if not isinstance(xax, np.ndarray):
            xax = (np.asarray(xax)).ravel()

    # Compute the moments of the distribution for later purposes
    # This provides Imax, V and Sigma from 1d moments
    momProf = moment012(xax, data)

    if isinstance(params, np.ndarray): params = params.tolist()
    if params is not None:
        if (len(params) > deg_gh):
            print("ERROR: input parameter array (params) is "
                  "larger than expected")
            print("ERROR: It is %d while it should be smaller or "
                  "equal to %s (deg_gh)", len(params), deg_gh)
            return 0., 0., 0., 0.
        elif (len(params) < deg_gh):
            if verbose:
                print("WARNING: Your input parameters do not fit the "
                      "Degre set up for the GH moments")
                print("WARNING: the given value of deg_gh (%d) will be "
                      "kept ", deg_gh)
                print("WARNING: A guess will be used for the input fitting "
                      "parameters ")

    default_params = np.zeros(deg_gh, dtype=np.float64) + 0.02
    default_params[:2] = momProf[1:]
    default_minpars = np.zeros(deg_gh, dtype=np.float64) - 0.2
    default_minpars[:2] = [np.min(xax), np.min(np.diff(xax)) / 3.]
    default_maxpars = np.zeros(deg_gh, dtype=np.float64) + 0.2
    default_maxpars[:2] = [np.max(xax), (np.max(xax) - np.min(xax)) / 3.]
    default_limitedmin = [True] * deg_gh
    default_limitedmax = [True] * deg_gh
    default_fixed = [False] * deg_gh

    # Set up the default parameters if needed
    params = _set_ghparameters(params, default_params, deg_gh)
    fixed = _set_ghparameters(fixed, default_fixed, deg_gh)
    limitedmin = _set_ghparameters(limitedmin, default_limitedmin, deg_gh)
    limitedmax = _set_ghparameters(limitedmax, default_limitedmax, deg_gh)
    minpars = _set_ghparameters(minpars, default_minpars, deg_gh)
    maxpars = _set_ghparameters(maxpars, default_maxpars, deg_gh)

    # mpfit function which returns the residual from the best fit Gauss-Hermite
    # Parameters are just V, Sigma, H3,... Hn - the amplitudes are optimised at each step
    def mpfitfun(p, fjac=None, x=None, err=None, data=None):
        GH = np.concatenate(([1.0], p))
        ufit = gausshermite(x, GH)
        GH[0] = _solve_amplitude(data, ufit, err)
        if err is None:
            return [0, _fitGH_residuals(GH, x, data)]
        else:
            return [0, _fitGH_residuals_err(GH, x, data, err)]

    # Printing routine for mpfit
    def mpfitprint(mpfitfun, p, iter, fnorm, functkw=None,
                   parinfo=None, quiet=0, dof=None):
        print("Chi2 = ", fnorm)

    # Information about the parameters
    parnames = {0: "V", 1: "S"}
    for i in range(2, deg_gh): parnames[i] = "H_%02d" % (i + 1)

    # Information about the parameters
    if veryverbose:
        print("--------------------------------------")
        print("GUESS:            ")
        print("------")
        for i in range(deg_gh):
            print(" %s :  %8.3f" % (parnames[i], params[i]))
        print("--------------------------------------")

    parinfo = [{'n': ii, 'value': params[ii], 'limits': [minpars[ii], maxpars[ii]],
                'limited': [limitedmin[ii], limitedmax[ii]], 'fixed': fixed[ii],
                'parname': parnames[ii], 'error': ii} for ii in range(len(params))]

    # Fit with mpfit of q, sigma, pa on xax, yax, and data (+err)
    fa = {'x': xax, 'data': data, 'err': err}

    if verbose:
        print("------ Starting the minimisation -------")
    result = mpfit(mpfitfun, functkw=fa, iterfunct=mpfitprint, nprint=50, parinfo=parinfo, **kwargs)
    # Recompute the best amplitudes to output the right parameters
    # And renormalising them
    GH = np.concatenate(([1.0], result.params))
    ufit = gausshermite(xax, GH)
    Iamp = _solve_amplitude(data, ufit, err)
    Ibestpar_array = np.concatenate(([Iamp], result.params))

    if result.status == 0:
        raise Exception(result.errmsg)

    if verbose:
        print("=====")
        print("FIT: ")
        print("=================================")
        print("        I         V         Sig   ")
        print("   %8.3f  %8.3f   %8.3f " % (Ibestpar_array[0], Ibestpar_array[1], Ibestpar_array[2]))
        print("=================================")
        for i in range(2, deg_gh):
            print("GH %02d: %8.4f " % (i + 1, Ibestpar_array[i + 1]))
        print("=================================")

        print("Chi2: ", result.fnorm, " Reduced Chi2: ", result.fnorm / len(data))

    return Ibestpar_array, result, gausshermite(xax, Ibestpar_array)

# LMFIT version of the Gauss Hermite fitting routine
def _extract_gh_params(params, parnames):
    """Extract the parameters from the formatted list """
    deg_gh = len(parnames)
    p = np.zeros(deg_gh, dtype=np.float64)
    for i in range(deg_gh):
        p[i] = params[parnames[i]].value
    return p

def fitgh_lmfit(xax, data, deg_gh=2, err=None, params=None,
                fixed=None, limitedmin=None,
                limitedmax=None, minpars=None, maxpars=None,
                verbose=True, veryverbose=False, **kwargs):
    """Fitting routine for a Gauss-Hermite function using lmfit

    :param xax: x axis
    :param data: count axis
    :param err: error corresponding to data
    :param deg_gh: order of the Gauss-Hermite moments to fit. Must be >=2
        If =2 (default), the programme will only fit a single Gaussian
        If > 2, it will add as many h_n as needed.
    :param params: Input fit parameters: Vgauss, Sgauss, h3, h4, h5...
        Length should correspond to deg_gh. If not, a guess will be used.
    :param fixed: Is parameter fixed?
    :param limitedmin/minpars: set lower limits on each parameter (default: width>0)
    :param limitedmax/maxpars: set upper limits on each parameter
    :param iprint: if > 0 and verbose, print every iprint iterations of lmfit. default is 50
    :param lmfit_method: method to pass on to lmfit ('leastsq', 'lbfgsb', 'anneal'). Default is leastsq (most efficient for the problem)
    :param **kwargs: Will be passed to LMFIT, you can for example use xtol, gtol, ftol, quiet
    :param verbose: self-explanatory
    :param veryverbose: self-explanatory

    :returns Fitparam: Fitted parameters
    :returns Result: Structure including the results
    :returns Fit: Fitted values

    """
    # Extracting specific arguments
    iprint = kwargs.pop("iprint", 0)
    lmfit_method = kwargs.pop("lmfit_method", 'leastsq')

    # Method check
    lmfit_methods = ['leastsq', 'lbfgsb', 'anneal']
    if lmfit_method not in lmfit_methods:
        print("ERROR: method must be one of the three following methods : ", lmfit_methods)

    # Setting up epsfcn if not forced by the user
    if "epsfcn" not in list(kwargs.keys()): kwargs["epsfcn"] = 0.01
    if "xtol" not in list(kwargs.keys()): kwargs["xtol"] = 1.e-10
    if "gtol" not in list(kwargs.keys()): kwargs["gtol"] = 1.e-10
    if "ftol" not in list(kwargs.keys()): kwargs["ftol"] = 1.e-10

    # If no coordinates is given, create them and use pixel as a unit
    # The x axis will be from 0 to N-1, N being the number of data points
    # which will be considered as "ordered" on a regular 1d grid
    if xax is None:
        xax = np.arange(len(data))
    else:
        # Make sure that x is an array (if given)
        if not isinstance(xax, np.ndarray):
            xax = (np.asarray(xax)).ravel()

    # Compute the moments of the distribution for later purposes
    # This provides Imax, V and Sigma from 1d moments
    momProf = moment012(xax, data)

    if isinstance(params, np.ndarray): params = params.tolist()
    if params is not None:
        if (len(params) > deg_gh):
            print("ERROR: input parameter array (params) is larger than expected")
            print("ERROR: It is %d while it should be smaller or equal to %s (deg_gh)", len(params), deg_gh)
            return 0., 0., 0., 0.
        elif (len(params) < deg_gh):
            if verbose:
                print("WARNING: Your input parameters do not fit the Degre set up for the GH moments")
                print("WARNING: the given value of deg_gh (%d) will be kept ", deg_gh)
                print("WARNING: A guess will be used for the input fitting parameters ")

    default_params = np.zeros(deg_gh, dtype=np.float64) + 0.02
    default_params[:2] = momProf[1:]
    default_minpars = np.zeros(deg_gh, dtype=np.float64) - 0.2
    default_minpars[:2] = [np.min(xax), np.min(np.diff(xax)) / 3.]
    default_maxpars = np.zeros(deg_gh, dtype=np.float64) + 0.2
    default_maxpars[:2] = [np.max(xax), (np.max(xax) - np.min(xax)) / 3.]
    default_limitedmin = [True] * deg_gh
    default_limitedmax = [True] * deg_gh
    default_fixed = [False] * deg_gh

    # Set up the default parameters if needed
    params = _set_ghparameters(params, default_params, deg_gh)
    fixed = _set_ghparameters(fixed, default_fixed, deg_gh)
    limitedmin = _set_ghparameters(limitedmin, default_limitedmin, deg_gh)
    limitedmax = _set_ghparameters(limitedmax, default_limitedmax, deg_gh)
    minpars = _set_ghparameters(minpars, default_minpars, deg_gh)
    maxpars = _set_ghparameters(maxpars, default_maxpars, deg_gh)

    class input_residuals():
        def __init__(self, iprint, verbose):
            self.iprint = iprint
            self.verbose = verbose
            self.aprint = 0

    # lmfit function which returns the residual from the best fit Gauss Hermite function
    # Parameters are V, S, h3, h4... - the amplitudes are optimised at each step
    #                                  as this is a linear problem then
    def opt_lmfit(pars, myinput=None, x=None, err=None, data=None):
        """ Provide the residuals for the lmfit minimiser
            in the case of a Gauss-Hermite function
        """
        p = _extract_gh_params(pars, parnames)
        GH = np.concatenate(([1.0], p))
        ufit = gausshermite(x, GH)
        GH[0] = _solve_amplitude(data, ufit, err)
        if err is None:
            res = _fitGH_residuals(GH, x, data)
        else:
            res = _fitGH_residuals_err(GH, x, data, err)
        lmfit_iprint(res, myinput)
        return res

    def lmfit_iprint(res, myinput):
        """ Printing function for the iteration in lmfit
        """
        if (myinput.iprint > 0) & (myinput.verbose):
            if myinput.aprint == myinput.iprint:
                print("Chi2 = %g" % ((res * res).sum()))
                myinput.aprint = 0
            myinput.aprint += 1

    # Information about the parameters
    parnames = {0: "V", 1: "S"}
    for i in range(2, deg_gh): parnames[i] = "H_%02d" % (i + 1)

    Lparams = Parameters()
    if veryverbose:
        print("-------------------")
        print("GUESS:     ")
        print("-------------------")
    for i in range(deg_gh):
        Lparams.add(parnames[i], value=params[i],
                    min=minpars[i], max=maxpars[i], vary=not fixed[i])
        if veryverbose:
            print("%s %02d: %8.3f" % (parnames[i], i + 1, params[i]))
    if veryverbose: print("--------------------------------------")

    # Setting up the printing option
    myinput = input_residuals(iprint, verbose)

    # Doing the minimisation with lmfit
    if verbose:
        print("------ Starting the minimisation -------")
    result = minimize(opt_lmfit, Lparams, method=lmfit_method,
                      args=(myinput, xax, err, data), **kwargs)

    # Recompute the best amplitudes to output the right parameters
    # And renormalising them
    p = _extract_gh_params(result.params, parnames)
    GH = np.concatenate(([1.0], p))
    ufit = gausshermite(xax, GH)
    Iamp = _solve_amplitude(data, ufit, err)
    Ibestpar_array = np.concatenate(([Iamp], p))

    if verbose:
        print("=====")
        print("FIT: ")
        print("=================================")
        print("        I       V         Sig   ")
        print("  %8.3f  %8.3f   %8.3f " % (Ibestpar_array[0], Ibestpar_array[1], Ibestpar_array[2]))
        print("=================================")
        for i in range(2, deg_gh):
            print("GH %02d: %8.4f " % (i + 1, Ibestpar_array[i + 1]))
        print("=================================")

        print("Chi2: ", result.chisqr, " Reduced Chi2: ", result.redchi)

    return Ibestpar_array, result, gausshermite(xax, Ibestpar_array)

def _set_ghparameters(parlist, default, deg_gh) :
    """ Set up the parameters given a default

    if the Input does not have the right size (deg_gh) it fills in the rest
    with the default

    Input is :
    ==========
    : param parlist : the input parameters you wish to set
    : param default : the default when needed
    : param deg_gh   : the right size (degree of Gauss-Hermite to fit)

    :returns : parameter list

    """
    if parlist is None : parlist = default

    # Replicate the parlist with the right size first
    newparlist = [parlist[0]] * deg_gh
    # Now computing the length of the input parlist
    lparlist = len(parlist)
    # And completing it with the right default if needed
    newparlist[:lparlist] = parlist
    newparlist[lparlist:] = default[lparlist:]

    return newparlist

# Return the difference between a model and the data WITH ERRORS
def _fitGH_residuals_err(par, x, data, err) :
    """ Residual function for Gauss Hermite fit.
    Include Errors.
    """
    return ((data.ravel() - gausshermite(x, par)) / err.ravel())

# Return the difference between a model and the data WITH ERRORS
def _fitGH_chi2_err(par, x, data, err) :
    """ Sum of squares of residuals  function
    for Gauss Hermite fit. Include Errors.
    """
    return np.sum(((data.ravel() - gausshermite(x, par)) / err.ravel())**2)

# Return the difference between a model and the data
def _fitGH_residuals(par, x, data) :
    """ Residual function for Gauss Hermite fit. No Errors.
    """
    return (data.ravel() - gausshermite(x, par))

# Return the difference between a model and the data
def _fitGH_chi2(par, x, data) :
    """ Sum of squares of residuals function for Gauss Hermite fit. No Errors.
    """
    return np.sum((data.ravel() - gausshermite(x, par))**2)
