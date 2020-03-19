# -*- coding: utf-8 -*-
"""
This module includes the new LOSVD class
"""
import numpy as np
from .fit_losvd import fitgh

class LOSVD(object) :
    """ Provide a class for LOSVDs
    which includes the coordinates (X, Y, V)
    """
    def __init__(self, binx=None, biny=None, binv=None,
                 losvd=None, err_losvd=None) :
        """Class for Line of Sight Velocity distributions
        """
        self.binx = binx
        self.biny = biny
        self.binv = binv
        self.losvd = losvd
        if err_losvd is None:
            err_losvd = np.full(self.losvd.shape[:2], None)
        self.err_losvd = err_losvd

    def fitlosvd(self, degree=4, **kwargs):
        """Fit the given LOSVDs with Gauss-Hermite functions
        """

        self.GH = np.zeros((5, self.losvd.shape[0], self.losvd.shape[1]))
        self.bestfit = np.zeros_like(self.losvd)
        self.resfit = []
        for i in range(self.losvd.shape[0]):
            for j in range(self.losvd.shape[1]):
                self.GH[:,i,j], res, self.bestfit[i,j] = fitgh(self.binv,
                                                            self.losvd[i,j],
                                                            deg_gh=degree,
                                                            err=self.err_losvd[i,j],
                                                            **kwargs)
                self.resfit.append(res)