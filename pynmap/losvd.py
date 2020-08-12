# -*- coding: utf-8 -*-
"""
This module includes the new LOSVD class
"""
import numpy as np
import copy

from .fit_losvd import fitgh
from .misc_io import rebin_2darray as rebin2d
from .misc_io import rebin_1darray as rebin1d
from .misc_functions import find_centerodd, find_centereven, moment012

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import gaussian_fwhm_to_sigma

class LOSVD(object) :
    """ Provide a class for LOSVDs
    which includes the coordinates (X, Y, V)
    """
    def __init__(self, binx=None, biny=None, binv=None,
                 losvd=None, err_losvd=None, fwhmx=0., fwhmy=0., name='losvd') :
        """Class for Line of Sight Velocity distributions
        """
        self.name = name
        self.binx = binx
        self.biny = biny
        self.binv = binv
        self.stepx = np.average(np.diff(self.binx))
        self.stepy = np.average(np.diff(self.biny))
        self.losvd = losvd
        if err_losvd is None:
            err_losvd = np.full(self.losvd.shape[:2], None)
        self.err_losvd = err_losvd
        self.fwhmx = fwhmx
        self.fwhmy = fwhmy

    @property
    def extent(self):
        return [np.min(self.binx), np.max(self.binx), np.min(self.biny), np.max(self.biny)]

    def vmoments(self, **kwargs):
        """Get velocity moments from LOSVDs
        """

        self.mom0 = np.zeros((self.losvd.shape[0], self.losvd.shape[1]))
        self.mom1 = np.zeros_like(self.mom0)
        self.mom2 = np.zeros_like(self.mom0)
        for i in range(self.losvd.shape[0]):
            for j in range(self.losvd.shape[1]):
                self.mom0[i,j], self.mom1[i,j], self.mom2[i,j] = moment012(self.binv,
                                                            self.losvd[i,j])

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

    def rebin_spatial(self, factorx=1, factory=1, function='sum'):

        newshape = (self.losvd.shape[0] // factorx, self.losvd.shape[1] // factory)
        print("Newshape will be {}".format(newshape))
        binx = rebin1d(self.binx, newshape[0], function=function)
        biny = rebin1d(self.biny, newshape[1], function=function)

        losvd = np.zeros(newshape+(self.losvd.shape[2],))
        for k in range(self.losvd.shape[2]):
            losvd[:,:,k] = rebin2d(self.losvd[:,:,k], 
                                 newshape, function=function)

        if self.err_losvd[0,0] is not None:
            err_losvd = np.zeros_like(losvd)
            for k in range(self.losvd.shape[2]):
                err_losvd[:,:,k] = rebin2d(self.err_losvd[:,:,k], 
                                         newshape, function=function)
        else:
            err_losvd = None

        relosvd = LOSVD(binx, biny, self.binv, losvd, 
                        err_losvd, self.fwhmx, self.fwhmy)

        return relosvd

    def convolve_spatial(self, fwhmx=None, fwhmy=None):

        if fwhmx is None and fwhmy is None:
            print("No convolution to be done with both FWHM as None")
            return
        elif fwhmx <= 0. and fwhmy <= 0.:
            print("No convolution to be done with FWHM <= 0")
            return

        if fwhmx > self.fwhmx:
            conv_fwhmx = np.sqrt(fwhmx**2 - self.fwhmx**2)
            reached_fwhmx = fwhmx
        else:
            print("Warning: objective FWHM_X smaller than present value")
            conv_fwhmx = 0.
            reached_fwhmx = self.fwhmx
        if fwhmy > self.fwhmy:
            conv_fwhmy = np.sqrt(fwhmy**2 - self.fwhmy**2)
            reached_fwhmy = fwhmy
        else:
            print("Warning: objective FWHM_Y smaller than present value")
            conv_fwhmy = 0.
            reached_fwhmy = self.fwhmy

        closvd = copy.deepcopy(self)
        print("LOSVD FWHM: XLOS[{0}] YLOS[{1}]".format(
                 self.fwhmx, self.fwhmy))
        print("Starting the Convolution to reach FWHM: XCLOS[{0}] YCLOS[{1}]".format(
                 reached_fwhmx, reached_fwhmy))
        print("Convolving with quadratic residuals FWHM: DX[{0}] DY[{1}]".format(
                 conv_fwhmx, conv_fwhmy))

        # Building the kernel
        kernel = Gaussian2DKernel(x_stddev=conv_fwhmx * gaussian_fwhm_to_sigma / self.stepx,
                                  y_stddev=conv_fwhmy * gaussian_fwhm_to_sigma / self.stepy)
        # Doing the convolution per image slice
        for k in range(self.losvd.shape[2]):
            closvd.losvd[:,:,k] = convolve(self.losvd[:,:,k], kernel)

        if self.err_losvd[0,0] is not None:
            for k in range(self.losvd.shape[2]):
                closvd.err_losvd[:,:,k] = convolve(self.err_losvd[:,:,k], kernel)

        closvd.fwhmx = reached_fwhmx
        closvd.fwhmy = reached_fwhmy

        # Returning the result
        return closvd

    def show_IVS(self, text_info="", cutI=None, cutV=None, cutS=None, ncontours=6, dmag=0.15, 
                 moments="gh", peak_flux=True):
        import cmocean
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec
        import numpy.ma as ma

        if moments == "gh":
            # If peak flux
            if peak_flux:
                flux = self.GH[0]
            # Else total flux including width
            else:
                flux = self.GH[0] * np.sqrt(2. * np.pi) * self.GH[2]
            vel = self.GH[1]
            disp = self.GH[2]
        elif moments == "true":
            flux = self.mom0
            vel = self.mom1
            disp = self.mom2
        else:
            print("Define which moments ['true' or 'gh']")
            return

        if len(text_info) > 0:
            text_info = f" - {text_info}"
        # Cuts for the maps
        X, Y = np.meshgrid(self.binx, self.biny)
        if cutI is None:
            selI = (flux > 0.)
            minI, maxI = find_centereven(X[selI], Y[selI], np.log10(flux[selI]))
            minI -= dmag
        else:
            minI, maxI = np.log10(cutI[0]), np.log10(cutI[1])
        if cutV is None:
            cval, minV, maxV = find_centerodd(X[selI], Y[selI], vel[selI])
            amplV = maxV - cval
            minV, maxV = -amplV, amplV
        else:
            minV, maxV = cutV[0], cutV[1]
        if cutS is None:
            minS, maxS = find_centereven(X[selI], Y[selI], disp[selI])
        else:
            minS, maxS = cutS[0], cutS[1]
        mask = (np.log10(flux.T*1.1) < minI) | (np.isnan(flux))
        mI = ma.array(np.log10(flux.T), mask=mask)
        mV = ma.array(vel.T, mask=mask)
        mS = ma.array(disp.T, mask=mask)
        fig = plt.figure(num=1, figsize=(10,4))
        ax = plt.clf()
        fig.suptitle(f'{self.name}{text_info}', fontsize=16)
        gs = GridSpec(2, 3)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[:, 1])
        ax3 = fig.add_subplot(gs[:, 2])
        im1 = ax1.imshow(mI, cmap=cmocean.cm.thermal, extent=self.extent, vmin=minI, vmax=maxI)
        im2 = ax2.imshow(mV, cmap=cmocean.cm.balance, extent=self.extent, vmin=minV, vmax=maxV)
        im3 = ax3.imshow(mS, cmap=cmocean.cm.thermal, extent=self.extent, vmin=minS, vmax=maxS)
        ax1.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='g')
        ax2.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='k')
        ax3.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='k')
        ax1.tick_params(labelsize=14)
        ax2.tick_params(labelleft=False, labelsize=14)
        ax3.tick_params(labelleft=False, labelsize=14)
        fig.colorbar(im1, ax=ax1, fraction=0.03, shrink=0.8, orientation='horizontal', ticks=[], pad=0.1)
        fig.colorbar(im2, ax=ax2, fraction=0.03, shrink=0.8, orientation='horizontal', ticks=[], pad=0.1)
        fig.colorbar(im3, ax=ax3, fraction=0.03, shrink=0.8, orientation='horizontal', ticks=[], pad=0.1)
        def settext(im, ax, label):
            l1,l2 = im.get_clim()
            ax.text(0.02,0.02, "{0:4.0f}/{1:4.0f}".format(l1, l2), verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
            ax.text(0.5, 0.02, label, fontsize=14, transform=ax.transAxes, horizontalalignment='center', bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
    
        settext(im1, ax1, "log(Flux)")
        settext(im2, ax2, "V")
        settext(im3, ax3, r"$\sigma$")
        plt.subplots_adjust(left=0.07, right=0.98, bottom=0.05, top=0.95, wspace=0.05)

    def show_maps(self, text_info="", cutI=None, cutV=None, cutS=None, cutH=None, 
                  ncontours=5, peak_flux=True, moments="gh"):
        import cmocean
        from matplotlib import pyplot as plt
        from matplotlib.gridspec import GridSpec

        if moments == "gh":
            # If peak flux
            if peak_flux:
                flux = self.GH[0]
            # Else total flux including width
            else:
                flux = self.GH[0] * np.sqrt(2. * np.pi) * self.GH[2]
            vel = self.GH[1]
            disp = self.GH[2]
            mom3 = self.GH[3]
            mom4 = self.GH[4]
        elif moments == "true":
            flux = self.mom0
            vel = self.mom1
            disp = self.mom2
            mom3 = np.zeros_like(vel)
            mom4 = np.zeros_like(vel)
        else:
            print("Define which moments ['true' or 'gh']")
            return

        if len(text_info) > 0:
            text_info = f" - {text_info}"
        # Cuts for the maps
        X, Y = np.meshgrid(self.binx, self.biny)
        if cutI is None:
            selI = (flux > 0.)
            minI, maxI = find_centereven(X[selI], Y[selI], np.log10(flux[selI]))
        else:
            minI, maxI = np.log10(cutI[0]), np.log10(cutI[1])
        if cutV is None:
            cval, minV, maxV = find_centerodd(X[selI], Y[selI], vel[selI])
            amplV = maxV - cval
            minV, maxV = -amplV, amplV
        else:
            minV, maxV = cutV[0], cutV[1]
        if cutS is None:
            minS, maxS = find_centereven(X[selI], Y[selI], disp[selI])
        else:
            minS, maxS = cutS[0], cutS[1]
        if cutH is None:
            minH, maxH = -0.15, 0.15
        else:
            minH, maxH = cutH[0], cutH[1]
        fig = plt.figure(num=1, figsize=(7,10))
        ax = plt.clf()
        fig.suptitle(f'{self.name}{text_info}', fontsize=16)
        gs = GridSpec(6, 4)
        ax1 = fig.add_subplot(gs[:2, 1:3])
        ax2 = fig.add_subplot(gs[2:4, :2])
        ax3 = fig.add_subplot(gs[2:4, 2:])
        ax4 = fig.add_subplot(gs[4:, :2])
        ax5 = fig.add_subplot(gs[4:, 2:])
        im1 = ax1.imshow(np.log10(flux.T), cmap=cmocean.cm.thermal, extent=self.extent, vmin=minI, vmax=maxI)
        im2 = ax2.imshow(vel.T, cmap=cmocean.cm.balance, extent=self.extent, vmin=minV, vmax=maxV)
        im3 = ax3.imshow(disp.T, cmap=cmocean.cm.thermal, extent=self.extent, vmin=minS, vmax=maxS)
        im4 = ax4.imshow(mom3.T, cmap=cmocean.cm.balance, extent=self.extent, vmin=minH, vmax=maxH)
        im5 = ax5.imshow(mom4.T, cmap=cmocean.cm.thermal, extent=self.extent, vmin=minH, vmax=maxH)
        ax1.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='g')
        ax2.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='k')
        ax3.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='k')
        ax4.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='k')
        ax5.contour(X, Y, np.log10(flux.T), np.linspace(minI, maxI, ncontours), colors='k')
        ax1.tick_params(labelbottom=False, labelsize=14)
        ax2.tick_params(labelbottom=False, labelsize=14)
        ax3.tick_params(labelleft=False, labelbottom=False, labelsize=14)
        ax4.tick_params(labelsize=14)
        ax5.tick_params(labelleft=False, labelsize=14)
        fig.colorbar(im5, ax=[ax5], fraction=0.03, shrink=0.8, ticks=[], location="top")
        fig.colorbar(im4, ax=[ax4], fraction=0.03, shrink=0.8, ticks=[], location="top")
#        ax2.text(0.02,0.96, "%4.0f/%4.0f"%(minV, maxV), verticalalignment='top',rotation='vertical',transform = ax2.transAxes, fontsize=10, bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
#        ax3.text(0.02,0.96, "%4.0f/%4.0f"%(minS, maxS), verticalalignment='top',rotation='vertical',transform = ax3.transAxes, fontsize=10, bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
#        ax4.text(0.02,0.96, "%4.1f/%4.1f"%(minH, maxH), verticalalignment='top',rotation='vertical',transform = ax4.transAxes, fontsize=10, bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
#        ax5.text(0.02,0.96, "%4.1f/%4.1f"%(minH, maxH), verticalalignment='top',rotation='vertical',transform = ax5.transAxes, fontsize=10, bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
        def settext(im, ax, label): 
            l1,l2 = im.get_clim()
            ax.text(0.02,0.96, "{0:4.2f}/{1:4.2f}".format(l1, l2), verticalalignment='top',rotation='vertical',transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
            ax.text(0.5, 0.02, label, fontsize=14, transform=ax.transAxes, horizontalalignment='center', bbox=dict(boxstyle="square", ec=(0,0,0), fc=(1,1,1)))
    
        settext(im1, ax1, "log(Flux)")
        settext(im2, ax2, "V")
        settext(im3, ax3, r"$\sigma$")
        settext(im4, ax4, r"$h_3$")
        settext(im5, ax5, r"$h_4$")

        plt.subplots_adjust(left=0.07, right=0.95, bottom=0.05, top=0.94, wspace=0.2)
