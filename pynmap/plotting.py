# -*- coding: utf-8 -*-
"""Plotting module for certain dedicated functions (dispersions, etc)
"""

def plot_sigma(Rbin, v, s, Rmin=0., Rmax=None, Vmax=None,
               snapname="Snapshot", suffix="",
               figure_folder="Figures/", save=False) :
    """Plot the 3 sigma profiles (cylindrical coordinates)
    """
    from matplotlib import pyplot as plt

    if Vmax is None:
        Vmax = np.maximum(np.max(v), np.max(s)) * 1.1

    plt.clf()
    plt.plot(Rbin, s[0], 'r-', label=r'$\sigma_R$')
    plt.plot(Rbin, s[1], 'b-', label=r'$\sigma_{\theta}$')
    plt.plot(Rbin, s[2], 'k-', label=r'$\sigma_z$')
    plt.plot(Rbin, v[0], 'r--', label=r'$V_R$')
    plt.plot(Rbin, v[1], 'b--', label=r'$V_{\theta}$')
    plt.plot(Rbin, v[2], 'k--', label=r'$V_z$')
    plt.legend()
    plt.xlabel("R [kpc]", fontsize=20)
    plt.ylabel("$\sigma$ [km/s]", fontsize=20)
    plt.xlim(Rmin, Rmax)
    plt.ylim(0, Vmax)
    plt.title("Model = %s - %s"%(snapname, suffix))
    plt.tight_layout()
    if save:
        plt.savefig(figure_folder+"Fig_%s_%s.png"%(snapname, suffix))
