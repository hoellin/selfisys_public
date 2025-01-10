#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Copyright (C) 2024 Tristan Hoellinger
# Distributed under the GNU General Public License v3.0 (GPLv3).
# See the LICENSE file in the root directory for details.
# SPDX-License-Identifier: GPL-3.0-or-later
# ----------------------------------------------------------------------

__author__ = "Tristan Hoellinger"
__version__ = "0.1.0"
__date__ = "2024"
__license__ = "GPLv3"

"""Visualisation utilities for the exploratory examples in SelfiSys.
"""

import numpy as np
import matplotlib.pyplot as plt
from selfisys.utils.plot_params import *

# Configure global plotting settings
setup_plotting()


def plot_power_spectrum(
    G_sim, true_P, k_s, planck_Pk, Pbins, Pbins_bnd, size, L, wd, title=None, display=True
):
    """
    Plot a power spectrum over Fourier modes, its linear interpolation
    over specified support points, and a given binning for comparison.

    Parameters
    ----------
    G_sim : pysbmy.power.FourierGrid
        Fourier grid object containing the `k_modes` attribute.
    true_P : pysbmy.power.PowerSpectrum
        Power spectrum object containing the `powerspectrum` attribute.
    k_s : array-like
        Support points in k-space.
    planck_Pk : array-like
        Power spectrum values at the support points.
    Pbins : array-like
        Centres of the Φ bins in k-space.
    Pbins_bnd : array-like
        Boundaries of the Φ bins in k-space.
    size : float
        Box size in number of grid cells.
    L : float
        Box length in Mpc/h.
    wd : str
        Working directory path for saving the figure.
    title : str, optional
        Title for the figure. Default is None.
    display : bool, optional
        Whether to display the figure. Default is True.

    Returns
    -------
    None
    """
    import os
    from selfisys.utils.logger import PrintInfo

    plt.figure(figsize=(15, 5))

    # Plot power spectrum data
    plt.plot(G_sim.k_modes, true_P.powerspectrum, label=r"$P(k)$ (over all modes)")
    plt.plot(k_s, planck_Pk, label=r"$P(k)$ (binned–linear interpolation)", linestyle="dashed")

    # Configure axes
    plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    plt.ylabel(r"$[{\rm Mpc}/h]^3$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(np.clip(k_s.min() - 2e-4, 1e-4, None), k_s.max())
    plt.ylim(1e1, 1e5)
    plt.grid(which="both", axis="y")

    # Plot vertical lines for support points and binning
    plt.vlines(k_s[:-1], ymin=1e1, ymax=1e5, colors="green", linestyles="dotted", linewidth=0.6)
    plt.axvline(
        k_s[-1],
        color="green",
        linestyle="dotted",
        linewidth=0.6,
        label=r"$\boldsymbol{\uptheta}$ support points",
    )
    plt.vlines(
        Pbins,
        ymin=1e1,
        ymax=5e2,
        colors="red",
        linestyles="dashed",
        linewidth=0.5,
        label=r"$\boldsymbol{\Phi}$ bin centres",
    )
    plt.vlines(
        Pbins_bnd,
        ymin=1e1,
        ymax=1e2 / 2,
        colors="blue",
        linestyles="dashed",
        linewidth=0.5,
        label=r"$\boldsymbol{\Phi}$ bin boundaries",
    )

    # Plot the Nyquist frequency
    nyquist_freq = np.pi * size / L
    plt.axvline(
        nyquist_freq, ymax=1 / 6.0, color="orange", linestyle="-", linewidth=2, label="Nyquist"
    )

    # Add legend, optional title, and save the figure
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    if title:
        plt.title(title)
    output_dir = os.path.join(wd, "Figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "summary.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    PrintInfo(f"Figure saved to: {output_path}")

    if display:
        plt.show()
    plt.close()


def relative_error_analysis(
    G_sim, true_P, k_s, planck_Pk, Pbins, Pbins_bnd, size, L, wd, display=True
):
    """
    Compute and plot the relative error between the interpolated and
    true power spectra.

    Parameters
    ----------
    G_sim : pysbmy.power.FourierGrid
        Fourier grid object containing the `k_modes` attribute.
    true_P : pysbmy.power.PowerSpectrum
        Power spectrum object containing the `powerspectrum` attribute.
    k_s : array-like
        Support points in k-space.
    planck_Pk : array-like
        Power spectrum values at the support points.
    Pbins : array-like
        Centres of the Φ bins in k-space.
    Pbins_bnd : array-like
        Boundaries of the Φ bins in k-space.
    size : float
        Box size in number of grid cells.
    L : float
        Box length in Mpc/h.
    wd : str
        Working directory path for saving the figure.
    display : bool, optional
        Whether to display the figure. Default is True.

    Returns
    -------
    None
    """
    import os
    from scipy.interpolate import InterpolatedUnivariateSpline
    from selfisys.utils.logger import PrintInfo

    # Interpolate the power spectrum
    spline = InterpolatedUnivariateSpline(k_s, planck_Pk, k=5)
    rec_Pk = spline(G_sim.k_modes[1:])
    true_spectrum = true_P.powerspectrum[1:]
    xx = G_sim.k_modes[1:]

    # Compute relative errors
    rel_err = (rec_Pk - true_spectrum) / true_spectrum
    indices_all = slice(None)
    indices_nyquist = np.where((xx >= k_s.min()) & (xx <= np.pi * size / L))[0]
    indices_k2e1 = np.where(xx <= 2e-1)[0]

    max_relerr = np.max(np.abs(rel_err[indices_all]))
    max_relerr_nyquist = np.max(np.abs(rel_err[indices_nyquist]))
    max_relerr_2e1 = np.max(np.abs(rel_err[indices_k2e1]))

    # Create the figure
    plt.figure(figsize=(15, 5))
    plt.plot(
        xx,
        rel_err,
        label=r"$\left(P_\textrm{interp}-P_{\mathrm{true}}\right)/P_{\mathrm{true}}$",
    )
    plt.xlabel(r"$k\,[h/\mathrm{Mpc}]$")
    plt.ylabel("Relative error")
    plt.xscale("log")
    plt.xlim(np.clip(k_s.min() - 2e-4, 1e-4, None), k_s.max())
    plt.ylim(-0.1, 0.1)
    plt.grid(which="both", axis="y")

    # Vertical lines for binning and support points
    plt.axvline(
        x=Pbins[0],
        color="red",
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\boldsymbol\Phi$ bin centres",
    )
    plt.axvline(x=Pbins[-1], color="red", linestyle="dashed", linewidth=0.5)
    for k in Pbins[1:-1]:
        plt.axvline(x=k, ymax=1 / 6.0, color="red", linestyle="dashed", linewidth=0.5)
    for k in k_s[:-1]:
        plt.axvline(x=k, color="green", linestyle="dotted", linewidth=0.6)
    plt.axvline(
        x=k_s[-1],
        color="green",
        linestyle="dotted",
        linewidth=0.6,
        label=r"$\boldsymbol\uptheta$ support points",
    )
    plt.axvline(
        x=Pbins_bnd[0],
        ymax=1 / 3.0,
        color="blue",
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\boldsymbol\Phi$ bin boundaries",
    )
    plt.axvline(x=Pbins_bnd[-1], ymax=1 / 3.0, color="blue", linestyle="dashed", linewidth=0.5)
    for k in Pbins_bnd[1:-1]:
        plt.axvline(x=k, ymax=1 / 12.0, color="blue", linestyle="dashed", linewidth=0.5)

    # Nyquist and fundamental frequencies
    plt.axvline(
        x=2 * np.pi / L,
        ymax=1 / 6.0,
        color="orange",
        linestyle="-",
        linewidth=2,
        label="Fundamental mode",
    )
    plt.axvline(
        x=np.pi * size / L,
        ymax=1 / 6.0,
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Nyquist",
    )

    # Add title, legend, and save the figure
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.title(
        "Relative error between interpolated and true Planck 2018 power spectrum\n"
        f"over the {G_sim.k_modes.size} modes of the Fourier grid (max: {max_relerr * 100:.3f}\\%)"
    )
    output_dir = os.path.join(wd, "Figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "summary_relerr.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    PrintInfo(f"Figure saved to: {output_path}")

    # Print summary of relative errors
    PrintInfo(f"Max relative error over all support points: {max_relerr * 100:.3f}%")
    PrintInfo(f"Max relative error up to 1D Nyquist frequency: {max_relerr_nyquist * 100:.3f}%")
    PrintInfo(f"Max relative error up to k = 2e-1: {max_relerr_2e1 * 100:.3f}%")

    if display:
        plt.show()
    plt.close()


def plot_comoving_distance_redshift(
    zz, cosmo, means_com, L, Lcorner, wd, colours_list=COLOUR_LIST, display=True
):
    """
    Plot comoving distance as a function of redshift, highlighting key
    scales.

    Parameters
    ----------
    zz : array-like
        Redshift range for the plot.
    cosmo : astropy.cosmology object
        Cosmology instance for calculating comoving distances.
    means_com : array-like
        Mean comoving distances of selection functions.
    L : float
        Box side length in Gpc/h.
    Lcorner : float
        Diagonal of the box (sqrt(3) * L) in Gpc/h.
    wd : str
        Working directory for saving figures.
    colours_list : list
        List of colours for selection function annotations.
    display : bool, optional
        Whether to display the figure. Default is True.
    """
    d = cosmo.comoving_distance(zz) / 1e3  # Convert to Gpc/h

    plt.figure(figsize=(12, 5.2))
    plt.plot(zz, d, label="Comoving distance")
    plt.axhline(
        L, color="black", linewidth=1, linestyle="--", label=rf"$L = {L:.2f}\textrm{{ Gpc}}/h$"
    )
    plt.axhline(
        Lcorner,
        color="orange",
        linewidth=1,
        linestyle="--",
        label=rf"$L_\textrm{{corner}} = {Lcorner:.2f}\textrm{{ Gpc}}/h$",
    )

    # Annotate key redshifts
    d_np = d.value
    z_L = zz[np.argmin(np.abs(d_np - L))]
    z_corner = zz[np.argmin(np.abs(d_np - Lcorner))]
    plt.axvline(z_L, color="black", linewidth=0.5, alpha=0.5, linestyle="-")
    plt.axvline(z_corner, color="orange", linewidth=0.5, alpha=0.5, linestyle="-")
    plt.text(z_L, 1.07 * d_np.max(), rf"$z(L) = {z_L:.2f}$", fontsize=GLOBAL_FS_TINY - 2)
    plt.text(
        z_corner,
        1.07 * d_np.max(),
        rf"$z(\sqrt{{3}}\,L) = {z_corner:.2f}$",
        fontsize=GLOBAL_FS_TINY - 2,
    )

    # Annotate the selection functions' means
    z_means = np.array([zz[np.argmin(np.abs(d_np - m))] for m in means_com])
    for i, z_mean in enumerate(z_means):
        plt.axvline(z_mean, color=colours_list[i], linestyle="--", linewidth=1)
        plt.text(
            z_mean - 0.07,
            L + 0.2,
            rf"$z(\mu_{{{i+1}}} = {means_com[i]:.2f}) = {z_mean:.2f}$",
            fontsize=GLOBAL_FS_TINY - 2,
            rotation=90,
        )

    # Add labels, legend, and save the figure
    plt.xlabel("Redshift $z$")
    plt.ylabel(r"Comoving distance [Gpc$/h$]")
    plt.grid(which="both", axis="both", linestyle="-", linewidth=0.3, color="gray", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{wd}selection_functions_z.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(f"{wd}selection_functions_z.png", bbox_inches="tight", dpi=300, transparent=True)
    if display:
        plt.show()
    plt.close()


def redshift_distance_conversion(
    zz, cosmo, means_com, L, Lcorner, xx, wd, colours_list=COLOUR_LIST, display=True
):
    """
    Plot the conversion between comoving distance and redshift; return
    the redshifts corresponding to the selection functions' means.

    Parameters
    ----------
    zz : array-like
        Redshift range for the plot.
    cosmo : astropy.cosmology object
        Cosmology instance for calculating comoving distances.
    means_com : array-like
        Mean comoving distances of selection functions.
    L : float
        Box side length in Gpc/h.
    Lcorner : float
        Diagonal of the box (sqrt(3) * L) in Gpc/h.
    xx : array-like
        Comoving distances at which to compute redshift.
    wd : str
        Working directory for saving figures.
    colours_list : list
        List of colours for selection function annotations.
    display : bool, optional
        Whether to display the figure. Default is True.

    Returns
    -------
    spline : scipy.interpolate.UnivariateSpline
        Linear interpolator to convert comoving distances to redshifts.
    """
    from scipy.interpolate import UnivariateSpline

    # Convert comoving distances to redshifts using a linear interpolation
    d_np = (cosmo.comoving_distance(zz) / 1e3).value  # Gpc/h
    spline = UnivariateSpline(d_np, zz, k=1, s=0)
    z_x = spline(xx)

    plt.figure(figsize=(12, 5))
    plt.plot(xx, z_x)

    # Annotate key scales
    plt.axvline(
        L, color="black", linewidth=1, linestyle="--", label=rf"$L = {L:.2f}\textrm{{ Gpc}}/h$"
    )
    plt.axhline(spline(L), color="black", linewidth=1, linestyle="--")
    plt.axvline(
        Lcorner,
        color="orange",
        linewidth=1,
        linestyle="--",
        label=rf"$L_\textrm{{corner}} = {Lcorner:.2f}\textrm{{ Gpc}}/h$",
    )
    plt.axhline(spline(Lcorner), color="orange", linewidth=1, linestyle="--")
    plt.text(L + 0.08, spline(L) - 0.14, rf"$z(L) = {spline(L):.2f}$", fontsize=GLOBAL_FS_TINY - 2)
    plt.text(
        Lcorner - 1.2,
        spline(Lcorner) - 0.17,
        rf"$z(\sqrt{{3}}\,L) = {spline(Lcorner):.2f}$",
        fontsize=GLOBAL_FS_TINY - 2,
    )

    # Annotate the selection functions' means
    z_means = spline(means_com)
    for i, z_mean in enumerate(z_means):
        plt.axvline(means_com[i], color=colours_list[i], linestyle="--", linewidth=1)
        plt.axhline(z_mean, color=colours_list[i], linestyle="--", linewidth=1)
        plt.text(
            L + 0.08,
            z_mean - 0.14,
            rf"$z(\mu_{{{i+1}}} = {means_com[i]:.2f}) = {z_mean:.2f}$",
            fontsize=GLOBAL_FS_TINY - 2,
        )

    # Add labels, legend, and save the figure
    plt.xlabel(r"Comoving distance [Gpc$/h$]")
    plt.ylabel("Redshift $z$")
    plt.grid(which="both", axis="both", linestyle="-", linewidth=0.3, color="gray", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{wd}redshift_distance_conversion.pdf", bbox_inches="tight", dpi=300)
    if display:
        plt.show()
    plt.close()

    return spline


def plot_selection_functions_def_in_z(
    xx_of_zs,
    res,
    res_mis,
    z_means,
    cosmo,
    L,
    stds_z,
    wd,
    display=True,
):
    """
    Plot radial lognormal (in redshift) selection functions against
    comoving distances.

    Parameters
    ----------
    xx_of_zs : array-like
        Comoving distances mapped from redshift.
    res : list of array-like
        Selection functions for the well-specified model.
    res_mis : list of array-like
        Selection functions for the mis-specified model.
    z_means : array-like
        Mean redshifts of every galaxy population.
    cosmo : object
        Cosmology object.
    L : float
        Box side length in comoving distance units.
    stds_z : array-like
        Standard deviations of redshift distributions.
    wd : str
        Working directory for saving figures.
    display : bool, optional
        Whether to display the figure. Default is True.

    Returns
    -------
    None
    """
    from matplotlib.ticker import FormatStrFormatter

    colours_list = COLOUR_LIST[: len(res)]

    plt.figure(figsize=(10, 5))

    # Plot well-specified selection functions
    for i, r in enumerate(res):
        plt.plot(xx_of_zs, r, color=colours_list[i])
    plt.plot(xx_of_zs, res[-1], color="black", alpha=0, label="Model A")

    # Plot mis-specified selection functions
    for i, r_mis in enumerate(res_mis):
        plt.plot(xx_of_zs, r_mis, linestyle="--", color=colours_list[i])
    plt.plot(xx_of_zs, res_mis[-1], linestyle="--", color="black", alpha=0, label="Model B")

    # Define x-ticks and labels
    xticks = [0, np.sqrt(3) * L]
    xtick_labels = [r"$0$", r"$\sqrt 3\,L \simeq {:.2f}$".format(np.sqrt(3) * L)]
    plt.axvline(L, color="black", linestyle="-", linewidth=1, zorder=0)

    # Annotate populations
    for i, mean in enumerate(z_means):
        std = stds_z[i]
        mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
        sig2 = np.log(1 + std**2 / mean**2)
        mode = np.exp(mu - sig2)

        dmode = cosmo.comoving_distance(mode).value / 1e3
        dmean = cosmo.comoving_distance(mean).value / 1e3

        xticks.extend([dmean])
        xtick_labels.extend([f"{dmean:.2f}"])
        plt.axvline(dmean, color=colours_list[i], linestyle="-.", linewidth=1)
        plt.axvline(dmode, color=colours_list[i], linestyle="-", linewidth=1)
        plt.axvline(
            mode,
            color=colours_list[i],
            alpha=0,
            linewidth=1,
            label=f"Population {i+1}",
        )

    # Configure axes, labels, ticks, legend
    plt.xlabel(r"$r\,[{\rm Gpc}/h]$", fontsize=GLOBAL_FS_LARGE)
    plt.ylabel(r"$R_i(r)$", fontsize=GLOBAL_FS_LARGE)
    plt.xticks(xticks, xtick_labels)
    plt.tick_params(axis="x", which="major", size=8, labelsize=GLOBAL_FS_SMALL)
    plt.tick_params(axis="y", which="major", size=8, labelsize=GLOBAL_FS_SMALL)
    plt.grid(which="both", axis="both", linestyle="-", linewidth=0.4, color="gray", alpha=0.5)

    maxs = [np.max(r) for r in res]
    yticks = [0] + maxs
    plt.yticks(yticks)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    legend = plt.legend(frameon=True, loc="upper right", fontsize=GLOBAL_FS_LARGE)
    legend.get_frame().set_edgecolor("white")
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    plt.tight_layout()
    plt.savefig(f"{wd}selection_functions_com.pdf", bbox_inches="tight", dpi=300)
    if display:
        plt.show()
    plt.close()


def plot_galaxy_field_slice(g, size, L, wd, id_obs, limits="minmax", display=True):
    """
    Plot a 2D slice of the observed field.

    Parameters
    ----------
    g : ndarray
        2D array representing the observed field slice.
    size : int
        Number of grid points along each axis.
    L : float
        Size of the simulation box (in Mpc/h).
    wd : str
        Working directory for saving output files.
    id_obs : int or str
        Identifier for the observation, used in file naming.
    limits : str, optional
        Colormap scaling method. Options: 'minmax', 'truncate', 'max'.
    display : bool, optional
        Whether to display the figure. Default is True.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import colors

    # Define colormap and set scaling limits
    GalaxyMap = create_colormap("GalaxyMap")

    if limits == "max":
        maxcol = np.max(np.abs(g))
        mincol = -maxcol
        cmap = GalaxyMap
    elif limits == "truncate":
        maxcol = np.min([np.max(-g), np.max(g)])
        mincol = -maxcol
        cmap = "PiYG"
    elif limits == "minmax":
        maxcol = np.max(g)
        mincol = np.min(g)
        cmap = GalaxyMap

    divnorm = colors.TwoSlopeNorm(vmin=mincol, vcenter=0, vmax=maxcol)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(g, norm=divnorm, cmap=cmap)
    ax.invert_yaxis()  # Place origin at bottom-left
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    ticks = [mincol, mincol / 2, 0, maxcol / 3, 2 * maxcol / 3, maxcol]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{x:.2f}" for x in ticks], size=GLOBAL_FS_SMALL)
    cbar.set_label(r"$\delta_\textrm{g}$", size=GLOBAL_FS)
    ax.set_xticks(
        [size * i / 4.0 for i in range(5)], [f"{L * 1e-3 * i / 4:.1f}" for i in range(5)]
    )
    ax.set_yticks(
        [size * i / 4.0 for i in range(5)], [f"{L * 1e-3 * i / 4:.1f}" for i in range(5)]
    )
    ax.set_xlabel(r"Gpc/$h$", size=GLOBAL_FS)
    ax.set_ylabel(r"Gpc/$h$", size=GLOBAL_FS)

    # Save or display
    if display:
        plt.show()
    else:
        plt.savefig(f"{wd}Figures/g_{id_obs}.png", bbox_inches="tight", dpi=300)
        plt.savefig(f"{wd}Figures/g_{id_obs}.pdf", bbox_inches="tight", dpi=300)
    plt.close()
