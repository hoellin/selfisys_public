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

"""Plotting routines for the SelfiSys project.
"""

import gc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from selfisys.utils.plot_params import *
from selfisys.utils.logger import getCustomLogger

logger = getCustomLogger(__name__)

# Configure global plotting settings
setup_plotting()


def plot_selection_functions(
    x,
    res,
    res_mis,
    params,
    L,
    corner,
    axis="com",
    zz=None,
    zcorner=None,
    z_L=None,
    path=None,
    force_plot=False,
    labsAB=True,
):
    """
    Plot selection functions.

    Parameters
    ----------
    x : array-like
        x-axis values (e.g., comoving distances or redshifts).
    res : list of array-like
        Selection functions for Model A.
    res_mis : list of array-like, optional
        Selection functions for Model B (optional).
    params : tuple of (array-like, array-like, array-like), optional
        Standard deviations, means, and normalisation factors for the
        multiple galaxy populations.
    L : float
        Box size.
    corner : float
        Diagonal box size.
    axis : str, optional
        x-axis type ('com' for comoving distance, 'redshift' for
        redshift).
    zz : array-like, optional
        Mapping between comoving distances and redshifts.
    zcorner : float, optional
        Redshift corresponding to the diagonal box size.
    z_L : float, optional
        Redshift corresponding to the box side length.
    path : str, optional
        Path to save the output plot.
    force_plot : bool, optional
        If True, displays the plot even if a path is specified.
    labsAB : bool, optional
        If True, labels models as 'Model A' and 'Model B'.

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.
    """
    logger.info("Plotting selection functions...")

    try:
        colours_list = COLOUR_LIST[: len(res)]
        plt.figure(figsize=(10, 5))

        # Plot rescaled selection functions for Model A
        for i, r in enumerate(res):
            plt.plot(x, r, color=colours_list[i])  # , label=None)

        # Plot rescaled selection functions for Model B, if provided
        if res_mis is not None:
            label_a = "Model A" if labsAB else None
            plt.plot(x, res[-1], color="black", alpha=0, label=label_a)
            for i, r_mis in enumerate(res_mis):
                plt.plot(x, r_mis, linestyle="--", color=colours_list[i])
            label_b = "Model B" if labsAB else None
            plt.plot(x, res_mis[-1], linestyle="--", color="black", alpha=0, label=label_b)

        # Configure x-axis ticks and labels
        xticks = [0, corner]
        xtick_labels = [r"$0$"]
        if axis == "com" and corner is not None:
            xtick_labels.append(r"$\sqrt 3\,L \simeq {:.2f}$".format(corner))
        elif axis == "redshift" and corner is not None:
            xtick_labels.append(r"$z(\sqrt 3\,L) \simeq {:.3f}$".format(corner))
        if z_L is not None and L is not None:
            xticks.insert(1, L)
            xtick_labels.insert(1, r"$L=3.6$")

        # Annotate populations
        for i, mean in enumerate(params[1]):
            mean_plt = x[np.argmin(np.abs(zz - mean))] if zz is not None else mean
            plt.axvline(mean_plt, color=colours_list[i], linestyle="--", linewidth=1.5)
            lab_pop = f"Population {i + 1}"
            plt.axvline(mean_plt, color=colours_list[i], alpha=0, linewidth=2, label=lab_pop)
            xticks.append(mean_plt)
            xtick_labels.append(r"${:.2f}$".format(mean_plt))

        # Set axis labels
        xlabel = r"$r\,[{\rm Gpc}/h]$" if axis == "com" else r"$z$"
        ylabel = r"$R_i(r)$" if zcorner is None else r"$R_i$"
        plt.xlabel(xlabel, fontsize=GLOBAL_FS_LARGE)
        plt.ylabel(ylabel, fontsize=GLOBAL_FS_LARGE)

        # Configure ticks and grid
        plt.xticks(xticks, xtick_labels)
        plt.tick_params(axis="x", which="major", size=8, labelsize=GLOBAL_FS_SMALL)
        plt.tick_params(axis="y", which="major", size=8, labelsize=GLOBAL_FS_SMALL)
        plt.grid(which="both", axis="both", linestyle="-", linewidth=0.4, color="gray", alpha=0.5)

        # Add legend, save and display plot
        fs_legend = GLOBAL_FS_LARGE
        loc_legend = (0.6, 0.35)
        legend = plt.legend(frameon=True, loc=loc_legend, fontsize=fs_legend)
        for handle in legend.legend_handles:
            handle.set_alpha(1)

        # Handle dual x-axes for redshift and comoving distance
        if zcorner is not None:
            ax2 = plt.gca().twiny()
            ax2.set_xlabel(r"$z$", fontsize=GLOBAL_FS_LARGE)
            zticks = (
                np.concatenate([[0], [z_L], [zcorner], params[1]])
                if z_L
                else np.concatenate([[0], [zcorner], params[1]])
            )
            ax2.set_xticks(xticks)
            ax2.set_xticklabels([r"${:.2f}$".format(z) for z in zticks])
            ax2.tick_params(axis="x", which="major", size=8, labelsize=GLOBAL_FS_SMALL)
            ax2.grid(
                which="both", axis="both", linestyle="-", linewidth=0.4, color="gray", alpha=0.5
            )

        plt.tight_layout()

        if path:
            plt.savefig(path, bbox_inches="tight", dpi=300)
            plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", dpi=300)
            logger.info("Figure saved to %s", path)
        if not path or force_plot:
            plt.show()
        plt.close()

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e


def plotly_3d(field, size=128, L=None, colormap="RdYlBu", limits="max"):
    """
    Create an interactive 3D plot of volume slices using Plotly.

    Parameters
    ----------
    field : array-like
        3D data field to visualise.
    size : int, optional
        Size of the field along one dimension. Default is 128.
    L : float, optional
        Physical size of the field in Mpc/h. Used for axis labels only.
    colormap : str, optional
        Colour map for visualisation. Default is 'RdYlBu'.
    limits : str, optional
        Colour scale limits ('max', 'truncate', or 'default'). Default
        is 'max'.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    import plotly.graph_objects as go

    volume = field.T
    rows, cols = volume[0].shape

    # Define colour scale limits
    if limits == "max":
        maxcol = np.max(np.abs(volume))
        mincol = -maxcol
    elif limits == "truncate":
        maxcol = min(np.max(-volume), np.max(volume))
        mincol = -maxcol
    else:
        maxcol = np.max(volume)
        mincol = np.min(volume)
    midcol = np.mean(volume)

    # Generate frames for the animation
    nb_frames = size
    frames = [
        go.Frame(
            data=go.Surface(
                z=(size - k) * np.ones((rows, cols)),
                surfacecolor=np.flipud(volume[cols - 1 - k]),
                cmin=mincol,
                cmid=midcol,
                cmax=maxcol,
            ),
            name=str(k),  # Frames must be named for proper animation
        )
        for k in range(nb_frames)
    ]

    # Initial plot configuration
    fig = go.Figure(
        frames=frames,
        data=go.Surface(
            z=size * np.ones((rows, cols)),
            surfacecolor=np.flipud(volume[cols // 2]),
            colorscale=colormap,
            cmin=mincol,
            cmid=midcol,
            cmax=maxcol,
            colorbar=dict(thickness=20, ticklen=4),
        ),
    )

    def frame_args(duration):
        """Helper function to set animation frame arguments."""
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    # Add animation slider
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Configure layout with or without physical size
    layout_config = dict(
        title="Slices in density field",
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(range=[0, size - 1], autorange=False),
            xaxis_title="x [Mpc/h]",
            yaxis_title="y [Mpc/h]",
            zaxis_title="z [Mpc/h]",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {"args": [None, frame_args(50)], "label": "&#9654;", "method": "animate"},
                    {"args": [[None], frame_args(0)], "label": "&#9724;", "method": "animate"},
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )
    if L is not None:
        layout_config["scene"]["xaxis"] = dict(
            ticktext=[0, L / 2, L],
            tickvals=[0, size / 2, size],
            title="x [Mpc/h]",
        )
        layout_config["scene"]["yaxis"] = dict(
            ticktext=[0, L / 2, L],
            tickvals=[0, size / 2, size],
            title="y [Mpc/h]",
        )
        layout_config["scene"]["zaxis"]["ticktext"] = [0, L / 2, L]
        layout_config["scene"]["zaxis"]["tickvals"] = [0, size / 2, size]

    fig.update_layout(**layout_config)
    return fig


def plot_observations(
    k_s,
    theta_gt,
    planck_Pk_EH,
    P_0,
    Pbins,
    phi_obs,
    Npop,
    path=None,
    force_plot=False,
):
    """
    Plot the observed power spectra and related quantities.

    Parameters
    ----------
    k_s : ndarray
        Array of wavenumbers.
    theta_gt : ndarray
        Ground truth theta values.
    planck_Pk_EH : ndarray
        Planck power spectrum values.
    P_0 : float
        Normalisation constant for power spectra.
    Pbins : ndarray
        Vector of bin boundaries for the summary statistics.
    phi_obs : ndarray
        Observed summaries.
    Npop : int
        Number of populations.
    path : str, optional
        Path to save the output plot.
    force_plot : bool, optional
        If True, displays the plot even if a path is specified.

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.

    """
    logger.info("Plotting observations...")

    # Sanity checks
    if len(k_s) == 0 or len(theta_gt) == 0 or len(planck_Pk_EH) == 0:
        logger.warning("One or more input arrays are empty. The plot may be incomplete.")

    if len(k_s) != len(theta_gt) or len(k_s) != len(planck_Pk_EH):
        logger.error("Mismatch in array lengths. Plotting may not reflect all data.")

    try:
        _, ax1 = plt.subplots(figsize=(12, 5))

        # Plot theta values
        ax1.plot(k_s, theta_gt / P_0, label=r"$\boldsymbol{\uptheta}_{\mathrm{gt}}$", color="C0")
        ax1.set_xscale("log")
        ax1.semilogx(
            k_s,
            planck_Pk_EH / P_0,
            label=r"$P_{\mathrm{Planck}}(k)/P_0(k)$",
            color="C1",
            lw=0.5,
        )

        ax1.set_xlabel(r"$k$ [$h$/Mpc]")
        ax1.set_ylabel(r"$[{\mathrm{Mpc}}/h]^3$")
        ax1.grid(which="both", axis="y", linestyle="dotted", linewidth=0.6)

        # Vertical lines for theta support wavenumbers
        for k in k_s[:-1]:
            ax1.axvline(x=k, color="green", linestyle="dotted", linewidth=0.6)
        ax1.axvline(
            x=k_s[-1],
            color="green",
            linestyle="dotted",
            linewidth=0.6,
            label=r"$\boldsymbol{\uptheta}$ support wavenumbers",
        )

        # Vertical lines for Phi bin centres
        ax2 = ax1.twinx()
        ax1.axvline(x=Pbins[0], color="red", linestyle="dashed", linewidth=0.5)
        ax2.axvline(
            x=Pbins[-1],
            color="red",
            linestyle="dashed",
            linewidth=0.5,
            label=r"$\boldsymbol{\Phi}$-bins centres",
        )
        for k in Pbins[1:-1]:
            ax1.axvline(x=k, ymax=0.167, color="red", linestyle="dashed", linewidth=0.5)

        ax1.legend(loc="lower left", fontsize=GLOBAL_FS)
        ax1.set_xlim(max(1e-4, k_s.min() - 2e-4), k_s.max())
        ax1.set_ylim(7e-1, 1.6e0)

        # Plot observations
        len_obs = len(phi_obs) // Npop

        if len(phi_obs) % Npop != 0:
            logger.warning(
                "Length of 'phi_obs' is not divisible by the number of populations. "
                "Ensure input dimensions are consistent.",
            )

        for i in range(Npop):
            ax2.plot(
                Pbins,
                phi_obs[i * len_obs : (i + 1) * len_obs],
                marker="x",
                label=r"$\boldsymbol{\Phi}_{\mathrm{obs}}$, population " + str(i),
                linewidth=0.5,
                color=COLOUR_LIST[i],
            )

        ax2.legend(loc="upper right", fontsize=GLOBAL_FS)

        if path:
            plt.savefig(path, bbox_inches="tight", dpi=300)
            plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight", dpi=300)
            logger.info("Figure saved to %s", path)
        if not path or force_plot:
            plt.show()
        plt.close()

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e


def plot_reconstruction(
    k_s,
    Pbins,
    prior_theta_mean,
    prior_theta_covariance,
    posterior_theta_mean,
    posterior_theta_covariance,
    theta_gt,
    P_0,
    phi_obs=None,
    suptitle=None,
    theta_fid=None,
    savepath=None,
    force_plot=False,
    legend_loc="upper right",
    enforce_ylims=True,
):
    """
    Plot the prior, posterior and ground truth power spectra.

    Parameters
    ----------
    k_s : ndarray
        Array of wavenumbers.
    Pbins : ndarray
        Vector of bin boundaries for the summary statistics.
    prior_theta_mean : ndarray
        Mean of the prior distribution.
    prior_theta_covariance : ndarray
        Covariance of the prior distribution.
    posterior_theta_mean : ndarray
        Mean of the posterior distribution.
    posterior_theta_covariance : ndarray
        Covariance of the posterior distribution.
    theta_gt : ndarray
        Ground truth power spectrum.
    P_0 : float
        Normalisation constant for the power spectrum.
    phi_obs : ndarray, optional
        Observed summaries.
    suptitle : str, optional
        Plot title. Leave empty for no title.
    theta_fid : ndarray, optional
        Fiducial theta for hyperparameter tuning (for some priors).
    savepath : str or Path, optional
        Path to save the plot.
    force_plot : bool, optional
        If True, displays the plot even if a path is specified.
    legend_loc : str, optional
        Location of the plot legend.
    enforce_ylims : bool, optional
        Enforce y-axis limits.

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.
    """
    logger.info("Generating power spectrum reconstruction plot.")

    try:
        fig, ax = plt.subplots(figsize=(14, 5))

        # Prior
        ax.plot(
            k_s,
            prior_theta_mean,
            linestyle="-",
            color="gold",
            label="$\\boldsymbol{\\uptheta}_0$ (prior)",
        )
        ax.fill_between(
            k_s,
            prior_theta_mean - 2 * np.sqrt(np.diag(prior_theta_covariance)),
            prior_theta_mean + 2 * np.sqrt(np.diag(prior_theta_covariance)),
            color="gold",
            alpha=0.2,
        )

        # Fiducial theta used for hyperparameter tuning with some priors
        if theta_fid is not None:
            ax.plot(
                k_s,
                theta_fid,
                linestyle="--",
                color="C1",
                label="$\\boldsymbol{\\uptheta}_{\\mathrm{fid}}$ (fiducial)",
            )

        # Posterior
        ax.plot(
            k_s,
            posterior_theta_mean,
            color="C2",
            label="$\\boldsymbol{\\upgamma}$ (reconstruction)",
        )
        ax.fill_between(
            k_s,
            posterior_theta_mean - 2 * np.sqrt(np.diag(posterior_theta_covariance)),
            posterior_theta_mean + 2 * np.sqrt(np.diag(posterior_theta_covariance)),
            color="C2",
            alpha=0.35,
        )

        # Ground truth
        ax.plot(
            k_s,
            theta_gt / P_0,
            color="C0",
            label="$\\boldsymbol{\\uptheta}_\\mathrm{gt}$ (groundtruth)",
        )

        # Plot the binning
        ymin, ymax = ax.get_ylim()
        for i in range(len(k_s)):
            ax.axvline(
                k_s[i],
                ymin=ymin,
                ymax=ymax,
                linestyle=":",
                linewidth=0.8,
                color="green",
                alpha=0.5,
            )
        ax.vlines(
            Pbins, ymin=ymin, ymax=0.8, linestyle="--", linewidth=0.8, color="red", alpha=0.5
        )

        # Overlay observations if provided
        if phi_obs is not None:
            ax2 = ax.twinx()
            ax2.plot(Pbins, phi_obs, "C3.-", label=r"$\Phi_{O}$ (observations)")
            ax2.legend(loc="lower right", fontsize=GLOBAL_FS_LARGE)

        ax.set_xlim([k_s.min() - 0.0001, k_s.max()])
        ax.set_xscale("log")
        ax.grid(visible=True, which="both", linestyle=":", color="grey")
        ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax.xaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax.xaxis.set_tick_params(which="minor", length=4)
        ax.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax.yaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        if enforce_ylims:
            ax.set_ylim([0.85, 1.35])
        ax.set_xlabel("$k$ [$h$/Mpc]", size=GLOBAL_FS_LARGE)
        ax.set_ylabel("$\\theta(k) = P(k)/P_0(k)$", size=GLOBAL_FS_LARGE)
        ax.legend(loc=legend_loc, fontsize=GLOBAL_FS_LARGE)

        plt.suptitle(suptitle, fontsize=GLOBAL_FS_XLARGE) if suptitle else None

        # Save / display
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=300, format="png", transparent=True)
            fig.savefig(
                savepath[:-4] + "_white.png",
                bbox_inches="tight",
                dpi=300,
                format="png",
                transparent=False,
            )
            fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
            fig.suptitle("")
            fig.savefig(
                savepath[:-4] + "_notitle.png",
                bbox_inches="tight",
                dpi=300,
                format="png",
                transparent=True,
            )
            fig.savefig(
                savepath[:-4] + "_white_notitle.png",
                bbox_inches="tight",
                dpi=300,
                format="png",
                transparent=False,
            )
            fig.savefig(
                savepath[:-4] + "_notitle.pdf",
                bbox_inches="tight",
                dpi=300,
                format="pdf",
            )

            logger.info("Figure saved to %s", savepath)

        if force_plot or savepath is None:
            plt.show()

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e
    finally:
        plt.close(fig)
        del fig, ax
        gc.collect()


def plot_fisher(F0, params_names_fisher, title=None, path=None):
    """
    Plot the Fisher matrix as a heatmap.

    Parameters
    ----------
    F0 : ndarray
        Fisher matrix.
    params_names_fisher : list of str
        Names for the axes.
    title : str, optional
        Title of the plot. Default is "Fisher matrix".
    path : str or Path, optional
        Path to save the plot. If None, the plot is displayed.

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.
    """
    import seaborn as sns

    logger.info("Generating Fisher matrix plot.")
    try:
        plt.figure(figsize=(10, 10))

        # Normalisation for the colourmap
        F0min, F0max = F0.min(), F0.max()
        center = 0 if F0min < 0 and F0max > 0 else np.mean(F0)
        divnorm = colors.TwoSlopeNorm(vmin=F0min, vcenter=center, vmax=F0max)

        # Plot the Fisher matrix
        sns.heatmap(
            F0,
            annot=True,
            fmt=".2e",
            cmap="RdBu_r",
            norm=divnorm,
            square=True,
            cbar_kws={"shrink": 0.8},
        )

        plt.xticks(np.arange(len(params_names_fisher)) + 0.5, params_names_fisher, rotation=0)
        plt.yticks(np.arange(len(params_names_fisher)) + 0.5, params_names_fisher, rotation=0)
        plt.title(title or "Fisher matrix")

        if path:
            path = Path(path)
            for fmt in ["png", "pdf"]:
                plt.savefig(
                    path.with_suffix(f".{fmt}"),
                    bbox_inches="tight",
                    dpi=300,
                    format=fmt,
                    transparent=True,
                )

            logger.info("Figure saved to %s", path)
        else:
            logger.info("Displaying plot.")
            plt.show()

    except Exception as e:
        logger.critical("Unexpected error during Fisher matrix plotting: %s", str(e))
        raise RuntimeError("Fisher matrix plotting failed.") from e
    finally:
        plt.close()
        gc.collect()


def plot_mocks_compact(
    NORM,
    N,
    P,
    Pbins,
    phi_obs,
    Phi_0,
    f_0,
    C_0,
    suptitle=None,
    force_plot=False,
    savepath=None,
):
    """
    Plot and compare observed and simulated power spectra.

    Parameters
    ----------
    NORM : float
        Normalisation factor for the observed spectra.
    N : int
        Number of mock data realisations.
    P : int
        Number of bins for the summaries.
    Pbins : ndarray
        Vector of bin boundaries for the summary statistics.
    phi_obs : ndarray
        Observed power spectrum.
    Phi_0 : ndarray
        Mock realisations of the power spectrum.
    f_0 : ndarray
        Mean power spectrum.
    C_0 : ndarray
        Covariance matrix of the mock summaries.
    suptitle : str, optional
        Title for the plot.
    force_plot : bool, optional
        If True, displays the plot even if a path is specified.
    savepath : str or Path, optional
        Path to save the plot.

    Raises
    ------
    RuntimeError
        If an unexpected error occurs during plotting.
    """
    logger.info("Plotting mock power spectra (compact)...")

    alpha_mocks = 0.25
    alpha_binning = 0.3

    try:
        Phi_0_full = Phi_0.copy()
        phi_obs_full = phi_obs.copy()
        f_0_full = f_0.copy()
        C_0_full = C_0.copy()

        idx = 0
        Phi_0 = Phi_0[:, idx * P : (idx + 1) * P]
        phi_obs = phi_obs[idx * P : (idx + 1) * P]
        f_0 = f_0[idx * P : (idx + 1) * P]
        C_0 = C_0[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]

        COLOUR_LIST_means = ["darkorchid", "saddlebrown", "mediumvioletred"]

        fig = plt.figure(figsize=(10, 10))
        gs0 = gridspec.GridSpec(
            3,
            2,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.0, 1.0, 1.0],
            wspace=0.0,
            hspace=0.0,
        )
        gs0.update(right=1.0, left=0.0)
        ax0 = plt.subplot(gs0[0, 0])
        ax0b = plt.subplot(gs0[0, 1])
        ax01 = plt.subplot(gs0[1, 0], sharex=ax0)
        ax01b = plt.subplot(gs0[1, 1])
        ax02 = plt.subplot(gs0[2, 0], sharex=ax0)
        ax02b = plt.subplot(gs0[2, 1])
        axx0x = [[ax01, ax01b], [ax02, ax02b]]

        # Observed power spectrum (normalised)
        ax0.semilogx(
            Pbins,
            phi_obs * NORM,
            linewidth=2,
            color="black",
            label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
            zorder=3,
        )

        # Realisations at the expansion point
        for i in range(N - 1):
            ax0.semilogx(Pbins, Phi_0[i], color="C7", alpha=alpha_mocks, linewidth=0.7)

        # Average value
        ax0.semilogx(
            Pbins,
            Phi_0[N - 1],
            color="C7",
            alpha=alpha_mocks,
            linewidth=0.7,
        )
        ax0.semilogx(
            Pbins,
            f_0,
            linewidth=2,
            color=COLOUR_LIST_means[idx],
            linestyle="--",
            label=r"$\textbf{f}_0$",
            zorder=2,
        )

        # 2-sigma intervals
        ax0.fill_between(
            Pbins,
            f_0 - 2 * np.sqrt(np.diag(C_0)),
            f_0 + 2 * np.sqrt(np.diag(C_0)),
            color=COLOUR_LIST[idx],
            alpha=0.4,
            label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
            zorder=2,
        )

        # Plot the binning
        (ymin, ymax) = ax0.get_ylim()
        ax0.set_ylim([ymin, ymax])
        for i in range(len(Pbins)):
            ax0.plot(
                (Pbins[i], Pbins[i]),
                (ymin, ymax),
                linestyle="--",
                linewidth=0.8,
                color="red",
                alpha=alpha_binning,
                zorder=1,
            )

        ax0.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
        ax0.set_ylabel("population 1", size=GLOBAL_FS)
        ax0.legend(fontsize=GLOBAL_FS, loc="upper right")
        ax0.xaxis.set_ticks_position("both")
        ax0.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        for axis in ["top", "bottom", "left", "right"]:
            ax0.spines[axis].set_linewidth(1.0)
        ax0.xaxis.set_tick_params(which="major", length=6)
        ax0.xaxis.set_tick_params(which="minor", length=4)
        if suptitle is not None:
            ax0.set_title(
                r"$\boldsymbol{\Phi}$ (" + suptitle + ")", y=1.05, fontsize=GLOBAL_FS_LARGE
            )
        else:
            ax0.set_title(r"$\boldsymbol{\Phi}$", y=1.05, fontsize=GLOBAL_FS_LARGE)
        ax0.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
        ax0.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # Same as above but normalise everything by the observations:
        normalisation = phi_obs

        ax0b.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])

        # Observed power spectrum (normalised)
        ax0b.semilogx(
            Pbins,
            NORM * phi_obs / normalisation,
            linewidth=2,
            color="black",
            label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
            zorder=3,
        )

        for i in range(N - 1):
            ax0b.semilogx(
                Pbins,
                Phi_0[i] / normalisation,
                color="C7",
                alpha=alpha_mocks,
                linewidth=0.7,
            )

        ax0b.semilogx(
            Pbins,
            Phi_0[N - 1] / normalisation,
            color="C7",
            alpha=alpha_mocks,
            linewidth=0.7,
            label=r"$\boldsymbol{\Phi}_{\theta_0}$",
        )

        ax0b.semilogx(
            Pbins,
            f_0 / normalisation,
            linewidth=2,
            color=COLOUR_LIST_means[idx],
            linestyle="--",
            label=r"$\textbf{f}_0$",
            zorder=2,
        )

        # 2-sigma intervals
        ax0b.fill_between(
            Pbins,
            f_0 / normalisation - 2 * np.sqrt(np.diag(C_0)) / normalisation,
            f_0 / normalisation + 2 * np.sqrt(np.diag(C_0)) / normalisation,
            color=COLOUR_LIST[idx],
            alpha=0.4,
            label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
            zorder=2,
        )

        # Plot the binning
        (ymin, ymax) = ax0b.get_ylim()
        ax0b.set_ylim([ymin, ymax])
        for i in range(len(Pbins)):
            ax0b.plot(
                (Pbins[i], Pbins[i]),
                (ymin, ymax),
                linestyle="--",
                linewidth=0.8,
                color="red",
                alpha=alpha_binning,
                zorder=1,
            )

        ax0b.set_xlabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS)
        ax0b.xaxis.set_ticks_position("both")
        ax0b.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        for axis in ["top", "bottom", "left", "right"]:
            ax0b.spines[axis].set_linewidth(1.0)
        ax0b.xaxis.set_tick_params(which="major", length=6)
        ax0b.xaxis.set_tick_params(which="minor", length=4)
        ax0b.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
        ax0b.yaxis.tick_right()
        if suptitle is not None:
            ax0b.set_title(
                r"$\boldsymbol{\Phi}/\boldsymbol{\Phi}_\mathrm{O}$ (" + suptitle + ")",
                y=1.05,
                fontsize=GLOBAL_FS,
            )
        else:
            ax0b.set_title(
                r"$\boldsymbol{\Phi}/\boldsymbol{\Phi}_\mathrm{O}$", y=1.05, fontsize=GLOBAL_FS
            )

        for ax, axb in axx0x:
            idx += 1
            Phi_0 = Phi_0_full[:, idx * P : (idx + 1) * P]
            phi_obs = phi_obs_full[idx * P : (idx + 1) * P]
            f_0 = f_0_full[idx * P : (idx + 1) * P]
            C_0 = C_0_full[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]

            # Same plot but for the other axes:
            ax.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])

            # Observed power spectrum (normalised)
            ax.semilogx(Pbins, phi_obs, linewidth=2, color="black", zorder=3)
            for i in range(N - 1):
                ax.semilogx(Pbins, Phi_0[i], color="C7", alpha=alpha_mocks, linewidth=0.7)
            ax.semilogx(Pbins, Phi_0[N - 1], color="C7", alpha=alpha_mocks, linewidth=0.7)
            ax.semilogx(
                Pbins,
                f_0,
                linewidth=2,
                color=COLOUR_LIST_means[idx],
                linestyle="--",
                label=r"$\textbf{f}_0$",
                zorder=2,
            )

            # 2-sigma intervals
            ax.fill_between(
                Pbins,
                f_0 - 2 * np.sqrt(np.diag(C_0)),
                f_0 + 2 * np.sqrt(np.diag(C_0)),
                color=COLOUR_LIST[idx],
                alpha=0.4,
                label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
                zorder=2,
            )

            # Plot the binning:
            (ymin, ymax) = ax.get_ylim()
            ax.set_ylim([ymin, ymax])
            for i in range(len(Pbins)):
                ax.plot(
                    (Pbins[i], Pbins[i]),
                    (ymin, ymax),
                    linestyle="--",
                    linewidth=0.8,
                    color="red",
                    alpha=alpha_binning,
                    zorder=1,
                )
            ax.set_ylabel("population " + str(idx + 1), size=GLOBAL_FS)
            ax.set_xlabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS)
            ax.tick_params(axis="x", which="major", labelsize=GLOBAL_FS_SMALL, pad=8)
            ax.xaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.0)
            ax.xaxis.set_tick_params(which="major", length=6)
            ax.xaxis.set_tick_params(which="minor", length=4)
            ax.legend(loc="upper right", fontsize=GLOBAL_FS)

            # Normalise everything by the observations:
            normalisation = phi_obs

            axb.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])

            # Observed power spectrum (normalised)
            axb.semilogx(
                Pbins,
                phi_obs / normalisation,
                linewidth=2,
                color="black",
                label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
                zorder=3,
            )

            for i in range(N - 1):
                axb.semilogx(
                    Pbins,
                    Phi_0[i] / normalisation,
                    color="C7",
                    alpha=alpha_mocks,
                    linewidth=0.7,
                )

            axb.semilogx(
                Pbins,
                Phi_0[N - 1] / normalisation,
                color="C7",
                alpha=alpha_mocks,
                linewidth=0.7,
                label=r"$\boldsymbol{\Phi}_{\theta_0}$",
            )

            axb.semilogx(
                Pbins,
                f_0 / normalisation,
                linewidth=2,
                color=COLOUR_LIST_means[idx],
                linestyle="--",
                label=r"$\textbf{f}_0$",
                zorder=2,
            )

            # 2-sigma intervals
            axb.fill_between(
                Pbins,
                f_0 / normalisation - 2 * np.sqrt(np.diag(C_0)) / normalisation,
                f_0 / normalisation + 2 * np.sqrt(np.diag(C_0)) / normalisation,
                color=COLOUR_LIST[idx],
                alpha=0.4,
                label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
                zorder=2,
            )

            # Plot the binning
            (ymin, ymax) = axb.get_ylim()
            axb.set_ylim([ymin, ymax])
            for i in range(len(Pbins)):
                axb.plot(
                    (Pbins[i], Pbins[i]),
                    (ymin, ymax),
                    linestyle="--",
                    linewidth=0.8,
                    color="red",
                    alpha=alpha_binning,
                    zorder=1,
                )

            axb.set_xlabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS)
            axb.tick_params(axis="x", which="major", labelsize=GLOBAL_FS_SMALL, pad=8)
            axb.xaxis.set_ticks_position("both")
            axb.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            axb.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
            axb.yaxis.tick_right()
            for axis in ["top", "bottom", "left", "right"]:
                axb.spines[axis].set_linewidth(1.0)
            axb.xaxis.set_tick_params(which="major", length=6)
            axb.xaxis.set_tick_params(which="minor", length=4)

        if savepath:
            savepath = Path(savepath)
            for fmt in ["png", "pdf"]:
                fig.savefig(savepath.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=300)
            logger.info("Plot saved to %s", savepath)
        if force_plot or not savepath:
            plt.show()

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e
    finally:
        plt.close(fig)
        del fig, ax0, ax0b, ax01, ax01b, ax02, ax02b
        gc.collect()


def plot_C(
    C_0,
    X,
    Y,
    Pbins,
    CMap,
    binning=True,
    suptitle=None,
    savepath=None,
    force=False,
):
    """
    Plot covariance matrix.

    Parameters
    ----------
    C_0 : ndarray
        Covariance matrix.
    X : ndarray
        X-axis grid for plotting.
    Y : ndarray
        Y-axis grid for plotting.
    Pbins : ndarray
        Vector of bin boundaries for the summary statistics.
    CMap : str
        Colormap for the plot.
    binning : bool, optional
        Whether to overlay bin lines on the plot. Default is True.
    suptitle : str, optional
        Title for the plot. If None, a default title is used.
    savepath : str or Path, optional
        Path to save the plot. If None, the plot is displayed.
    force : bool, optional
        If True, displays the plot even if savepath is specified.

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.
    """

    from itertools import product

    logger.info("Plotting covariance matrix...")
    try:
        fig, axs = plt.subplots(3, 3, figsize=(13, 11))
        P = len(Pbins)

        # Determine vmin, vmax and central value for TwoSlopeNorm
        vmin, vmax = C_0.min(), C_0.max()
        centerval = 0 if vmin < 0 and vmax > 0 else np.mean(C_0)
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=centerval, vmax=vmax)

        for i, j in product(range(3), range(3)):
            C_0_ij = C_0[i * P : (i + 1) * P, j * P : (j + 1) * P]
            imat = 2 - i  # Invert to place origin at bottom left
            ax = axs[imat, j]
            ax.set_aspect("equal")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")

            # Plot the covariance matrix
            im1 = ax.pcolormesh(X, Y, C_0_ij[:-1, :-1], shading="flat", norm=divnorm, cmap=CMap)

            # Overlay the binning grid (if enabled)
            if binning:
                for n in range(len(Pbins)):
                    ax.plot(
                        (Pbins[n], Pbins[n]),
                        (Pbins.min(), Pbins.max()),
                        linestyle="--",
                        linewidth=0.5,
                        color="red",
                        alpha=0.5,
                    )
                    ax.plot(
                        (Pbins.min(), Pbins.max()),
                        (Pbins[n], Pbins[n]),
                        linestyle="--",
                        linewidth=0.5,
                        color="red",
                        alpha=0.5,
                    )

            # Custom ticks for boundary axes
            if i == 0:
                ax.xaxis.set_tick_params(
                    which="both", direction="in", width=1.0, labelsize=GLOBAL_FS
                )
                ax.xaxis.set_tick_params(which="major", length=6)
                ax.xaxis.set_tick_params(which="minor", length=4)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.yaxis.set_tick_params(
                    which="both", direction="in", width=1.0, labelsize=GLOBAL_FS
                )
                ax.yaxis.set_tick_params(which="major", length=6)
                ax.yaxis.set_tick_params(which="minor", length=4)
            else:
                ax.set_yticks([])

        # Set title and adjust layout
        suptitle = r"$\textbf{C}_0$" if suptitle is None else r"$\textbf{C}_0$ (" + suptitle + ")"
        plt.suptitle(suptitle, y=0.94, x=0.45, size=GLOBAL_FS + GLOBAL_FS_XLARGE)
        plt.subplots_adjust(wspace=0, hspace=0)

        # Colourbar
        cbar = fig.colorbar(
            im1, ax=axs.ravel().tolist(), shrink=1, pad=0.009, aspect=40, orientation="vertical"
        )
        cbar.ax.tick_params(
            axis="y", direction="in", width=1.0, length=6, labelsize=GLOBAL_FS_LARGE
        )
        cbar.update_normal(im1)
        cbar.mappable.set_clim(vmin=C_0[:-1, :-1].min(), vmax=C_0[:-1, :-1].max())

        loc_xticks = np.concatenate([np.linspace(vmin, 0, 5), np.linspace(0, vmax, 5)[1:]])
        val_xticks = np.round(loc_xticks, 2)
        cbar.set_ticks(loc_xticks, labels=val_xticks)

        # Axis labels
        fig.text(0.45, 0.04, r"$k$ [$h$/Mpc]", ha="center", size=GLOBAL_FS_XLARGE)
        fig.text(
            0.04, 0.5, r"$k'$ [$h$/Mpc]", va="center", rotation="vertical", size=GLOBAL_FS_XLARGE
        )

        # Save / display
        if savepath:
            savepath = Path(savepath)
            savepath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(savepath, bbox_inches="tight", dpi=300, format="png", transparent=True)
            fig.savefig(savepath.with_suffix(".pdf"), bbox_inches="tight", dpi=300)

            logger.info("Plot saved to %s", savepath)

        if force or not savepath:
            plt.show()

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e
    finally:
        plt.close(fig)
        del fig, axs, im1
        gc.collect()


def plot_prior_and_posterior_covariances(
    X,
    Y,
    k_s,
    prior_theta_covariance,
    prior_covariance,
    posterior_theta_covariance,
    P_0,
    force_plot=False,
    suptitle="",
    savepath=None,
):
    """
    Plot prior and posterior covariance matrices.

    Parameters
    ----------
    X : ndarray
        X-axis grid.
    Y : ndarray
        Y-axis grid.
    k_s : ndarray
        Wavenumbers.
    prior_theta_covariance : ndarray
        Prior covariance matrix for normalised spectra.
    prior_covariance : ndarray
        Prior covariance matrix for unnormalised spectra.
    posterior_theta_covariance : ndarray
        Posterior covariance matrix for normalised spectra.
    P_0 : ndarray
        Fiducial power spectrum used for normalisation.
    force_plot : bool, optional
        Display plot even if savepath is set.
    suptitle : str, optional
        Title for the plot.
    savepath : str or Path, optional
        Path to save the plot. If None, the plot is displayed.
    verbose : int, optional
        Verbosity level (0=silent, 1=default, 2=detailed).

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    logger.info("Plotting prior and posterior covariance matrices...")
    try:
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(figsize=(15, 14), nrows=2, ncols=2)
        fig.suptitle(suptitle, fontsize=GLOBAL_FS_XLARGE, y=0.99)

        # Covariance matrix of the prior (normalised spectra theta)
        ax0.set_aspect("equal")
        ax0.set_xscale("log")
        ax0.set_yscale("log")
        ax0.xaxis.set_ticks_position("both")
        ax0.yaxis.set_ticks_position("both")
        ax0.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax0.xaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax0.xaxis.set_tick_params(which="minor", length=4)
        ax0.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax0.yaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax0.yaxis.set_tick_params(which="minor", length=4)
        divider = make_axes_locatable(ax0)
        ax0_cb = divider.new_horizontal(size="5%", pad=0.10)
        im0 = ax0.pcolormesh(X, Y, prior_theta_covariance[:-1, :-1], cmap="Blues", shading="flat")
        for i in range(len(k_s)):
            ax0.plot(
                (k_s[i], k_s[i]),
                (k_s.min(), k_s.max()),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        for i in range(len(k_s)):
            ax0.plot(
                (k_s.min(), k_s.max()),
                (k_s[i], k_s[i]),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        ax0.set_title("$\\textbf{S}$", size=GLOBAL_FS_XLARGE)
        ax0.set_xlabel("$k$ [$h$/Mpc]", size=GLOBAL_FS_XLARGE)
        ax0.set_ylabel("$k$ [$h$/Mpc]", size=GLOBAL_FS_XLARGE)
        fig.add_axes(ax0_cb)
        cbar0 = fig.colorbar(im0, cax=ax0_cb)
        cbar0.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=GLOBAL_FS)

        # Covariance matrix of the prior (unnormalized spectra)
        ax1.set_aspect("equal")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.xaxis.set_ticks_position("both")
        ax1.yaxis.set_ticks_position("both")
        ax1.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax1.xaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax1.xaxis.set_tick_params(which="minor", length=4)
        ax1.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax1.yaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax1.yaxis.set_tick_params(which="minor", length=4)
        divider = make_axes_locatable(ax1)
        ax1_cb = divider.new_horizontal(size="5%", pad=0.10)
        im1 = ax1.pcolormesh(X, Y, prior_covariance[:-1, :-1], cmap="Purples", shading="flat")
        for i in range(len(k_s)):
            ax1.plot(
                (k_s[i], k_s[i]),
                (k_s.min(), k_s.max()),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        for i in range(len(k_s)):
            ax1.plot(
                (k_s.min(), k_s.max()),
                (k_s[i], k_s[i]),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        ax1.set_title(
            "$\\mathrm{diag}(\\textbf{P}_0) \\cdot \\textbf{S} \\cdot \\mathrm{diag}(\\textbf{P}_0)$",
            size=GLOBAL_FS_XLARGE,
        )
        ax1.set_xlabel("$k$ [$h$/Mpc]", size=GLOBAL_FS_XLARGE)
        ax1.set_ylabel("$k$ [$h$/Mpc]", size=GLOBAL_FS_XLARGE)
        fig.add_axes(ax1_cb)
        cbar1 = fig.colorbar(im1, cax=ax1_cb)
        cbar1.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=GLOBAL_FS)

        posterior_covariance = np.diag(P_0).dot(posterior_theta_covariance).dot(np.diag(P_0))

        # Covariance matrix of the posterior (normalised spectra theta)
        ax2.set_aspect("equal")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.xaxis.set_ticks_position("both")
        ax2.yaxis.set_ticks_position("both")
        ax2.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax2.xaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax2.xaxis.set_tick_params(which="minor", length=4)
        ax2.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax2.yaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax2.yaxis.set_tick_params(which="minor", length=4)
        divider = make_axes_locatable(ax2)
        ax2_cb = divider.new_horizontal(size="5%", pad=0.10)
        quantity = posterior_theta_covariance
        vmin = quantity.min()
        vmax = quantity.max()
        if vmin < 0 and vmax > 0:
            centerval = 0
        else:
            centerval = np.mean(quantity)
        norm_posterior = colors.TwoSlopeNorm(
            vmin=posterior_theta_covariance.min(),
            vcenter=centerval,
            vmax=posterior_theta_covariance.max(),
        )
        Blues_Reds = create_colormap("Blues_Reds")
        im2 = ax2.pcolormesh(
            X,
            Y,
            posterior_theta_covariance[:-1, :-1],
            cmap=Blues_Reds,
            norm=norm_posterior,
            shading="flat",
        )
        for i in range(len(k_s)):
            ax2.plot(
                (k_s[i], k_s[i]),
                (k_s.min(), k_s.max()),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        for i in range(len(k_s)):
            ax2.plot(
                (k_s.min(), k_s.max()),
                (k_s[i], k_s[i]),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        ax2.set_title("$\\boldsymbol{\\Gamma}$", size=GLOBAL_FS_XLARGE)
        ax2.set_xlabel("$k$ [$h$/Mpc]", size=GLOBAL_FS_XLARGE)
        ax2.set_ylabel("$k$ [$h$/Mpc]", size=GLOBAL_FS_XLARGE)
        fig.add_axes(ax2_cb)
        cbar2 = fig.colorbar(im2, cax=ax2_cb)
        cbar2.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=GLOBAL_FS)
        cbar2.mappable.set_clim(
            vmin=posterior_theta_covariance.min(), vmax=posterior_theta_covariance.max()
        )
        if posterior_theta_covariance.min() < 0 and posterior_theta_covariance.max() > 0:
            ticks = np.concatenate(
                [
                    np.linspace(posterior_theta_covariance.min(), 0, 5),
                    np.linspace(0, posterior_theta_covariance.max(), 5)[1:],
                ]
            )
        else:
            ticks = np.linspace(
                posterior_theta_covariance.min(), posterior_theta_covariance.max(), 6
            )
        ticks_labels = ["{:.1e}".format(tick) for tick in ticks]
        cbar2.set_ticks(ticks)
        cbar2.set_ticklabels(ticks_labels)

        # Covariance matrix of the posterior (unnormalised spectra)
        ax3.set_aspect("equal")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.xaxis.set_ticks_position("both")
        ax3.yaxis.set_ticks_position("both")
        ax3.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax3.xaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax3.xaxis.set_tick_params(which="minor", length=4)
        ax3.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax3.yaxis.set_tick_params(which="major", length=6, labelsize=GLOBAL_FS)
        ax3.yaxis.set_tick_params(which="minor", length=4)
        divider = make_axes_locatable(ax3)
        ax3_cb = divider.new_horizontal(size="5%", pad=0.10)
        quantity = posterior_covariance
        vmin = quantity.min()
        vmax = quantity.max()
        if vmin < 0 and vmax > 0:
            centerval = 0
        else:
            centerval = np.mean(quantity)
        norm_posterior_spectrum = colors.TwoSlopeNorm(
            vmin=posterior_covariance.min(),
            vcenter=centerval,
            vmax=posterior_covariance.max(),
        )
        Purples_Oranges = create_colormap("Purples_Oranges")
        im3 = ax3.pcolormesh(
            X,
            Y,
            posterior_covariance[:-1, :-1],
            cmap=Purples_Oranges,
            norm=norm_posterior_spectrum,
            shading="flat",
        )
        for i in range(len(k_s)):
            ax3.plot(
                (k_s[i], k_s[i]),
                (k_s.min(), k_s.max()),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        for i in range(len(k_s)):
            ax3.plot(
                (k_s.min(), k_s.max()),
                (k_s[i], k_s[i]),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        ax3.set_title(
            "$\\mathrm{diag}(\\textbf{P}_0) \\cdot \\boldsymbol{\\Gamma} \\cdot \\mathrm{diag}(\\textbf{P}_0)$",
            size=22,
        )
        ax3.set_xlabel("$k$ [$h$/Mpc]", size=22)
        ax3.set_ylabel("$k$ [$h$/Mpc]", size=22)
        fig.add_axes(ax3_cb)
        cbar3 = fig.colorbar(im3, cax=ax3_cb)
        cbar3.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=19)
        cbar3.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=19)
        cbar3.mappable.set_clim(vmin=posterior_covariance.min(), vmax=posterior_covariance.max())
        if posterior_covariance.min() < 0 and posterior_covariance.max() > 0:
            ticks = np.concatenate(
                [
                    np.linspace(posterior_covariance.min(), 0, 5),
                    np.linspace(0, posterior_covariance.max(), 5)[1:],
                ]
            )
        else:
            ticks = np.linspace(posterior_covariance.min(), posterior_covariance.max(), 6)
        ticks_labels = ["{:.1e}".format(tick) for tick in ticks]
        cbar3.set_ticks(ticks)
        cbar3.set_ticklabels(ticks_labels)

        fig.tight_layout()
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=300, format="png", transparent=True)
            fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
            fig.suptitle("")
            fig.savefig(
                savepath[:-4] + "_notitle.png",
                bbox_inches="tight",
                dpi=300,
                format="png",
                transparent=True,
            )
            fig.savefig(savepath[:-4] + "_notitle.pdf", bbox_inches="tight", dpi=300, format="pdf")
            logger.info("Plot saved to %s", savepath)
        if force_plot or not savepath:
            plt.show()

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e
    finally:
        plt.close(fig)
        del fig
        gc.collect()


def plot_gradients(
    Pbins,
    P,
    df_16_full,
    df_32_full,
    df_48_full,
    df_full,
    k_s,
    X,
    Y,
    fixscale=False,
    force=False,
    suptitle="",
    savepath=None,
):
    """
    Plot gradients.

    Parameters
    ----------
    Pbins : ndarray
        Vector of bin boundaries for the summary statistics.
    P : int
        Number of bins.
    df_16_full : ndarray
        Derivative with respect to the 16th input.
    df_32_full : ndarray
        Derivative with respect to the 32nd input.
    df_48_full : ndarray
        Derivative with respect to the 48th input.
    df_full : ndarray
        Full derivative.
    k_s : ndarray
        Wavenumbers.
    X : ndarray
        X-axis grid.
    Y : ndarray
        Y-axis grid.
    fixscale : bool, optional
        Fix the y-axis scale. Default is False.
    force : bool, optional
        Display plot even if savepath is set.
    suptitle : str, optional
        Title for the plot.
    savepath : str or Path, optional
        Path to save the plot. If None, the plot is displayed.

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.

    """
    logger.info("Plotting gradients...")

    try:
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(suptitle, y=0.95, fontsize=22)

        gs0 = gridspec.GridSpec(
            3,
            2,
            width_ratios=[1.0, 0.5],
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.0,
            wspace=0.2,
        )
        gs0.update(right=1.0, left=0.0)
        ax00 = plt.subplot(gs0[0, 0])
        ax01 = plt.subplot(gs0[1, 0], sharex=ax00)
        ax02 = plt.subplot(gs0[2, 0], sharex=ax00)

        gs1 = gridspec.GridSpec(
            3,
            2,
            width_ratios=[1.0, 0.5],
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.0,
            wspace=0.2,
        )
        gs1.update(top=0.881, bottom=0.112)
        ax10 = plt.subplot(gs1[0, 1])
        ax11 = plt.subplot(gs1[1, 1], sharex=ax10)
        ax12 = plt.subplot(gs1[2, 1], sharex=ax10)

        axx = [(ax00, ax10), (ax01, ax11), (ax02, ax12)]
        for axs, idx in zip(axx, range(3)):
            ax = axs[0]
            df_16 = np.copy(df_16_full[idx * P : (idx + 1) * P])
            df_32 = np.copy(df_32_full[idx * P : (idx + 1) * P])
            df_48 = np.copy(df_48_full[idx * P : (idx + 1) * P])
            df = df_full[idx * P : (idx + 1) * P]

            # Plot the three selected components of the derivative:
            ax.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
            ax.semilogx(Pbins, np.zeros_like(Pbins), linestyle=":", color="black")
            ax.semilogx(
                Pbins,
                df_16,
                linewidth=2,
                linestyle="-",
                color="C4",
                label=r"$(\nabla \mathbf{f}_0)^\intercal_{16}$",
                zorder=2,
            )
            ax.semilogx(
                Pbins,
                df_32,
                linewidth=2,
                linestyle="-",
                color="C0",
                label=r"$(\nabla \mathbf{f}_0)^\intercal_{32}$",
                zorder=2,
            )
            ax.semilogx(
                Pbins,
                df_48,
                linewidth=2,
                linestyle="-",
                color="C2",
                label=r"$(\nabla \mathbf{f}_0)^\intercal_{48}$",
                zorder=2,
            )
            if fixscale:
                ymin = np.min([df_16, df_32, df_48]) - 1e-2
                ymax = np.max([df_16, df_32, df_48]) + 1e-2
            else:
                (ymin, ymax) = ax.get_ylim()
            ax.set_ylim([ymin, ymax])

            # Binning
            for i in range(len(Pbins)):
                ax.plot(
                    (Pbins[i], Pbins[i]),
                    (ymin, ymax),
                    linestyle="--",
                    linewidth=0.8,
                    color="red",
                    alpha=0.5,
                    zorder=1,
                )
            ax.yaxis.grid(linestyle=":", color="grey")
            ax.set_ylabel("population " + str(idx + 1), size=21)
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax.yaxis.set_tick_params(which="both", direction="in", width=1.0)
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.0)
            ax.xaxis.set_tick_params(which="major", length=6)
            ax.xaxis.set_tick_params(which="minor", length=4)
            ax.yaxis.set_tick_params(which="major", length=6)

            # Plot the full gradient
            ax1 = axs[1]
            ax1.set_xlim([k_s.min(), k_s.max()])
            ax1.set_ylim([Pbins.min(), Pbins.max()])
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.xaxis.set_ticks_position("both")
            ax1.yaxis.set_ticks_position("both")
            ax1.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax1.xaxis.set_tick_params(which="major", length=6)
            ax1.xaxis.set_tick_params(which="minor", length=4)
            ax1.yaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax1.yaxis.set_tick_params(which="major", length=6)
            ax1.yaxis.set_tick_params(which="minor", length=4)
            quantity = df
            vmin = quantity.min()
            vmax = quantity.max()
            if vmin < 0 and vmax > 0:
                centerval = 0
            else:
                centerval = np.mean(quantity)
            norm_grad = colors.TwoSlopeNorm(vmin=vmin, vcenter=centerval, vmax=vmax)
            GradientMap = create_colormap("GradientMap")
            im1 = ax1.pcolormesh(
                X, Y, df[:-1, :-1], cmap=GradientMap, shading="flat", norm=norm_grad
            )
            ax1.plot(k_s, k_s, color="grey", linestyle="--")
            for i in range(len(k_s)):
                ax1.plot(
                    (k_s[i], k_s[i]),
                    (Pbins.min(), Pbins.max()),
                    linestyle=":",
                    linewidth=0.5,
                    color="green",
                    alpha=0.5,
                )
            for i in range(len(Pbins)):
                ax1.plot(
                    (k_s.min(), k_s.max()),
                    (Pbins[i], Pbins[i]),
                    linestyle="--",
                    linewidth=0.5,
                    color="red",
                    alpha=0.5,
                )
            ax1.set_ylabel(r"$k$ [$h$/Mpc]", size=17)
            cbar1 = fig.colorbar(im1, shrink=0.9, pad=0.01)
            cbar1.ax.tick_params(axis="y", direction="in", width=1.0, length=6)
            ticks = np.concatenate([np.linspace(df.min(), 0, 5), np.linspace(0, df.max(), 5)[1:]])
            cbar1.set_ticks(ticks)

            if idx == 0:
                ax.legend(fontsize=21, loc="upper left")
                ax1.set_title(r"Full gradient $\nabla \mathbf{f}_0$", size=21)
            elif idx == 2:
                ax.set_xlabel(r"$k$ [$h$/Mpc]", size=21)
                ax1.set_xlabel(r"$k$ [$h$/Mpc]", size=21)

        if savepath is None or force:
            plt.show()
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=300, format="png", transparent=True)
            fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
            fig.suptitle("")
            fig.savefig(
                savepath[:-4] + "_notitle.png",
                bbox_inches="tight",
                dpi=300,
                format="png",
                transparent=True,
            )
            fig.savefig(savepath[:-4] + "_notitle.pdf", bbox_inches="tight", dpi=300, format="pdf")

            logger.info("Plot saved to %s", savepath)

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e
    finally:
        plt.close(fig)
        del fig
        gc.collect()


def plot_mocks(
    NORM,
    N,
    P,
    Pbins,
    phi_obs,
    Phi_0,
    f_0,
    C_0,
    X,
    Y,
    CMap,
    suptitle=None,
    force_plot=False,
    savepath=None,
):
    """
    Plot mocks.

    Parameters
    ----------
    NORM : float
        Normalisation factor.
    N : int
        Number of mocks.
    P : int
        Number of bins.
    Pbins : ndarray
        Vector of bin boundaries for the summary statistics.
    phi_obs : ndarray
        Observed power spectrum.
    Phi_0 : ndarray
        Mock power spectra.
    f_0 : ndarray
        Averaged power spectrum.
    C_0 : ndarray
        Covariance matrix.
    X : ndarray
        X-axis grid.
    Y : ndarray
        Y-axis grid.
    CMap : str
        Colormap.
    suptitle : str, optional
        Title for the plot.
    force_plot : bool, optional
        Display plot even if savepath is set.
    savepath : str or Path, optional
        Path to save the plot. If None, the plot is displayed.

    Raises
    ------
    RuntimeError
        If unexpected errors occur during plotting.

    """
    logger.info("Plotting mocks and intra-population covariance...")

    alpha_mocks = 0.25
    alpha_binning = 0.3
    try:
        Phi_0_full = Phi_0.copy()
        phi_obs_full = phi_obs.copy()
        f_0_full = f_0.copy()
        C_0_full = C_0.copy()

        idx = 0
        Phi_0 = Phi_0[:, idx * P : (idx + 1) * P]
        phi_obs = phi_obs[idx * P : (idx + 1) * P]
        f_0 = f_0[idx * P : (idx + 1) * P]
        C_0 = C_0[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]

        COLOUR_LIST_means = ["darkorchid", "saddlebrown", "mediumvioletred"]

        fig = plt.figure(figsize=(15.5, 10))
        gs0 = gridspec.GridSpec(
            3,
            3,
            width_ratios=[1.0, 1.0, 1.0],
            height_ratios=[1.0, 1.0, 1.0],
            wspace=0.0,
            hspace=0.0,
        )
        gs0.update(right=1.0, left=0.0)
        ax0 = plt.subplot(gs0[0, 0])
        ax0b = plt.subplot(gs0[0, 1])
        ax01 = plt.subplot(gs0[1, 0], sharex=ax0)
        ax01b = plt.subplot(gs0[1, 1])
        ax02 = plt.subplot(gs0[2, 0], sharex=ax0)
        ax02b = plt.subplot(gs0[2, 1])

        axx0x = [[ax01, ax01b], [ax02, ax02b]]

        # Observed power spectrum (normalised)
        ax0.semilogx(
            Pbins,
            phi_obs * NORM,
            linewidth=2,
            color="black",
            label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
            zorder=3,
        )

        # Plot the Ne realisations and the average value
        for i in range(N - 1):
            ax0.semilogx(Pbins, Phi_0[i], color="C7", alpha=alpha_mocks, linewidth=0.7)
        ax0.semilogx(
            Pbins,
            Phi_0[N - 1],
            color="C7",
            alpha=alpha_mocks,
            linewidth=0.7,
        )
        ax0.semilogx(
            Pbins,
            f_0,
            linewidth=2,
            color=COLOUR_LIST_means[idx],
            linestyle="--",
            label=r"$\textbf{f}_0$",
            zorder=2,
        )

        # 2-sigma intervals
        ax0.fill_between(
            Pbins,
            f_0 - 2 * np.sqrt(np.diag(C_0)),
            f_0 + 2 * np.sqrt(np.diag(C_0)),
            color=COLOUR_LIST[idx],
            alpha=0.4,
            label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
            zorder=2,
        )

        # Plot the binning
        (ymin, ymax) = ax0.get_ylim()
        ax0.set_ylim([ymin, ymax])
        for i in range(len(Pbins)):
            ax0.plot(
                (Pbins[i], Pbins[i]),
                (ymin, ymax),
                linestyle="--",
                linewidth=0.8,
                color="red",
                alpha=alpha_binning,
                zorder=1,
            )

        ax0.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
        ax0.set_ylabel("population 1", size=GLOBAL_FS)
        ax0.legend(fontsize=GLOBAL_FS, loc="upper right")
        ax0.xaxis.set_ticks_position("both")
        ax0.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        for axis in ["top", "bottom", "left", "right"]:
            ax0.spines[axis].set_linewidth(1.0)
        ax0.xaxis.set_tick_params(which="major", length=6)
        ax0.xaxis.set_tick_params(which="minor", length=4)
        if suptitle is not None:
            ax0.set_title(
                r"$\boldsymbol{\Phi}$ (" + suptitle + ")", y=1.05, fontsize=GLOBAL_FS_LARGE
            )
        else:
            ax0.set_title(r"$\boldsymbol{\Phi}$", y=1.05, fontsize=GLOBAL_FS_LARGE)
        ax0.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
        ax0.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # Same as above but normalise everything by the observations:
        normalisation = phi_obs

        ax0b.semilogx(
            Pbins,
            NORM * phi_obs / normalisation,
            linewidth=2,
            color="black",
            label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
            zorder=3,
        )

        for i in range(N - 1):
            ax0b.semilogx(
                Pbins,
                Phi_0[i] / normalisation,
                color="C7",
                alpha=alpha_mocks,
                linewidth=0.7,
            )

        ax0b.semilogx(
            Pbins,
            Phi_0[N - 1] / normalisation,
            color="C7",
            alpha=alpha_mocks,
            linewidth=0.7,
            label=r"$\boldsymbol{\Phi}_{\theta_0}$",
        )

        ax0b.semilogx(
            Pbins,
            f_0 / normalisation,
            linewidth=2,
            color=COLOUR_LIST_means[idx],
            linestyle="--",
            label=r"$\textbf{f}_0$",
            zorder=2,
        )

        # 2-sigma intervals
        ax0b.fill_between(
            Pbins,
            f_0 / normalisation - 2 * np.sqrt(np.diag(C_0)) / normalisation,
            f_0 / normalisation + 2 * np.sqrt(np.diag(C_0)) / normalisation,
            color=COLOUR_LIST[idx],
            alpha=0.4,
            label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
            zorder=2,
        )

        # Plot the binning
        (ymin, ymax) = ax0b.get_ylim()
        ax0b.set_ylim([ymin, ymax])
        for i in range(len(Pbins)):
            ax0b.plot(
                (Pbins[i], Pbins[i]),
                (ymin, ymax),
                linestyle="--",
                linewidth=0.8,
                color="red",
                alpha=alpha_binning,
                zorder=1,
            )

        ax0b.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
        ax0b.set_xlabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS)
        ax0b.xaxis.set_ticks_position("both")
        ax0b.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        for axis in ["top", "bottom", "left", "right"]:
            ax0b.spines[axis].set_linewidth(1.0)
        ax0b.xaxis.set_tick_params(which="major", length=6)
        ax0b.xaxis.set_tick_params(which="minor", length=4)
        ax0b.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
        ax0b.yaxis.tick_right()
        if suptitle is not None:
            ax0b.set_title(
                r"$\boldsymbol{\Phi}/\boldsymbol{\Phi}_\mathrm{O}$ (" + suptitle + ")",
                y=1.05,
                fontsize=GLOBAL_FS,
            )
        else:
            ax0b.set_title(
                r"$\boldsymbol{\Phi}/\boldsymbol{\Phi}_\mathrm{O}$", y=1.05, fontsize=GLOBAL_FS
            )

        for ax, axb in axx0x:
            idx += 1
            Phi_0 = Phi_0_full[:, idx * P : (idx + 1) * P]
            phi_obs = phi_obs_full[idx * P : (idx + 1) * P]
            f_0 = f_0_full[idx * P : (idx + 1) * P]
            C_0 = C_0_full[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]

            # Same plot as above but for the other axes:
            ax.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
            ax.semilogx(Pbins, phi_obs, linewidth=2, color="black", zorder=3)
            for i in range(N - 1):
                ax.semilogx(Pbins, Phi_0[i], color="C7", alpha=alpha_mocks, linewidth=0.7)
            ax.semilogx(Pbins, Phi_0[N - 1], color="C7", alpha=alpha_mocks, linewidth=0.7)
            ax.semilogx(
                Pbins,
                f_0,
                linewidth=2,
                color=COLOUR_LIST_means[idx],
                linestyle="--",
                label=r"$\textbf{f}_0$",
                zorder=2,
            )

            # 2-sigma intervals
            ax.fill_between(
                Pbins,
                f_0 - 2 * np.sqrt(np.diag(C_0)),
                f_0 + 2 * np.sqrt(np.diag(C_0)),
                color=COLOUR_LIST[idx],
                alpha=0.4,
                label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
                zorder=2,
            )

            # Plot the binning
            (ymin, ymax) = ax.get_ylim()
            ax.set_ylim([ymin, ymax])
            for i in range(len(Pbins)):
                ax.plot(
                    (Pbins[i], Pbins[i]),
                    (ymin, ymax),
                    linestyle="--",
                    linewidth=0.8,
                    color="red",
                    alpha=alpha_binning,
                    zorder=1,
                )
            ax.set_ylabel("population " + str(idx + 1), size=GLOBAL_FS)
            ax.set_xlabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS)
            ax.tick_params(axis="x", which="major", labelsize=GLOBAL_FS_SMALL, pad=8)
            ax.xaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(1.0)
            ax.xaxis.set_tick_params(which="major", length=6)
            ax.xaxis.set_tick_params(which="minor", length=4)
            ax.legend(loc="upper right", fontsize=GLOBAL_FS)

            # Same as above but normalise everything by the observations:
            normalisation = phi_obs

            # Observed power spectrum
            axb.semilogx(
                Pbins,
                phi_obs / normalisation,
                linewidth=2,
                color="black",
                label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
                zorder=3,
            )

            for i in range(N - 1):
                axb.semilogx(
                    Pbins,
                    Phi_0[i] / normalisation,
                    color="C7",
                    alpha=alpha_mocks,
                    linewidth=0.7,
                )

            axb.semilogx(
                Pbins,
                Phi_0[N - 1] / normalisation,
                color="C7",
                alpha=alpha_mocks,
                linewidth=0.7,
                label=r"$\boldsymbol{\Phi}_{\theta_0}$",
            )
            axb.semilogx(
                Pbins,
                f_0 / normalisation,
                linewidth=2,
                color=COLOUR_LIST_means[idx],
                linestyle="--",
                label=r"$\textbf{f}_0$",
                zorder=2,
            )

            # 2-sigma intervals
            axb.fill_between(
                Pbins,
                f_0 / normalisation - 2 * np.sqrt(np.diag(C_0)) / normalisation,
                f_0 / normalisation + 2 * np.sqrt(np.diag(C_0)) / normalisation,
                color=COLOUR_LIST[idx],
                alpha=0.4,
                label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
                zorder=2,
            )

            # Plot the binning
            (ymin, ymax) = axb.get_ylim()
            axb.set_ylim([ymin, ymax])
            for i in range(len(Pbins)):
                axb.plot(
                    (Pbins[i], Pbins[i]),
                    (ymin, ymax),
                    linestyle="--",
                    linewidth=0.8,
                    color="red",
                    alpha=alpha_binning,
                    zorder=1,
                )

            axb.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
            axb.set_xlabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS)
            axb.tick_params(axis="x", which="major", labelsize=GLOBAL_FS_SMALL, pad=8)
            axb.xaxis.set_ticks_position("both")
            axb.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            axb.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)
            axb.yaxis.tick_right()
            for axis in ["top", "bottom", "left", "right"]:
                axb.spines[axis].set_linewidth(1.0)
            axb.xaxis.set_tick_params(which="major", length=6)
            axb.xaxis.set_tick_params(which="minor", length=4)

        axx1x = [plt.subplot(gs0[0, 2]), plt.subplot(gs0[1, 2]), plt.subplot(gs0[2, 2])]

        # Diagonal blocks of the covariance matrix (intra-population covariance)
        idx = 0
        for ax1 in axx1x:
            Phi_0 = Phi_0_full[:, idx * P : (idx + 1) * P]
            phi_obs = phi_obs_full[idx * P : (idx + 1) * P]
            f_0 = f_0_full[idx * P : (idx + 1) * P]
            C_0 = C_0_full[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]
            idx += 1

            C0min = C_0.min()
            C0max = C_0.max()
            if C0min < 0 and C0max > 0:
                centerval = 0
            else:
                centerval = np.mean(C_0)
            divnorm = colors.TwoSlopeNorm(vmin=C_0.min(), vcenter=centerval, vmax=C_0.max())

            # Plot the current block of the covariance matrix
            im1 = ax1.pcolormesh(X, Y, C_0[:-1, :-1], shading="flat", norm=divnorm, cmap=CMap)

            # Plot the binning
            for i in range(len(Pbins)):
                ax1.plot(
                    (Pbins[i], Pbins[i]),
                    (Pbins.min(), Pbins.max()),
                    linestyle="--",
                    linewidth=0.5,
                    color="red",
                    alpha=0.5,
                )
            for i in range(len(Pbins)):
                ax1.plot(
                    (Pbins.min(), Pbins.max()),
                    (Pbins[i], Pbins[i]),
                    linestyle="--",
                    linewidth=0.5,
                    color="red",
                    alpha=0.5,
                )
            if idx == 1:
                ax1.set_title(r"diagonal blocks of $\textbf{C}_0$", size=GLOBAL_FS, y=1.05)
            ax1.set_xlabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS)
            ax1.set_ylabel(r"$k$ [$h$/Mpc]", size=GLOBAL_FS_SMALL)

            ax1.set_aspect("equal")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.xaxis.set_ticks_position("both")
            ax1.yaxis.set_ticks_position("both")
            ax1.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax1.xaxis.set_tick_params(which="major", length=6)
            ax1.xaxis.set_tick_params(which="minor", length=4)
            ax1.yaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax1.yaxis.set_tick_params(which="major", length=6)
            ax1.yaxis.set_tick_params(which="minor", length=4)
            ax1.tick_params(axis="x", which="major", labelsize=GLOBAL_FS_SMALL, pad=8)
            ax1.tick_params(axis="y", which="major", labelsize=GLOBAL_FS_TINY)

            cbar1 = fig.colorbar(im1, shrink=0.9, pad=0.02, format="%.1e")
            cbar1.ax.tick_params(axis="y", direction="in", width=1.0, length=4)
            cbar1.ax.yaxis.set_tick_params(labelsize=GLOBAL_FS_SMALL)
            cbar1.update_normal(im1)
            vmin = C_0[:-1, :-1].min()
            vmax = C_0[:-1, :-1].max()
            cbar1.mappable.set_clim(vmin=vmin, vmax=vmax)
            if vmin < 0 and vmax > 0:
                ticks = np.concatenate(
                    [
                        np.linspace(vmin * 0.8, 0, 2),
                        np.linspace(0, 0.8 * vmax, 2)[1:],
                    ]
                )
            else:
                ticks = np.linspace(vmin, vmax, 5)
            cbar1.set_ticks(ticks)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            cbar1.ax.yaxis.set_major_formatter(formatter)
            cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            cbar1.ax.yaxis.get_offset_text().set_fontsize(GLOBAL_FS_TINY)
            cbar1.update_ticks()

        if savepath is not None:
            savepath = Path(savepath)
            fig.savefig(savepath, bbox_inches="tight", transparent=True, dpi=300, format="png")
            fig.savefig(savepath.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
            logger.info("Plot saved to %s", savepath)
        if force_plot or not savepath:
            plt.show()

    except Exception as e:
        logger.critical("Unexpected error during plotting: %s", str(e))
        raise RuntimeError("Plotting failed.") from e


def plot_histogram(
    data,
    recmahal,
    suptitle,
    savepath,
    bins=30,
    alpha=0.5,
    color="blue",
):
    """
    Plot a Mahalanobis distance histogram with key reference lines.

    Parameters
    ----------
    data : array-like
        Collection of Mahalanobis distances.
    recmahal : float
        Single Mahalanobis distance for the reconstruction
        (e.g. posterior vs. prior mean).
    suptitle : str
        Figure title.
    savepath : str
        Full path (including filename) to save the resulting plot.
    bins : int, optional
        Number of bins for the histogram. Default is 30.
    alpha : float, optional
        Transparency level for the histogram bars. Default is 0.5.
    color : str, optional
        Colour of the histogram bars. Default is "blue".

    Raises
    ------
    OSError
        If the file cannot be saved to the specified path.
    RuntimeError
        For unexpected plotting failures.
    """
    try:
        logger.debug("Starting plot_histogram with data of size %d.", len(data))

        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=bins, alpha=alpha, color=color)

        labrec = r"$d_\mathrm{M}(\boldsymbol{\gamma}_{\textrm{rec}}, \boldsymbol{\theta}_0 \mid \textbf{S})$"
        plt.axvline(recmahal, color="black", linestyle="-", linewidth=2, label=labrec)

        labx = r"$d_\mathrm{M}(\boldsymbol{\gamma}, \boldsymbol{\theta}_0 \mid \textbf{S})$"
        labmean = r"$\langle d_\mathrm{M}(\boldsymbol{\gamma}, \boldsymbol{\theta}_0 \mid \textbf{S}) \rangle$"
        plt.axvline(
            np.mean(data),
            color="tab:pink",
            linestyle="-",
            linewidth=2,
            label=labmean,
        )
        std = np.std(data)
        labstd = r"$\pm 1 \sigma$"
        plt.axvline(
            np.mean(data) + std,
            color="pink",
            linestyle="--",
            linewidth=1,
            label=labstd,
        )
        plt.axvline(
            np.mean(data) - std,
            color="pink",
            linestyle="--",
            linewidth=1,
        )

        plt.xlabel(labx, fontsize=GLOBAL_FS_LARGE)
        plt.ylabel("Density", fontsize=GLOBAL_FS_LARGE)
        plt.xticks(fontsize=GLOBAL_FS)
        plt.yticks([])
        plt.suptitle(suptitle, fontsize=GLOBAL_FS_XLARGE)
        plt.legend(fontsize=GLOBAL_FS_LARGE)

        plt.savefig(savepath, bbox_inches="tight", dpi=300, transparent=True)
        plt.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300)
        logger.info("Histogram plot saved to %s and %s.pdf", savepath, savepath[:-4])
        plt.close()

    except OSError as e:
        logger.error("File saving failed at path '%s': %s", savepath, str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error during histogram plotting: %s", str(e))
        raise RuntimeError("plot_histogram failed.") from e
    finally:
        gc.collect()
        logger.debug("plot_histogram: memory cleanup done.")
