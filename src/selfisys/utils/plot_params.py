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

"""
Plotting utilities and custom colormaps for the SelfiSys package.

This module provides custom Matplotlib settings, formatter classes, and
colormaps used for visualising results in the SelfiSys project.
"""


# Global font sizes
GLOBAL_FS = 20
GLOBAL_FS_LARGE = 22
GLOBAL_FS_XLARGE = 24
GLOBAL_FS_SMALL = 18
GLOBAL_FS_TINY = 16
COLOUR_LIST = ["C4", "C5", "C6", "C7"]


def reset_plotting():
    import matplotlib as mpl

    mpl.rcParams.update(mpl.rcParamsDefault)


def setup_plotting():
    """
    Configure Matplotlib plotting settings for consistent appearance.
    """
    import matplotlib.pyplot as plt
    import importlib.resources

    with importlib.resources.open_text("selfisys", "preamble.tex") as f:
        preamble = f.read()

    # Dictionary with rcParams settings
    rcparams = {
        "font.family": "serif",
        "font.size": GLOBAL_FS,  # Base font size
        "axes.titlesize": GLOBAL_FS_XLARGE,
        "axes.labelsize": GLOBAL_FS_LARGE,
        "axes.linewidth": 1.0,
        "xtick.labelsize": GLOBAL_FS_SMALL,
        "ytick.labelsize": GLOBAL_FS_SMALL,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 5,
        "xtick.minor.pad": 5,
        "ytick.major.pad": 5,
        "ytick.minor.pad": 5,
        "legend.fontsize": GLOBAL_FS_SMALL,
        "legend.title_fontsize": GLOBAL_FS_LARGE,
        "figure.titlesize": GLOBAL_FS_XLARGE,
        "figure.dpi": 300,
        "grid.color": "gray",
        "grid.linestyle": "dotted",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "text.usetex": True,
        "text.latex.preamble": preamble,
    }

    # Update rcParams
    plt.rcParams.update(rcparams)


def dynamic_text_scaling(fig_height):
    """
    Dynamically scale text sizes based on the vertical height of the
    figure.

    Parameters
    ----------
    fig_height : float
        Height of the figure in inches.

    Returns
    -------
    dict
        Dictionary of scaled font sizes for consistent appearance.
    """
    scaling_factor = fig_height / 6.0  # Reference height is 6 inches
    return {
        "font.size": GLOBAL_FS * scaling_factor,
        "axes.titlesize": GLOBAL_FS_XLARGE * scaling_factor,
        "axes.labelsize": GLOBAL_FS_LARGE * scaling_factor,
        "xtick.labelsize": GLOBAL_FS_SMALL * scaling_factor,
        "ytick.labelsize": GLOBAL_FS_SMALL * scaling_factor,
        "legend.fontsize": GLOBAL_FS_SMALL * scaling_factor,
        "legend.title_fontsize": GLOBAL_FS_LARGE * scaling_factor,
        "figure.titlesize": GLOBAL_FS_XLARGE * scaling_factor,
    }


class ScalarFormatterForceFormat_11:
    """
    Custom scalar formatter to enforce a specific number format with an
    offset.

    This formatter displays tick labels with one decimal place and
    includes the offset notation for powers of ten.
    """

    def __init__(self, useOffset=True, useMathText=True, useLocale=None):
        from matplotlib.ticker import ScalarFormatter

        self.formatter = ScalarFormatter(
            useOffset=useOffset, useMathText=useMathText, useLocale=useLocale
        )
        self.formatter.set_powerlimits((0, 0))

    def __call__(self, val, pos=None):
        return self.formatter.__call__(val, pos)

    def set_scientific(self, b):
        self.formatter.set_scientific(b)

    def set_useOffset(self, b):
        self.formatter.set_useOffset(b)

    def get_offset(self):
        offset = self.formatter.get_offset()
        if self.formatter.orderOfMagnitude != 0:
            return r"$\times 10^{%+d}$" % self.formatter.orderOfMagnitude
        else:
            return r"$\times 10^{+0}$"


def get_contours(Z, nBins, confLevels=(0.3173, 0.0455, 0.0027)):
    """
    Compute contour levels for given confidence levels.

    Parameters
    ----------
    Z : ndarray
        2D histogram or density estimate.
    nBins : int
        Number of bins along one axis.
    confLevels : tuple of float
        Confidence levels for which to compute contour levels.

    Returns
    -------
    chainLevels : ndarray
        Contour levels corresponding to the provided confidence levels.
    """
    import numpy as np

    Z = Z / Z.sum()
    nContourLevels = len(confLevels)
    chainLevels = np.ones(nContourLevels + 1)
    histOrdered = np.sort(Z.flat)
    histCumulative = np.cumsum(histOrdered)
    nBinsFlat = np.linspace(0.0, nBins**2, nBins**2)

    for l in range(nContourLevels):
        temp = np.interp(confLevels[l], histCumulative, nBinsFlat)
        chainLevels[nContourLevels - 1 - l] = np.interp(temp, nBinsFlat, histOrdered)

    return chainLevels


def create_colormap(name):
    """
    Create a custom colormap based on the specified name.

    Parameters
    ----------
    name : str
        The name of the colormap to create.

    Returns
    -------
    ListedColormap
        The requested custom colormap.

    Raises
    ------
    ValueError
        If the specified colormap name is not recognised.
    """
    import numpy as np
    from matplotlib import cm, colors, colormaps

    if name == "GalaxyMap":
        # Colormap for slices through galaxy density fields
        Ndots = 2**13
        stretch_top = 0.5
        truncate_bottom = 0.0
        stretch_bottom = 1.0

        top = cm.get_cmap("RdPu", Ndots)
        top = colors.LinearSegmentedColormap.from_list("", ["white", top(0.5), top(1.0)])
        bottom = cm.get_cmap("Greens_r", Ndots)
        bottom = colors.LinearSegmentedColormap.from_list("", [bottom(0), bottom(0.5), "white"])

        interp_top = np.linspace(0, 1, Ndots) ** stretch_top
        interp_bottom = np.linspace(truncate_bottom, 1, Ndots) ** stretch_bottom
        cols_galaxy = np.vstack((bottom(interp_bottom), top(interp_top)))
        return colors.ListedColormap(cols_galaxy, name="GalaxyMap")

    elif name == "GradientMap":
        # Colormap for gradient matrices
        Ndots = 2**13
        stretch_bottom = 6.0
        stretch_top = 1 / 2.5
        truncate_bottom = 0.35

        bottom = cm.get_cmap("BuGn_r", Ndots)
        top = cm.get_cmap("RdPu", Ndots)

        interp_top = np.linspace(0, 1, Ndots) ** stretch_top
        interp_bottom = np.linspace(truncate_bottom, 1, Ndots) ** stretch_bottom
        newcolors = np.vstack((bottom(interp_bottom), top(interp_top)))
        return colors.ListedColormap(newcolors, name="GradientMap")

    elif name == "CovarianceMap":
        # Colormap for the diagonal blocks of covariance matrices
        Ndots = 2**15
        stretch_top_1 = 0.3
        stretch_top_2 = 1.0
        stretch_bottom = 0.2
        middle = 0.4  # Middle of the positive scale, between 0 and 1
        cmap_name = "BrBG"
        top = colormaps[cmap_name]
        bottom = colormaps[cmap_name]

        interp_top = np.concatenate(
            (
                middle * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_1 + 0.5,
                (1 - middle) * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_2 + 0.5 + middle,
            )
        )
        interp_bottom = np.linspace(0.0, 1.0, Ndots) ** stretch_bottom - 0.5
        newcolors = np.vstack((bottom(interp_bottom), top(interp_top)))
        return colors.ListedColormap(newcolors, name="CovarianceMap")

    elif name == "FullCovarianceMap":
        # Colormap for full covariance matrices
        Ndots = 2**15
        stretch_top_1 = 0.3
        stretch_top_2 = 1.0
        middle_top = 0.4  # Middle of the positive scale, between 0 and 1
        stretch_bottom_1 = 1.0
        stretch_bottom_2 = 5.0
        middle_bottom = 0.7  # Middle of the negative scale, between 0 and 1
        colname = "PRGn_r"  # Options: "PRGn", "PRGn_r", "BrBG", "PuOr"
        top = colormaps[colname]
        bottom = colormaps[colname]

        interp_top = np.concatenate(
            (
                middle_top * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_1 + 0.5,
                (1 - middle_top) * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_2
                + 0.5
                + middle_top,
            )
        )
        interp_bottom = np.concatenate(
            (
                middle_bottom * np.linspace(0.0, 1, Ndots // 2) ** stretch_bottom_1 - 0.5,
                (1 - middle_bottom) * np.linspace(0.0, 1, Ndots // 2) ** stretch_bottom_2
                - 0.5
                + middle_bottom,
            )
        )
        newcolors = np.vstack((bottom(interp_bottom), top(interp_top)))
        return colors.ListedColormap(newcolors, name="FullCovarianceMap")

    elif name == "Blues_Reds":
        # Additional colormap combining blues and reds
        top = cm.get_cmap("Reds_r", 128)
        bottom = cm.get_cmap("Blues", 128)
        newcolors = np.vstack((top(np.linspace(0.7, 1, 128)), bottom(np.linspace(0, 1, 128))))
        return colors.ListedColormap(newcolors, name="Blues_Reds")

    elif name == "Purples_Oranges":
        # Additional colormap combining purples and oranges
        top = cm.get_cmap("Oranges_r", 128)
        bottom = cm.get_cmap("Purples", 128)
        newcolors = np.vstack((top(np.linspace(0.7, 1, 128)), bottom(np.linspace(0, 1, 128))))
        return colors.ListedColormap(newcolors, name="Purples_Oranges")

    else:
        raise ValueError(f"Colormap '{name}' is not defined.")


def create_all_colormaps():
    """
    Create all custom colormaps.

    Returns
    -------
    colormaps : dict
        Dictionary containing all custom colormaps.
    """
    colormaps_dict = {}
    colormap_names = [
        "GalaxyMap",
        "GradientMap",
        "CovarianceMap",
        "FullCovarianceMap",
        "Blues_Reds",
        "Purples_Oranges",
    ]
    for name in colormap_names:
        colormaps_dict[name] = create_colormap(name)
    return colormaps_dict
