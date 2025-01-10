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
This module provides utility functions for the examples.
"""


def clear_large_plot(fig):
    """
    Clear a figure to free up memory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to clear.
    """
    from IPython.display import clear_output

    del fig
    clear_output()
