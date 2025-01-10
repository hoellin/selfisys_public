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
Provides simple wrappers around pyselfi.utils functions for the SelfiSys
pipeline.
"""


def PrintMessage(required_verbosity: int, message: str, verbosity: int) -> None:
    """
    Print a message to standard output using pyselfi.utils.PrintMessage.

    Parameters
    ----------
    required_verbosity : int
        The verbosity level required to display the message.
    message : str
        The actual message to display.
    verbosity : int
        The current verbosity level (0=quiet, 1=normal, 2=debug).
    """
    from pyselfi.utils import PrintMessage as PSMessage

    if verbosity >= required_verbosity:
        PSMessage(3, message)


def indent() -> None:
    """Indent the standard output using pyselfi.utils."""
    from pyselfi.utils import INDENT

    INDENT()


def unindent() -> None:
    """Unindent the standard output using pyselfi.utils."""
    from pyselfi.utils import UNINDENT

    UNINDENT()
