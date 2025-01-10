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

"""Utility functions for parsing command-line arguments.
"""

import os
from argparse import ArgumentParser, ArgumentTypeError


def joinstrs(list_of_strs):
    """Join a list of strings into a single string.

    Parameters
    ----------
    list_of_strs : list of str
        List of strings to join.

    Returns
    -------
    str
        Concatenated string.
    """
    return "".join([str(x) for x in list_of_strs if x is not None])


def joinstrs_only(list_of_strs):
    """Join a list of strings into a single string, ignoring all
    non-string elements such as None values.

    Parameters
    ----------
    list_of_strs : list of str
        List of strings to join.

    Returns
    -------
    str
        Concatenated string.
    """
    return "".join([str(x) for x in list_of_strs if type(x) == str])


def check_files_exist(files):
    """Check if all files in the list exist.

    Parameters
    ----------
    files : list of str
        List of file paths to check.

    Returns
    -------
    bool
        True if all files exist, False otherwise.
    """

    return all(os.path.exists(f) for f in files)


def none_or_bool_or_str(value):
    """Convert a string to None, bool, or str.

    Parameters
    ----------
    value : str
        String to convert.

    Returns
    -------
    None, bool, or str
        Converted value.
    """
    if value == "None" or value == None:
        return None
    elif value == "True":
        return True
    elif value == "False":
        return False
    return value


def intNone(value):
    """Convert a string to None or int.

    Parameters
    ----------
    value : str
        String to convert.

    Returns
    -------
    None or int
        Converted value.
    """
    if value == "None" or value == None:
        return None
    else:
        return int(value)


def safe_npload(path):
    """Load a numpy array from a file.

    Parameters
    ----------
    path : str
        Path to the file to load.

    Returns
    -------
    None or np.ndarray
        Loaded array or None if the file does not exist.
    """
    import numpy as np

    val = np.load(path, allow_pickle=True)
    if val is None or val == "None" or val == None:
        return None
    else:
        return val


def bool_sh(value):
    """Convert a string to a boolean.

    Parameters
    ----------
    value : str
        String to convert.

    Returns
    -------
    bool
        Converted value.
    """
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
