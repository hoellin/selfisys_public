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

"""Tools for generating Gaussian random fields from given power spectra.
"""

from selfisys.utils.logger import getCustomLogger

logger = getCustomLogger(__name__)


def primordial_grf(
    L,
    N,
    seedphases,
    fname_powerspectrum,
    fname_outputinitialdensity,
    force_sim=False,
    return_g=False,
    verbose=0,
):
    """
    Generate a Gaussian random field from a specified input power
    spectrum.

    Parameters
    ----------
    L : float
        Side length of the simulation box in Mpc/h.
    N : int
        Grid resolution (number of cells per dimension).
    seedphases : int
        Seed for random phase generation (for reproducibility).
    fname_powerspectrum : str
        File path to the input power spectrum.
    fname_outputinitialdensity : str
        File path to store the generated initial density field.
    force_sim : bool, optional
        If True, regenerate the GRF even if the output file exists.
        Default is False.
    return_g : bool, optional
        If True, return the GRF as a numpy array. Default is False.
    verbose : int, optional
        Verbosity level (0 = silent, 1 = progress, 2 = detailed).
        Default is 0.

    Raises
    ------
    OSError
        If the power spectrum file cannot be read.
    RuntimeError
        If an unexpected error occurs during power spectrum reading.

    Returns
    -------
    numpy.ndarray or None
        The GRF data if `return_g` is True, otherwise None.
    """
    from os.path import exists
    from gc import collect
    from pysbmy.power import PowerSpectrum
    from pysbmy.field import Field

    # Skip simulation if output already exists and overwrite is not requested
    if not force_sim and exists(fname_outputinitialdensity):
        from pysbmy.field import read_basefield

        if verbose > 0:
            logger.info(f"{fname_outputinitialdensity} already exists. Skipping simulation.")
        return read_basefield(fname_outputinitialdensity).data if return_g else None

    # Read the power spectrum
    try:
        P = PowerSpectrum.read(fname_powerspectrum)
    except OSError as e:
        logger.error(f"Unable to read power spectrum file: {fname_powerspectrum}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error while reading power spectrum: {e}")
        raise

    # Generate the Gaussian random field
    if verbose > 1:
        g = Field.GRF(L, L, L, 0, 0, 0, N, N, N, P, 1e3, seedphases)  # a_init = 1e3
    else:
        from selfisys.utils.low_level import stdout_redirector
        from io import BytesIO

        # Suppress standard output to avoid cluttering logs
        with BytesIO() as f:
            with stdout_redirector(f):
                g = Field.GRF(L, L, L, 0, 0, 0, N, N, N, P, 1e3, seedphases)

    # Write the field to disk
    g.write(fname_outputinitialdensity)
    field = g.data.copy() if return_g else None

    # Free memory
    del g
    collect()

    return field
