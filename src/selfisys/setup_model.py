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
Set up parameters related to the grid and the fiducial power spectrum.
"""

import os.path
import gc
from typing import Optional, NamedTuple
import numpy as np
from h5py import File

from pysbmy.power import PowerSpectrum, FourierGrid, get_Pk
from selfisys.utils.logger import getCustomLogger
from selfisys.utils.tools import get_k_max

logger = getCustomLogger(__name__)


class ModelSetup(NamedTuple):
    size: int
    L: float
    P: int
    S: int
    G_sim_path: str
    G_ss_path: str
    Pbins_bnd: np.ndarray
    Pbins: np.ndarray
    k_s: np.ndarray
    P_ss_obj_path: str
    P_0: np.ndarray
    planck_Pk: np.ndarray


def setup_model(
    workdir: str,
    params_planck: dict,
    params_P0: dict,
    size: int = 256,
    L: float = 3600.0,
    S: int = 100,
    N_exact: int = 8,
    Pinit: int = 50,
    trim_threshold: int = 100,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
    force: bool = False,
) -> ModelSetup:
    """
    Set up the model by computing or loading necessary grids and
    parameters.

    Parameters
    ----------
    workdir : str
        Directory where the results will be stored.
    params_planck : dict
        Parameters for the Planck 2018 cosmology.
    params_P0 : dict
        Parameters for the normalisation power spectrum.
    size : int
        Number of elements in each direction of the box.
    L : float
        Comoving length of the box in Mpc/h.
    S : int
        Number of support wavenumbers for the input power spectra.
    N_exact : int
        Number of support wavenumbers matching the Fourier grid.
    Pinit : int
        Maximum number of bins for the summaries.
    trim_threshold : int
        Minimum number of modes required per bin.
    minval : float, optional
        Minimum k value for the summaries.
    maxval : float, optional
        Maximum k value for the summaries.
    force : bool
        If True, forces recomputation of the inputs.

    Returns
    -------
    ModelSetup
        A named tuple containing:
        - size (int): Number of elements in each direction of the box.
        - L (float): Comoving length of the box in Mpc/h.
        - P (int): Number of bins for the summaries.
        - S (int): Number of support wavenumbers for input powerspectra.
        - G_sim_path (str): Path to the full Fourier grid file.
        - G_ss_path (str): Path to the Fourier grid for summaries file.
        - Pbins_bnd (np.ndarray): Boundaries of summary bins.
        - Pbins (np.ndarray): Centres of the bins for the summaries.
        - k_s (np.ndarray): Support wavenumbers for input power spectra.
        - P_ss_obj_path (str): Path to the summary power spectrum file.
        - P_0 (np.ndarray): Normalisation power spectrum values.
        - planck_Pk (np.ndarray): Planck 2018 power spectrum values.
    """
    # Check input parameters
    if N_exact < 0 or N_exact > S:
        raise ValueError("Parameter 'N_exact' must be between 0 and 'S'.")

    # Define file paths
    G_sim_path = os.path.join(workdir, "G_sim.h5")
    k_s_path = os.path.join(workdir, "k_s.npy")
    G_ss_path = os.path.join(workdir, "G_ss.h5")
    P_ss_obj_path = os.path.join(workdir, "P_ss_obj.h5")
    P_0_path = os.path.join(workdir, "P_0.npy")
    theta_planck_path = os.path.join(workdir, "theta_planck.npy")
    Pbins_path = os.path.join(workdir, "Pbins.npy")
    Pbins_bnd_path = os.path.join(workdir, "Pbins_bnd.npy")

    # Compute or load the full Fourier grid
    if not os.path.exists(G_sim_path) or force:
        logger.info("Computing Fourier grid...")
        G_sim = FourierGrid(L, L, L, size, size, size)
        G_sim.write(G_sim_path)
        logger.info("Computing Fourier grid done.")
    else:
        logger.info("Loading Fourier grid.")
        G_sim = FourierGrid.read(G_sim_path)

    # Determine minimum and maximum k values
    if minval is None:
        minval = np.min(G_sim.k_modes[G_sim.k_modes != 0])
    if maxval is None:
        maxval = np.pi * size / L  # 1D Nyquist frequency

    # Compute or load support wavenumbers for the input power spectrum
    if not os.path.exists(k_s_path) or force:
        logger.diagnostic("Computing input power spectrum support wavenumbers...")
        k_s = np.zeros(S)
        sorted_knorms = np.sort(G_sim.k_modes.flatten())
        unique_indices = np.unique(np.round(sorted_knorms, 5), return_index=True)[1]
        sorted_knorms_corrected = sorted_knorms[unique_indices]
        k_s[:N_exact] = sorted_knorms_corrected[1 : N_exact + 1]
        k_s_max = get_k_max(L, size)
        k_s[N_exact:] = np.logspace(
            np.log10(sorted_knorms_corrected[N_exact]),
            np.log10(k_s_max),
            S - N_exact + 1,
        )[1:]
        np.save(k_s_path, k_s)
        logger.diagnostic("Computing input power spectrum support wavenumbers done.")
    else:
        logger.diagnostic("Loading input power spectrum support wavenumbers.")
        try:
            k_s = np.load(k_s_path)
        except (IOError, FileNotFoundError) as e:
            logger.error(f"Failed to load k_s from {k_s_path}: {e}")
            raise

    # Initialise Pbins
    Pbins_left_bnds_init = np.logspace(
        np.log10(minval), np.log10(maxval), Pinit + 1, dtype=np.float32
    )
    Pbins_left_bnds_init = Pbins_left_bnds_init[:-1]

    # Compute or load Fourier grid for the summaries
    if not os.path.exists(G_ss_path) or force:
        G_ss = FourierGrid(
            L,
            L,
            L,
            size,
            size,
            size,
            k_modes=Pbins_left_bnds_init,
            kmax=maxval,
            trim_bins=True,
            trim_threshold=trim_threshold,
        )
        G_ss.write(G_ss_path)
    else:
        G_ss = FourierGrid.read(G_ss_path)
    P = G_ss.NUM_MODES

    # Compute or load Pbins and Pbins_bnd
    if not os.path.exists(Pbins_path) or not os.path.exists(Pbins_bnd_path) or force:
        k_ss_max_offset = Pbins_left_bnds_init[-1] - Pbins_left_bnds_init[-2]
        logger.diagnostic(f"k_ss_max_offset: {k_ss_max_offset:.5f}")
        Pbins_bnd = G_ss.k_modes
        Pbins_bnd = np.concatenate([Pbins_bnd, [Pbins_bnd[-1] + k_ss_max_offset]])
        Pbins = (Pbins_bnd[1:] + Pbins_bnd[:-1]) / 2
        np.save(Pbins_path, Pbins)
        np.save(Pbins_bnd_path, Pbins_bnd)
    else:
        try:
            Pbins = np.load(Pbins_path)
            Pbins_bnd = np.load(Pbins_bnd_path)
        except (IOError, FileNotFoundError) as e:
            logger.error(f"Failed to load Pbins or Pbins_bnd: {e}")
            raise

    # Compute or load BBKS spectrum for normalisation
    if not os.path.exists(P_0_path) or force:
        P_0 = get_Pk(k_s, params_P0)
        np.save(P_0_path, P_0)
    else:
        try:
            P_0 = np.load(P_0_path)
        except (IOError, FileNotFoundError) as e:
            logger.error(f"Failed to load P_0 from {P_0_path}: {e}")
            raise

    if not os.path.exists(P_ss_obj_path) or force:
        P_0_ss = get_Pk(G_ss.k_modes, params_P0)
        P_ss_obj = PowerSpectrum.from_FourierGrid(G_ss, powerspectrum=P_0_ss, cosmo=params_P0)
        P_ss_obj.write(P_ss_obj_path)
    else:
        P_ss_obj = PowerSpectrum.read(P_ss_obj_path)

    # Compute or load Planck power spectrum
    if not os.path.exists(theta_planck_path) or force:
        planck_Pk = get_Pk(k_s, params_planck)
        np.save(theta_planck_path, planck_Pk)
    else:
        try:
            planck_Pk = np.load(theta_planck_path)
        except (IOError, FileNotFoundError) as e:
            logger.error(f"Failed to load theta_planck from {theta_planck_path}: {e}")
            raise

    # Clean up
    del G_sim, G_ss, P_ss_obj, Pbins_left_bnds_init
    gc.collect()

    return ModelSetup(
        size,
        L,
        P,
        S,
        G_sim_path,
        G_ss_path,
        Pbins_bnd,
        Pbins,
        k_s,
        P_ss_obj_path,
        P_0,
        planck_Pk,
    )


def compute_alpha_cv(
    workdir: str,
    k_s: np.ndarray,
    size: int,
    L: float,
    window_fct_path: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Compute the cosmic variance parameter alpha_cv.

    Parameters
    ----------
    workdir : str
        Directory where the results will be stored.
    k_s : np.ndarray
        Support wavenumbers.
    size : int
        Number of elements in each direction of the box.
    L : float
        Comoving length of the box in Mpc/h.
    window_fct_path : str, optional
        Path to the window function file.
    force : bool
        If True, forces recomputation of the inputs.

    """
    from scipy.optimize import curve_fit

    alpha_cv_path = os.path.join(workdir, "alpha_cv.npy")
    alpha_cv_eff_path = os.path.join(workdir, "alpha_cv_eff.npy")

    if not os.path.exists(alpha_cv_path) or force:
        logger.info("Computing cosmic variance alpha_cv...")
        k_s_bnd = np.concatenate([k_s, [np.inf]])

        G_sim = FourierGrid.read(os.path.join(workdir, "G_sim.h5")).k_modes.flatten()
        knorms = np.sort(G_sim)

        Nks, _ = np.histogram(knorms, bins=k_s_bnd)

        del knorms, G_sim

        nyquist_frequency = np.pi * size / L
        idx_nyquist = np.searchsorted(k_s, nyquist_frequency)

        def cubic_func(x, a):
            return a * x**3

        try:
            popt, _ = curve_fit(cubic_func, k_s[:idx_nyquist], Nks[:idx_nyquist])
        except RuntimeError as e:
            logger.error(f"Curve fitting failed: {e}")
            raise

        alpha_cv = np.sqrt(1 / popt[0])
        np.save(alpha_cv_path, alpha_cv)
        logger.info(f"Computing cosmic variance alpha_cv done. alpha_cv = {alpha_cv}")

        if window_fct_path is not None:
            # Compute alpha_cv with approximate correction for the effective volume
            nnz = 0
            with File(window_fct_path, "r") as f:
                for ipop in range(3):
                    mask = f["select_fct"][:][ipop]
                    nnz += np.sum(mask)
            nnz_size = nnz ** (1 / 3.0)  # Side length of a cube containing nnz voxels
            eff_L = nnz_size * L / size

            alpha_cv_eff = alpha_cv * (L / eff_L) ** 1.5
            logger.info(f"Effective length: {eff_L * 1e-3} Gpc/h")
            logger.info(f"Effective volume: {(eff_L * 1e-3) ** 3} (Gpc/h)^3")
            logger.info(f"alpha_cv_eff = {alpha_cv_eff}")
            np.save(alpha_cv_eff_path, alpha_cv_eff)

    gc.collect()
