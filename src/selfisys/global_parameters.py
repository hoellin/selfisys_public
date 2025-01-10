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

"""Global parameters for this project."""

import os
from pathlib import Path
import numpy as np

WHICH_SPECTRUM = "class"  # available options are "eh" and "class"

# Load global paths from environment variables
ROOT_PATH = os.getenv("SELFISYS_ROOT_PATH")
if ROOT_PATH is None:
    raise EnvironmentError("Please set the 'SELFISYS_ROOT_PATH' environment variable.")
OUTPUT_PATH = os.getenv("SELFISYS_OUTPUT_PATH")
if OUTPUT_PATH is None:
    raise EnvironmentError("Please set the 'SELFISYS_OUTPUT_PATH' environment variable.")

# Default verbose level
# 0: errors only, 1: info, 2: warnings+, 3: all diagnostics, 4+: debug
DEFAULT_VERBOSE_LEVEL = 2

# Baseline seeds for reproducibility
BASELINE_SEEDNORM = 100030898
BASELINE_SEEDNOISE = 200030898
BASELINE_SEEDPHASE = 300030898
SEEDPHASE_OBS = 100030896
SEEDNOISE_OBS = 100030897

# Fiducial cosmological parameters
h_planck = 0.6766
Omega_b_planck = 0.02242 / h_planck**2
Omega_m_planck = 0.3111
nS_planck = 0.9665
sigma8_planck = 0.8102

planck_mean = np.array([h_planck, Omega_b_planck, Omega_m_planck, nS_planck, sigma8_planck])
planck_cov = np.diag(np.array([0.0042, 0.00030, 0.0056, 0.0038, 0.0060]) ** 2)

# Mock unknown ground truth parameters for consistency checks
h_obs = 0.679187146124996
Omega_b_obs = 0.0487023481098232
Omega_m_obs = 0.3053714257403574
nS_obs = 0.9638467785003454
sigma8_obs = 0.8210464735135183

omegas_gt = np.array([h_obs, Omega_b_obs, Omega_m_obs, nS_obs, sigma8_obs])

# Mapping from cosmological parameter names to corresponding indices
cosmo_params_names = [r"$h$", r"$\Omega_b$", r"$\Omega_m$", r"$n_S$", r"$\sigma_8$"]
cosmo_params_name_to_idx = {"h": 0, "Omega_b": 1, "Omega_m": 2, "n_s": 3, "sigma8": 4}

# Minimum k value used in the normalisation of the summaries
MIN_K_NORMALISATION = 4e-2

params_planck_kmax_missing = {
    "h": h_planck,
    "Omega_r": 0.0,
    "Omega_q": 1.0 - Omega_m_planck,
    "Omega_b": Omega_b_planck,
    "Omega_m": Omega_m_planck,
    "m_ncdm": 0.0,
    "Omega_k": 0.0,
    "tau_reio": 0.066,
    "n_s": nS_planck,
    "sigma8": sigma8_planck,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
    "WhichSpectrum": WHICH_SPECTRUM,
}

params_BBKS_kmax_missing = {
    "h": h_planck,
    "Omega_r": 0.0,
    "Omega_q": 1.0 - Omega_m_planck,
    "Omega_b": Omega_b_planck,
    "Omega_m": Omega_m_planck,
    "m_ncdm": 0.0,
    "Omega_k": 0.0,
    "tau_reio": 0.066,
    "n_s": nS_planck,
    "sigma8": sigma8_planck,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
    "WhichSpectrum": "BBKS",
}

params_cosmo_obs_kmax_missing = {
    "h": h_obs,
    "Omega_r": 0.0,
    "Omega_q": 1.0 - Omega_m_obs,
    "Omega_b": Omega_b_obs,
    "Omega_m": Omega_m_obs,
    "m_ncdm": 0.0,
    "Omega_k": 0.0,
    "tau_reio": 0.066,
    "n_s": nS_obs,
    "sigma8": sigma8_obs,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
    "WhichSpectrum": WHICH_SPECTRUM,
}

# Default hyperparameters for the wiggle-less prior from [leclercq2019primordial].
THETA_NORM_GUESS = 0.05
K_CORR_GUESS = 0.01

# Base ID for the observations
BASEID_OBS = "obs"
