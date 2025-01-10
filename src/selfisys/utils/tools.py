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
Utilities for the SelfiSys package, including tools for cosmological
parameter handling, power spectrum computations, and prior sampling.
"""


def none_or_bool_or_str(value):
    """
    Convert string representations of None, True, and False to their
    respective Python objects; otherwise, return the input value.
    """
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def get_k_max(L, size):
    """
    Compute the maximum wavenumber for a given box size.

    Parameters
    ----------
    L : float
        Size of the box in Mpc/h.
    size : int
        Number of grid cells along each dimension.

    Returns
    -------
    float
        Maximum wavenumber in h/Mpc.
    """
    from numpy import pi, sqrt

    return int(1e3 * sqrt(3) * pi * size / L + 1) * 1e-3


def custom_stat(vec):
    """
    Compute a custom statistic for use with
    `scipy.stats.binned_statistic`.

    Assumes the data power spectrum is inverse-Gamma distributed (as in
    [jasche2010bayesian] and [leclercq2019primordial]). Returns "NaN"
    for vectors with insufficient elements, as expected by
    `scipy.stats.binned_statistic`.

    Parameters
    ----------
    vec : array-like
        Input vector for computation.

    Returns
    -------
    float or str
        Custom statistic or NaN if input is invalid.
    """
    if len(vec) <= 2 or sum(vec) == 0:
        return "NaN"
    return sum(vec) / (len(vec) - 2)


def cosmo_vector_to_Simbelmyne_dict(x, kmax=1.4):
    """
    Convert a vector of cosmological parameters into a dictionary
    compatible with `pysbmy`.

    Parameters
    ----------
    x : array-like
        Vector of cosmological parameters.
    kmax : float, optional
        Maximum wavenumber for the power spectrum computation.

    Returns
    -------
    dict
        Dictionary of cosmological parameters compatible with `pysbmy`.
    """
    from selfisys.global_parameters import WHICH_SPECTRUM

    return {
        "h": x[0],
        "Omega_r": 0.0,
        "Omega_q": 1.0 - x[2],
        "Omega_b": x[1],
        "Omega_m": x[2],
        "m_ncdm": 0.0,
        "Omega_k": 0.0,
        "tau_reio": 0.066,
        "n_s": x[3],
        "sigma8": x[4],
        "w0_fld": -1.0,
        "wa_fld": 0.0,
        "k_max": kmax,
        "WhichSpectrum": WHICH_SPECTRUM,
    }


def cosmo_vector_to_class_dict(x, lmax=2500, kmax=1.4):
    """
    Convert a vector of cosmological parameters into a dictionary
    compatible with `classy`.

    Parameters
    ----------
    x : array-like
        Vector of cosmological parameters.
    lmax : int, optional
        Maximum multipole for the power spectrum computation.
    kmax : float, optional
        Maximum wavenumber for the power spectrum computation.

    Returns
    -------
    dict
        Dictionary of cosmological parameters compatible with `classy`.
    """
    return {
        "output": "lCl mPk",
        "l_max_scalars": lmax,
        "lensing": "no",
        "N_ncdm": 0,
        "P_k_max_h/Mpc": kmax,
        "h": x[0],
        "Omega_b": x[1],
        "Omega_m": x[2],
        "n_s": x[3],
        "sigma8": x[4],
    }


def params_ids_to_Simbelmyne_dict(params_vals, params_ids, fixed, kmax):
    """
    Convert a list of cosmological parameters into a dictionary
    compatible with `pysbmy`.

    Fixed parameters remain unchanged unless overridden by
    `params_vals`.

    Parameters
    ----------
    params_vals : array-like
        Values of the parameters to be modified.
    params_ids : array-like
        Indices of the parameters to be modified.
    fixed : array-like
        Base values of the parameters.
    kmax : float
        Maximum wavenumber for the power spectrum computation.

    Returns
    -------
    dict
        Dictionary of cosmological parameters compatible with `pysbmy`.
    """
    from numpy import copy

    x = copy(fixed)
    x[params_ids] = params_vals
    return cosmo_vector_to_Simbelmyne_dict(x, kmax=kmax)


def get_summary(params_vals, params_ids, Omegas_fixed, bins, normalisation=None, kmax=1.4):
    """
    Compute the normalised power spectrum summary for a given parameter
    set.

    Parameters
    ----------
    params_vals : array-like
        Parameter values to update.
    params_ids : array-like
        Indices of the parameters to update.
    Omegas_fixed : array-like
        Fixed base values of parameters.
    bins : array-like
        Power spectrum bins.
    normalisation : float, optional
        Normalisation factor for the summary.
    kmax : float, optional
        Maximum wavenumber for power spectrum computation.

    Returns
    -------
    array
        Normalised power spectrum summary.
    """
    from pysbmy.power import get_Pk
    from numpy import array

    phi = get_Pk(bins, params_ids_to_Simbelmyne_dict(params_vals, params_ids, Omegas_fixed, kmax))

    return array(phi) / normalisation if normalisation else array(phi)


def summary_to_score(params_ids, omega0, F0, F0_inv, f0, dw_f0, C0_inv, phi):
    """
    Compute the Fisher score.

    Parameters
    ----------
    params_ids : array-like
        Indices of the parameters.
    omega0 : array-like
        Cosmological parameters at the expansion point.
    F0 : array-like
        Fisher information matrix.
    F0_inv : array-like
        Inverse Fisher information matrix.
    f0 : array-like
        Mean model at the expansion point.
    dw_f0 : array-like
        Derivative of the mean model.
    C0_inv : array-like
        Inverse covariance matrix.
    phi : array-like
        Observed summary.

    Returns
    -------
    array
        Fisher score.
    """
    return omega0[params_ids] + F0_inv @ dw_f0.T @ C0_inv @ (phi - f0)


def fisher_rao(Com, Com_obs, F0):
    """
    Compute the Fisher-Rao distance between two summaries.

    Parameters
    ----------
    Com : array-like
        Computed summary.
    Com_obs : array-like
        Observed summary.
    F0 : array-like
        Fisher information matrix.

    Returns
    -------
    float
        Fisher-Rao distance.
    """
    from numpy import sqrt

    diff = Com - Com_obs
    return sqrt(diff.T @ F0 @ diff)


def sample_omega_from_prior(nsample, omega_mean, omega_cov, params_ids, seed=None):
    """
    Sample cosmological parameters from a prior distribution.

    Ensures physical validity by clipping values to [eps, 1-eps].

    Parameters
    ----------
    nsample : int
        Number of samples to draw.
    omega_mean : array-like
        Prior mean vector.
    omega_cov : array-like
        Prior covariance matrix.
    params_ids : array-like
        Indices of the parameters to sample.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    array
        Sampled cosmological parameters.
    """
    from numpy import array, ix_, clip
    from numpy.random import default_rng

    if seed is None:
        raise ValueError("A seed value is mandatory.")

    rng = default_rng(seed)
    OO_unbounded = rng.multivariate_normal(
        array(omega_mean)[params_ids],
        array(omega_cov)[ix_(params_ids, params_ids)],
        nsample,
    )
    eps = 1e-5
    return clip(OO_unbounded, eps, 1 - eps)
