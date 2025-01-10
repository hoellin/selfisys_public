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

"""Tools to define normalisation constants for the hidden box."""

import os
import numpy as np
from typing import Tuple, Dict, Any

from selfisys.global_parameters import MIN_K_NORMALISATION
from selfisys.hiddenbox import HiddenBox


def worker_normalisation(
    hidden_box: HiddenBox,
    params: Tuple[Dict[str, Any], list, list, bool],
) -> np.ndarray:
    """Worker function to compute the normalisation constants,
    compatible with Python multiprocessing.

    Parameters
    ----------
    hidden_box : HiddenBox
        Instance of the HiddenBox class.
    params : tuple
        A tuple containing (cosmo, seedphase, seednoise, force).

    Returns
    -------
    phi : ndarray
        Computed summary statistics.
    """
    (
        cosmo,
        seedphase,
        seednoise,
        force,
    ) = params
    name = (
        "norm"
        + "__"
        + "_".join([str(int(s)) for s in seedphase])
        + "__"
        + "_".join([str(int(s)) for s in seednoise])
    )

    if hidden_box.verbosity > 1:
        hidden_box._PrintMessage(1, "Running simulation...")
        hidden_box._indent()
        phi = hidden_box.make_data(
            cosmo,
            name,
            seedphase,
            seednoise,
            force,
            force,
            force,
            force,
        )
        hidden_box._unindent()
    elif hidden_box.verbosity > 0:
        from selfisys.utils.low_level import (
            stdout_redirector,
        )
        from io import BytesIO

        f = BytesIO()
        with stdout_redirector(f):
            phi = hidden_box.make_data(
                cosmo,
                name,
                seedphase,
                seednoise,
                force,
                force,
                force,
                force,
            )
        f.close()
    else:
        from selfisys.utils.low_level import (
            stdout_redirector,
            stderr_redirector,
        )
        from io import BytesIO

        f = BytesIO()
        g = BytesIO()
        with stdout_redirector(f), stderr_redirector(g):
            phi = hidden_box.make_data(
                cosmo,
                name,
                seedphase,
                seednoise,
                force,
                force,
                force,
                force,
            )
        f.close()
        g.close()

    return phi


def worker_normalisation_wrapper(args):
    """Wrapper function for the worker_normalisation function.

    Parameters
    ----------
    args : tuple
        A tuple containing (hidden_box, params).

    Returns
    -------
    phi : ndarray
        Computed summary statistics.
    """
    hidden_box, params = args
    return worker_normalisation(hidden_box, params)


def worker_normalisation_public(
    hidden_box,
    cosmo: Dict[str, Any],
    N: int,
    i: int,
):
    """Run the i-th simulation required to compute the normalisation
    constants.

    Parameters
    ----------
    hidden_box : HiddenBox
        Instance of the HiddenBox class.
    cosmo : dict
        Cosmological and some infrastructure parameters.
    N : int
        Total number of realisations required.
    i : int
        Index of the simulation to be computed.
    """
    params = (
        cosmo,
        [
            i,
            hidden_box._HiddenBox__global_seednorm,
        ],
        [
            i + N,
            hidden_box._HiddenBox__global_seednorm,
        ],
        False,
    )
    worker_normalisation(hidden_box, params)


def define_normalisation(
    hidden_box: HiddenBox,
    Pbins: np.ndarray,
    cosmo: Dict[str, Any],
    N: int,
    min_k_norma: float = MIN_K_NORMALISATION,
    npar: int = 1,
    force: bool = False,
) -> np.ndarray:
    """Define the normalisation constants for the HiddenBox instance.

    Parameters
    ----------
    hidden_box : HiddenBox
        Instance of the HiddenBox class.
    Pbins : ndarray
        Array of P bin values.
    cosmo : dict
        Cosmological and infrastructure parameters.
    N : int
        Number of realisations required.
    min_k_norma : float, optional
        Minimum k value to compute the normalisation constants.
    npar : int, optional
        Number of parallel processes to use. Default is 1.
    force : bool, optional
        If True, force recomputation. Default is False.

    Returns
    -------
    norm_csts : ndarray
        Normalisation constants for the HiddenBox instance.
    """
    import tqdm.auto as tqdm
    from multiprocessing import Pool

    hidden_box._PrintMessage(
        0,
        "Defining normalisation constants...",
    )
    hidden_box._indent()
    indices = np.where(Pbins > min_k_norma)
    tasks = [
        (
            hidden_box,
            (
                cosmo,
                [
                    i,
                    hidden_box._HiddenBox__global_seednorm,
                ],
                [
                    i + N,
                    hidden_box._HiddenBox__global_seednorm,
                ],
                force,
            ),
        )
        for i in range(N)
    ]

    ncors = os.cpu_count()
    nprocs = min(npar, ncors)

    norm_csts_list = np.zeros((hidden_box._Npop, N))
    if npar > 1:
        with Pool(nprocs) as p:
            for j, val in enumerate(
                tqdm.tqdm(
                    p.imap(
                        worker_normalisation_wrapper,
                        tasks,
                    ),
                    total=N,
                )
            ):
                norm_csts_list[:, j] = np.array(
                    [
                        np.mean(
                            val[i * hidden_box.Psingle : (i + 1) * hidden_box.Psingle][indices]
                        )
                        for i in range(hidden_box._Npop)
                    ]
                )
    else:
        for j, val in enumerate(
            tqdm.tqdm(
                map(
                    worker_normalisation_wrapper,
                    tasks,
                ),
                total=N,
            )
        ):
            val = np.array(val)
            norm_csts_list[:, j] = np.array(
                [
                    np.mean(val[i * hidden_box.Psingle : (i + 1) * hidden_box.Psingle][indices])
                    for i in range(hidden_box._Npop)
                ]
            )
    norm_csts = np.mean(norm_csts_list, axis=1)
    hidden_box._unindent()
    hidden_box._PrintMessage(
        0,
        "Defining normalisation constants done.",
    )

    return norm_csts
