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

"""Helper functions to handle paths and file names.
"""


import os


def _get_prefix(prefix_mocks: str, suffix: str, sim_id=None, d=None, p=None) -> str:
    """
    Get file prefix.

    Parameters
    ----------
    prefix_mocks : str | None
        Prefix for the mock data files. If None, defaults to base
        suffix.
    suffix : str
        Base suffix for the file name (e.g., "mocks" or "g").
    sim_id : int, optional
        Simulation ID. Used if d and p are not provided.
    d : int, optional
        Direction index.
    p : int, optional
        Simulation index. If both d and p are provided, they take
        precedence over sim_id.

    Returns
    -------
    str
        Formatted file name string.
    """
    prefix = f"{prefix_mocks}_{suffix}" if prefix_mocks else suffix
    if d is not None and p is not None:
        return f"{prefix}_d{d}_p{p}.h5"
    return f"{prefix}_{sim_id}.h5"


def get_file_names(
    fsimdir: str,
    sim_id: int,
    sim_params: str,
    TimeSteps: list[int],
    prefix_mocks: str,
    gravity_on: bool = True,
    return_g: bool = False,
) -> dict:
    """
    Generate file paths for a given simulation ID and parameters.

    Parameters
    ----------
    fsimdir : str
        Path to the simulation directory.
    sim_id : int
        Simulation ID.
    sim_params : str
        Simulation parameters.
    TimeSteps : list of int
        List of time steps.
    prefix_mocks : str | None
        Prefix for mock data files. If None, defaults to "mocks".
    gravity_on : bool, optional
        Whether gravity is active. Default is True.
    return_g : bool, optional
        If True, return the file name for the observed galaxy field.
        Default is False.

    Returns
    -------
    dict
        Dictionary containing simulation inputs / outputs file paths.
    """
    datadir = os.path.join(fsimdir, "data")
    names = {
        "fname_cosmo": os.path.join(datadir, f"input_cosmo_{sim_id}.json"),
        "fname_power_spectrum": os.path.join(datadir, f"input_power_{sim_id}.h5"),
        "fname_outputinitialdensity": os.path.join(datadir, f"initial_density_{sim_id}.h5"),
        "fname_mocks": os.path.join(datadir, _get_prefix(prefix_mocks, "mocks", sim_id)),
        "fname_g": (
            os.path.join(datadir, _get_prefix(prefix_mocks, "g", sim_id)) if return_g else None
        ),
        "fname_simparfile": None,
        "fname_whitenoise": None,
        "seedname_whitenoise": None,
        "fnames_outputLPTdensity": None,
        "fnames_outputrealspacedensity": None,
        "fnames_outputdensity": None,
        "fname_simlogs": None,
    }

    if gravity_on:
        names.update(
            {
                "fname_simparfile": os.path.join(datadir, f"sim_{sim_id}"),
                "fname_whitenoise": os.path.join(
                    datadir, f"initial_density_white_noise_{sim_id}.h5"
                ),
                "seedname_whitenoise": os.path.join(datadir, f"initial_density_wn_{sim_id}_seed"),
                "fnames_outputLPTdensity": os.path.join(datadir, f"output_density_{sim_id}.h5"),
                "fname_simlogs": os.path.join(datadir, f"logs_sim_{sim_id}.txt"),
            }
        )
        if sim_params.startswith(("split", "custom")):
            names["fnames_outputdensity"] = [
                os.path.join(datadir, f"output_density_{sim_id}_{i}.h5") for i in TimeSteps[::-1]
            ]
            names["fnames_outputrealspacedensity"] = [
                os.path.join(datadir, f"output_realdensity_{sim_id}_{i}.h5")
                for i in TimeSteps[::-1]
            ]
        else:
            names["fnames_outputdensity"] = [os.path.join(datadir, f"output_density_{sim_id}.h5")]
            names["fnames_outputrealspacedensity"] = [
                os.path.join(datadir, f"output_realdensity_{sim_id}.h5")
            ]

    return names


def file_names_evaluate(
    simdir: str,
    sd: str,
    d: int,
    i: int,
    sim_params: str,
    TimeSteps: list[int],
    prefix_mocks: str,
    abc: bool = False,
    gravity_on: bool = True,
) -> dict:
    """
    Generate file paths for the given simulation id and parameters.

    Parameters
    ----------
    simdir : str
        Path to the simulation directory.
    sd : str
        Path to the simulation directory for the given direction.
    d : int
        Direction index (-1 for mock data, 0 for the expansion point, or
        1 to S).
    i : int
        Simulation index.
    sim_params : str
        Simulation parameters.
    TimeSteps : list of int
        List of time steps.
    prefix_mocks : str | None
        Prefix for mock data files. If None, defaults to "mocks".
    abc : bool, optional
        If True, appends the ABC index to the white noise path.
    gravity_on : bool, optional
        Whether gravity is active. Default is True.

    Returns
    -------
    dict
        Dictionary containing simulation inputs / outputs file paths.
    """
    names = {
        "fname_power_spectrum": os.path.join(sd, f"input_power_d{d}.h5"),
        "fname_outputinitialdensity": os.path.join(sd, f"initial_density_d{d}_p{i}.h5"),
        "fname_mocks": os.path.join(sd, _get_prefix(prefix_mocks, "mocks", d=d, p=i)),
        "fname_simlogs": os.path.join(sd, f"logs_sim_d{d}_p{i}.txt"),
        "fname_simparfile": None,
        "fname_whitenoise": None,
        "seedname_whitenoise": None,
        "fnames_outputLPTdensity": None,
        "fnames_outputrealspacedensity": None,
        "fnames_outputdensity": None,
        "fname_g": None,
    }

    if gravity_on:
        dir_wn = os.path.join(simdir, "..", "wn") if not abc else os.path.join(simdir, "wn", abc)
        names.update(
            {
                "fname_simparfile": os.path.join(sd, f"sim_d{d}_p{i}"),
                "fname_whitenoise": os.path.join(dir_wn, f"initial_density_white_p{i}.h5"),
                "seedname_whitenoise": os.path.join(dir_wn, f"initial_density_white_p{i}"),
                "fnames_outputLPTdensity": os.path.join(sd, f"output_density_d{d}_p{i}.h5"),
            }
        )
        if sim_params.startswith(("split", "custom")):
            names["fnames_outputrealspacedensity"] = [
                os.path.join(sd, f"output_realdensity_d{d}_p{i}_{j}.h5") for j in TimeSteps[::-1]
            ]
            names["fnames_outputdensity"] = [
                os.path.join(sd, f"output_density_d{d}_p{i}_{j}.h5") for j in TimeSteps[::-1]
            ]
        else:
            names["fnames_outputrealspacedensity"] = [
                os.path.join(sd, f"output_realdensity_d{d}_p{i}.h5")
            ]
            names["fnames_outputdensity"] = [os.path.join(sd, f"output_density_d{d}_p{i}.h5")]

    return names
