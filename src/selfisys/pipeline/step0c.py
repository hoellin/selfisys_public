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
Step 0c of the SelfiSys pipeline.

Compute the normalisation constants (based on the simulations performed
in step 0b using LPT or COLA) for the SelfiSys pipeline.
"""

import gc
import numpy as np

from selfisys.utils.parser import ArgumentParser, none_or_bool_or_str, bool_sh, safe_npload
from selfisys.global_parameters import *
from selfisys.utils.tools import get_k_max
from selfisys.utils.logger import getCustomLogger, INDENT, UNINDENT

logger = getCustomLogger(__name__)

parser = ArgumentParser(
    description=(
        "Step 0c of the SelfiSys pipeline. "
        "Compute the normalisation constants based on the simulations performed in step 0b."
    )
)
parser.add_argument("--wd", type=str, help="Absolute path of the working directory.")
parser.add_argument(
    "--npar_norm",
    type=int,
    help=(
        "Number of simulations to load in parallel when computing the summaries. "
        "Note that the overdensity fields were already computed at step 0b."
    ),
)
parser.add_argument(
    "--survey_mask_path",
    type=none_or_bool_or_str,
    default=None,
    help="Path to the survey mask for the well-specified model.",
)
parser.add_argument(
    "--effective_volume",
    type=bool_sh,
    default=False,
    help="Use the effective volume to compute alpha_cv.",
)
parser.add_argument(
    "--norm_csts_path",
    type=none_or_bool_or_str,
    default=None,
    help="Path to external normalisation constants. Mandatory for test_gravity=True.",
)
parser.add_argument(
    "--force",
    type=bool_sh,
    default=False,
    help="Force the recomputation of the mocks.",
)

args = parser.parse_args()

wd = args.wd
npar_norm = args.npar_norm
survey_mask_path = args.survey_mask_path
effective_volume = args.effective_volume
norm_csts_path = args.norm_csts_path
force = args.force

modeldir = wd + "model/"

# Consistency check: 'npar_norm' and 'norm_csts_path' are mutually exclusive
if not (npar_norm is None) ^ (norm_csts_path is None):
    raise ValueError("npar_norm and norm_csts_path are mutually exclusive.")

if __name__ == "__main__":
    try:
        # If the user normalisation constants are provided, load them
        if norm_csts_path is not None:
            INDENT()
            logger.info("Loading normalisation constants...")
            if not exists(norm_csts_path):
                raise ValueError("Normalisation constants not found.")
            else:
                norm_csts = np.load(norm_csts_path)
                np.save(modeldir + "norm_csts.npy", norm_csts)
                logger.info(
                    "External normalisation constants loaded and saved to model directory."
                )
            UNINDENT()
        else:
            # Otherwise, compute normalisation constants from simulation data
            from os.path import exists
            import pickle
            from pysbmy.timestepping import read_timestepping
            from selfisys.hiddenbox import HiddenBox
            from selfisys.normalise_hb import define_normalisation

            logger.info("Loading main parameters from 'other_params.pkl'...")
            with open(modeldir + "other_params.pkl", "rb") as f:
                other_params = pickle.load(f)

            size = other_params["size"]
            Np0 = other_params["Np0"]
            Npm0 = other_params["Npm0"]
            L = other_params["L"]
            S = other_params["S"]
            total_steps = other_params["total_steps"]
            aa = other_params["aa"]
            P = other_params["P"]
            G_sim_path = other_params["G_sim_path"]
            G_ss_path = other_params["G_ss_path"]
            P_ss_obj_path = other_params["P_ss_obj_path"]
            Nnorm = other_params["Nnorm"]
            sim_params = other_params["sim_params"]

            isstd = sim_params[:3] == "std"

            # Load radial selection
            radial_selection = np.load(modeldir + "radial_selection.npy", allow_pickle=True)
            if radial_selection is None:
                radial_selection = None
            selection_params = np.load(modeldir + "selection_params.npy")
            lin_bias = np.load(modeldir + "lin_bias.npy")
            Npop = len(lin_bias) if isinstance(lin_bias, np.ndarray) else 1
            obs_density = safe_npload(modeldir + "obs_density.npy")
            noise = np.load(modeldir + "noise.npy")

            k_max = get_k_max(L, size)  # k_max in h/Mpc
            # Cosmology at the expansion point:
            params_planck = params_planck_kmax_missing.copy()
            params_planck["k_max"] = k_max

            Pbins_bnd = np.load(modeldir + "Pbins_bnd.npy")
            Pbins = np.load(modeldir + "Pbins.npy")
            k_s = np.load(modeldir + "k_s.npy")
            P_0 = np.load(modeldir + "P_0.npy")

            def theta2P(theta):
                return theta * P_0

            # Set up the merged time-stepping if needed
            if not isstd:
                logger.info("Setting up time-stepping...")
                nsteps = [
                    round((aa[i + 1] - aa[i]) / (aa[-1] - aa[0]) * total_steps)
                    for i in range(len(aa) - 1)
                ]
                if sum(nsteps) != total_steps:
                    nsteps[nsteps.index(max(nsteps))] += total_steps - sum(nsteps)
                indices_steps_cumul = list(np.cumsum(nsteps) - 1)
                merged_path = modeldir + "merged.h5"
                TS_merged = read_timestepping(merged_path)

                if sim_params.startswith("custom") or sim_params.startswith("nograv"):
                    TimeStepDistribution = merged_path
                    eff_redshifts = 1 / aa[-1] - 1
                else:
                    raise NotImplementedError("Time-stepping strategy not yet implemented.")
                logger.info("Setting up time-stepping done.")
            else:
                TimeStepDistribution = None
                eff_redshifts = None
                indices_steps_cumul = None

            logger.info("Instantiating the HiddenBox for normalisation constants...")
            HB_selfi = HiddenBox(
                k_s=k_s,
                P_ss_path=P_ss_obj_path,
                Pbins_bnd=Pbins_bnd,
                theta2P=theta2P,
                P=P * Npop,
                size=size,
                L=L,
                G_sim_path=G_sim_path,
                G_ss_path=G_ss_path,
                Np0=Np0,
                Npm0=Npm0,
                fsimdir=wd[:-1],
                noise_std=noise,
                radial_selection=radial_selection,
                selection_params=selection_params,
                observed_density=obs_density,
                linear_bias=lin_bias,
                norm_csts=None,
                survey_mask_path=survey_mask_path,
                local_mask_prefix=None,
                sim_params=sim_params,
                TimeStepDistribution=TimeStepDistribution,
                TimeSteps=indices_steps_cumul,
                eff_redshifts=eff_redshifts,
                seedphase=BASELINE_SEEDPHASE,
                seednoise=BASELINE_SEEDNOISE,
                fixnoise=False,
                seednorm=BASELINE_SEEDNORM,
                reset=False,
                save_frequency=5,
            )
            logger.info("Instantiating the HiddenBox for normalisation constants done.")

            # Compute normalisation constants
            if not exists(modeldir + "norm_csts.npy") or force:
                if force:
                    HB_selfi.switch_recompute_pool()
                norm_csts = define_normalisation(
                    HB_selfi,
                    Pbins,
                    params_planck,
                    Nnorm,
                    min_k_norma=MIN_K_NORMALISATION,
                    npar=1,
                    force=force,
                )
                if force:
                    HB_selfi.switch_recompute_pool()
                np.save(modeldir + "norm_csts.npy", norm_csts)
                logger.info("Normalisation constants computed and saved.")
            else:
                logger.info("Normalisation constants already exist, skipping re-computation.")
                norm_csts = np.load(modeldir + "norm_csts.npy")

            logger.info("Normalisation constants: %s", norm_csts)

    except OSError as e:
        logger.error("File or directory access error in step 0c: %s", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred in step 0c: %s", str(e))
        raise RuntimeError("Step 0c failed.") from e
    finally:
        gc.collect()
        logger.info("step 0c of the SelfiSys pipeline: done.")
