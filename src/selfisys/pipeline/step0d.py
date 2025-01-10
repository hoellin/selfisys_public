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
Step 0d of the SelfiSys pipeline.

Generate the observations using the ground truth cosmology.
"""

import pickle
import gc
from os.path import exists
import numpy as np

from selfisys.utils.parser import (
    ArgumentParser,
    none_or_bool_or_str,
    bool_sh,
    joinstrs,
    safe_npload,
)
from selfisys.global_parameters import *
from selfisys.utils.tools import get_k_max
from selfisys.hiddenbox import HiddenBox
from selfisys.utils.logger import getCustomLogger, INDENT, UNINDENT

logger = getCustomLogger(__name__)

parser = ArgumentParser(
    description=(
        "Step 0d of the SelfiSys pipeline. "
        "Generate the observations using the ground truth cosmology."
    )
)
parser.add_argument("--wd", type=str, help="Absolute path of the working directory.")
parser.add_argument(
    "--prefix_mocks",
    type=none_or_bool_or_str,
    default=None,
    help="Prefix for the mock files.",
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
    "--name_obs",
    type=none_or_bool_or_str,
    default=None,
    help="Prefix for the observation files. If None, uses default name. "
    "Can be used for different data vectors.",
)
parser.add_argument(
    "--reset_window_function",
    type=bool_sh,
    default=False,
    help="Reset the window function.",
)
parser.add_argument(
    "--neglect_lightcone",
    type=bool_sh,
    default=False,
    help="Neglect lightcone effects even if snapshots at multiple redshifts are available.",
)
parser.add_argument(
    "--force_obs",
    type=bool_sh,
    default=False,
    help="Recompute the observations (e.g., to try a new cosmology).",
)
parser.add_argument(
    "--copy_obs_from",
    type=none_or_bool_or_str,
    default=None,
    help="Copy the observations from another project.",
)
parser.add_argument(
    "--copy_fields",
    type=bool_sh,
    default=False,
    help="Copy the fields from another project.",
)
parser.add_argument(
    "--save_g",
    type=bool_sh,
    default=False,
    help="Save the observed fields (g).",
)

args = parser.parse_args()

wd = args.wd
survey_mask_path = args.survey_mask_path
effective_volume = args.effective_volume
prefix_mocks = args.prefix_mocks
name_obs = "_" + args.name_obs if args.name_obs is not None else None
local_mask_prefix = args.name_obs if args.name_obs is not None else None
reset_window_function = args.reset_window_function
neglect_lightcone = args.neglect_lightcone
force_obs = args.force_obs
copy_obs_from = args.copy_obs_from
copy_fields = args.copy_fields
save_g = args.save_g

if copy_obs_from is not None and name_obs is None:
    raise ValueError(
        "If you want to copy the observations from another project, "
        "you must specify a name for the observation files."
    )
if copy_fields and copy_obs_from is None:
    raise ValueError(
        "If you want to copy the fields from another project, "
        "you must specify the project to copy from."
    )

if __name__ == "__main__":
    from pysbmy.timestepping import read_timestepping

    try:
        logger.info("Starting Step 0d of the SelfiSys pipeline.")
        logger.info("Setting up main parameters...")

        modeldir = wd + "model/"
        datadir = wd + "data/"

        # Copy the observations from another directory if specified
        if copy_obs_from is not None:
            from glob import glob
            import shutil

            logger.info("Copying observations from: %s", copy_obs_from)
            INDENT()
            theta_gt_path = joinstrs([copy_obs_from, "model/theta_gt", name_obs, ".npy"])
            phi_obs_path = joinstrs([copy_obs_from, "model/phi_obs", name_obs, ".npy"])
            field_prefix = joinstrs([copy_obs_from, "data/output_density_obs", name_obs, "_"])

            if not exists(theta_gt_path):
                raise FileNotFoundError(f"{theta_gt_path} not found. Check the path.")
            if not exists(phi_obs_path):
                raise FileNotFoundError(f"{phi_obs_path} not found. Check the path.")
            if len(glob(field_prefix + "*")) == 0:
                raise FileNotFoundError(
                    f"No files starting with {field_prefix} found. Check the path."
                )

            logger.diagnostic("Copying theta_gt and phi_obs files...")
            shutil.copy(theta_gt_path, f"{modeldir}theta_gt{name_obs}.npy")
            shutil.copy(phi_obs_path, f"{modeldir}phi_obs{name_obs}.npy")
            logger.diagnostic("Copying theta_gt and phi_obs files done.")

            if copy_fields:
                logger.diagnostic("Copying full fields...")
                for file in glob(field_prefix + "*"):
                    shutil.copy(file, datadir)
                logger.diagnostic("Copying full fields done.")
            UNINDENT()
        else:
            # Generating new observations
            if prefix_mocks is not None:
                modeldir_refined = modeldir + prefix_mocks + "/"
            else:
                modeldir_refined = modeldir

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
            sim_params_base = other_params["sim_params"]

            isstd = sim_params_base[:3] == "std"
            if isstd and copy_obs_from is None:
                # Workaround so that observations can be computed
                sim_params_base = sim_params_base + "0"
            sim_params = sim_params_base + BASEID_OBS

            radial_selection = safe_npload(modeldir + "radial_selection.npy")
            selection_params = np.load(modeldir + "selection_params.npy")
            lin_bias = np.load(modeldir + "lin_bias.npy")
            Npop = len(lin_bias) if isinstance(lin_bias, np.ndarray) else 1
            obs_density = safe_npload(modeldir + "obs_density.npy")
            noise = np.load(modeldir + "noise.npy")

            k_max = get_k_max(L, size)  # k_max in h/Mpc
            params_cosmo_obs = params_cosmo_obs_kmax_missing.copy()
            params_cosmo_obs["k_max"] = k_max

            logger.diagnostic("Loading main parameters.")
            Pbins_bnd = np.load(modeldir + "Pbins_bnd.npy")
            Pbins = np.load(modeldir + "Pbins.npy")
            k_s = np.load(modeldir + "k_s.npy")
            P_0 = np.load(modeldir + "P_0.npy")

            def theta2P(theta):
                return theta * P_0

            # Setup time-stepping if needed
            if not isstd:
                logger.info("Setting up the time-stepping for non-standard approach...")
                nsteps = [
                    round((aa[i + 1] - aa[i]) / (aa[-1] - aa[0]) * total_steps)
                    for i in range(len(aa) - 1)
                ]
                if sum(nsteps) != total_steps:
                    nsteps[nsteps.index(max(nsteps))] += total_steps - sum(nsteps)
                indices_steps_cumul = list(np.cumsum(nsteps) - 1)
                merged_path = modeldir + "merged.h5"
                TS_merged = read_timestepping(merged_path)

                if sim_params[:6] in ["custom", "nograv"]:
                    TimeStepDistribution = merged_path
                    eff_redshifts = 1 / aa[-1] - 1
                else:
                    raise NotImplementedError("Time-stepping strategy not yet implemented.")
                logger.info("Setting up the time-stepping for non-standard approach done.")
            else:
                TimeStepDistribution = None
                eff_redshifts = None
                indices_steps_cumul = None

            logger.info("Instantiating the HiddenBox...")
            # Load normalisation constants
            if not exists(modeldir + "norm_csts.npy"):
                raise ValueError("Normalisation constants not found.")
            norm_csts = np.load(modeldir + "norm_csts.npy")

            BB_selfi = HiddenBox(
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
                modeldir=modeldir_refined,
                noise_std=noise,
                radial_selection=radial_selection,
                selection_params=selection_params,
                observed_density=obs_density,
                linear_bias=lin_bias,
                norm_csts=norm_csts,
                survey_mask_path=survey_mask_path,
                local_mask_prefix=local_mask_prefix,
                sim_params=sim_params,
                TimeStepDistribution=TimeStepDistribution,
                TimeSteps=indices_steps_cumul,
                eff_redshifts=eff_redshifts,
                seedphase=BASELINE_SEEDPHASE,
                seednoise=BASELINE_SEEDNOISE,
                fixnoise=False,
                seednorm=BASELINE_SEEDNORM,
                reset=reset_window_function,
                save_frequency=5,
            )
            logger.info("Instantiating the HiddenBox done.")

            # Generate the ground truth spectrum
            if force_obs or not exists(joinstrs([modeldir, "theta_gt", name_obs, ".npy"])):
                logger.info("Generating ground truth spectrum for Step 0d.")
                from pysbmy.power import get_Pk

                theta_gt = get_Pk(k_s, params_cosmo_obs)
                np.save(joinstrs([modeldir, "theta_gt", name_obs]), theta_gt)
                logger.info("Generating ground truth spectrum for Step 0d done.")

            logger.info("Generating observations...")
            phi_obs_path = joinstrs([modeldir, "phi_obs", name_obs, ".npy"])
            if not exists(phi_obs_path) or force_obs:
                if neglect_lightcone:
                    BB_selfi.update(_force_neglect_lightcone=True)
                d_obs = -1
                BB_selfi.switch_recompute_pool()
                res = BB_selfi.make_data(
                    cosmo=params_cosmo_obs,
                    id=joinstrs([BASEID_OBS, name_obs]),
                    seedphase=SEEDPHASE_OBS,
                    seednoise=SEEDNOISE_OBS,
                    d=d_obs,
                    force_powerspectrum=force_obs,
                    force_parfiles=force_obs,
                    force_sim=force_obs,
                    force_cosmo=force_obs,
                    return_g=save_g,
                )
                BB_selfi.switch_recompute_pool()

                if save_g:
                    phi_obs, _ = res
                else:
                    phi_obs = res

                np.save(phi_obs_path, phi_obs)
            logger.info("Generating observations done.")

    except OSError as e:
        logger.error("File or directory access error during Step 0d: %s", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred in Step 0d: %s", str(e))
        raise RuntimeError("Step 0d failed.") from e
    finally:
        gc.collect()
        logger.info("Step 0d of the SelfiSys pipeline: done.")
