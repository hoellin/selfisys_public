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
Step 0a of the SELFI pipeline.

We use a blackbox forward data model with physics relying on COLA
simulations (using the Simbelmynë hierarchical probabilistic simulator).

This step generates the Simbelmynë parameter files needed for
normalising the blackbox and computes the white noise fields. At this
stage, the only simulation performed is to compute the ground truth
spectrum.
"""

import gc
import pickle
import numpy as np

from os.path import exists
from pathlib import Path

from selfisys.utils.parser import ArgumentParser, none_or_bool_or_str, bool_sh, intNone
from selfisys.global_parameters import *
from selfisys.setup_model import *
from selfisys.hiddenbox import HiddenBox
from selfisys.utils.tools import get_k_max
from selfisys.sbmy_interface import handle_time_stepping
from selfisys.utils.plot_utils import setup_plotting
from selfisys.normalise_hb import worker_normalisation_public
from selfisys.utils.logger import getCustomLogger, INDENT, UNINDENT

logger = getCustomLogger(__name__)

"""
Below is the core logic of step 0a.

Raises
------
OSError
    If file or directory access fails.
RuntimeError
    If unexpected issues occur (e.g., plotting or data generation
    failures).
"""

parser = ArgumentParser(
    description=(
        "Run the first step of the SelfiSys pipeline. "
        "Generates Simbelmynë parameter files for blackbox normalisation."
    )
)

parser.add_argument(
    "--wd_ext",
    type=str,
    help=(
        "Name of the working directory (relative to ROOT_PATH in "
        "`../global_parameters.py`), ending with a slash."
    ),
)
parser.add_argument(
    "--name",
    type=str,
    default="std",
    help=(
        "Suffix to the working directory for this run. "
        "White noise fields are shared between runs irrespective of name."
    ),
)
parser.add_argument(
    "--total_steps",
    type=int,
    default=None,
    help="Number of timesteps.",
)
parser.add_argument(
    "--aa",
    type=float,
    nargs="*",
    default=None,
    help="List of scale factors at which to synchronise kicks and drifts.",
)
parser.add_argument(
    "--size",
    type=int,
    default=512,
    help="Number of grid points in each direction.",
)
parser.add_argument(
    "--Np0",
    type=intNone,
    default=1024,
    help="Number of dark matter particles along each axis.",
)
parser.add_argument(
    "--Npm0",
    type=intNone,
    default=1024,
    help="Number of particle-mesh cells along each axis.",
)
parser.add_argument(
    "--L",
    type=int,
    default=3600,
    help="Size of the simulation box in Mpc/h.",
)
parser.add_argument(
    "--S",
    type=int,
    default=64,
    help="Number of support wavenumbers for the initial matter power spectrum.",
)
parser.add_argument(
    "--Pinit",
    type=int,
    default=50,
    help=(
        "Max number of bins for summaries. Actual count may be smaller since it is automatically "
        "tuned to ensure that each bin contains a sufficient number of modes."
    ),
)
parser.add_argument(
    "--Nnorm",
    type=int,
    default=10,
    help="Number of simulations for summary normalisation.",
)
parser.add_argument(
    "--Ne",
    type=int,
    default=300,
    help="Number of simulations at the expansion point for blackbox linearisation.",
)
parser.add_argument(
    "--Ns",
    type=int,
    default=10,
    help="Number of simulations for each gradient component at the expansion point.",
)
parser.add_argument(
    "--Delta_theta",
    type=float,
    default=1e-2,
    help="Finite difference step size for gradient computation.",
)
parser.add_argument(
    "--OUTDIR",
    type=str,
    help="Absolute path to the output directory.",
)
parser.add_argument(
    "--prior",
    type=str,
    default="planck2018",
    help='Prior type (e.g. "selfi2019", "planck2018", "planck2018_cv").',
)
parser.add_argument(
    "--nsamples_prior",
    type=int,
    default=int(5e4),
    help=(
        "Number of samples for computing the prior on the initial power spectrum "
        "(when using planck2018[_cv])."
    ),
)
parser.add_argument(
    "--radial_selection",
    type=none_or_bool_or_str,
    default="multiple_lognormal",
    help=(
        "Radial selection function. "
        'Set to "multiple_lognormal" for multi-population lognormal selection.'
    ),
)
parser.add_argument(
    "--selection_params",
    type=float,
    nargs="*",
    help="Parameters for the radial selection function (see hiddenbox.py).",
)
parser.add_argument(
    "--survey_mask_path",
    type=none_or_bool_or_str,
    default=None,
    help="Absolute path to the survey mask (if any).",
)
parser.add_argument(
    "--sim_params",
    type=none_or_bool_or_str,
    default=None,
    help="Parameters for the gravity solver.",
)
parser.add_argument(
    "--lin_bias",
    type=float,
    nargs="*",
    help="Linear biases.",
)
parser.add_argument(
    "--obs_density",
    type=none_or_bool_or_str,
    default=None,
    help="Observed density.",
)
parser.add_argument(
    "--noise",
    type=float,
    default=0.1,
    help="Noise level.",
)
parser.add_argument(
    "--force",
    type=bool_sh,
    default=False,
    help="Force recomputations if True.",
)

args = parser.parse_args()

if __name__ == "__main__":
    try:
        wd_ext = args.wd_ext
        name = args.name
        total_steps = args.total_steps
        aa = args.aa
        size = args.size
        Np0 = args.Np0
        Npm0 = args.Npm0
        L = args.L
        S = args.S
        Pinit = args.Pinit
        Nnorm = args.Nnorm
        Ne = args.Ne
        Ns = args.Ns
        Delta_theta = args.Delta_theta
        OUTDIR = args.OUTDIR
        prior_type = args.prior
        nsamples_prior = int(args.nsamples_prior)
        radial_selection = args.radial_selection
        if radial_selection == "multiple_lognormal":
            selection_params = np.reshape(np.array(args.selection_params), (3, -1))
        else:
            logger.error("Radial selection not yet implemented.")
            raise NotImplementedError("Only 'multiple_lognormal' is supported at present.")
        survey_mask_path = args.survey_mask_path
        sim_params = args.sim_params
        isstd = sim_params[:3] == "std"
        splitLPT = sim_params[:8] == "splitLPT"
        gravity_on = sim_params[:6] != "nograv"
        if isinstance(args.lin_bias, list):
            lin_bias = np.array(args.lin_bias)
        else:
            lin_bias = args.lin_bias
        Npop = len(lin_bias) if isinstance(lin_bias, np.ndarray) else 1
        obs_density = args.obs_density
        noise = args.noise
        force = args.force

        # Configure plotting aesthetics for consistent visualisation
        setup_plotting()

        # Create directories
        wd_noname = f"{OUTDIR}{wd_ext}{size}{int(L)}{Pinit}{Nnorm}/"
        wd = wd_noname + name + "/"
        modeldir = wd + "model/"
        figuresdir = wd + "Figures/"

        Path(wd + "RESULTS/").mkdir(parents=True, exist_ok=True)
        Path(modeldir).mkdir(parents=True, exist_ok=True)
        Path(wd_noname + "wn/").mkdir(parents=True, exist_ok=True)
        Path(wd + "data/").mkdir(parents=True, exist_ok=True)
        Path(figuresdir).mkdir(parents=True, exist_ok=True)
        Path(wd + "pool/").mkdir(parents=True, exist_ok=True)
        Path(wd + "score_compression/").mkdir(parents=True, exist_ok=True)

        for d in range(S + 1):
            dirsims = wd + f"pool/d{d}/"
            Path(dirsims).mkdir(parents=True, exist_ok=True)

        np.save(modeldir + "radial_selection.npy", radial_selection)
        np.save(modeldir + "selection_params.npy", selection_params)
        np.save(modeldir + "lin_bias.npy", lin_bias)
        np.save(modeldir + "obs_density.npy", obs_density)
        np.save(modeldir + "noise.npy", noise)

        logger.info("Setting up model parameters...")

        k_max = get_k_max(L, size)  # k_max in h/Mpc
        logger.info("Maximum wavenumber: k_max = %f", k_max)
        # Cosmo at the expansion point:
        params_planck = params_planck_kmax_missing.copy()
        params_planck["k_max"] = k_max
        # Fiducial BBKS spectrum for normalisation:
        params_BBKS = params_BBKS_kmax_missing.copy()
        params_BBKS["k_max"] = k_max
        # Observed cosmology:
        params_cosmo_obs = params_cosmo_obs_kmax_missing.copy()
        params_cosmo_obs["k_max"] = k_max

        params = setup_model(
            workdir=modeldir,
            params_planck=params_planck,
            params_P0=params_BBKS,
            size=size,
            L=L,
            S=S,
            Pinit=Pinit,
            force=True,
        )
        gc.collect()

        (
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
            planck_Pk_EH,
        ) = params

        other_params = {
            "size": size,
            "P": P,
            "Np0": Np0,
            "Npm0": Npm0,
            "L": L,
            "S": S,
            "total_steps": total_steps,
            "aa": aa,
            "G_sim_path": G_sim_path,
            "G_ss_path": G_ss_path,
            "P_ss_obj_path": P_ss_obj_path,
            "Pinit": Pinit,
            "Nnorm": Nnorm,
            "Ne": Ne,
            "Ns": Ns,
            "Delta_theta": Delta_theta,
            "sim_params": sim_params,
        }
        with open(modeldir + "other_params.pkl", "wb") as f:
            pickle.dump(other_params, f)

        # Save a human readable record of the parameters
        with open(wd + "params.txt", "w") as f:
            f.write("Parameters for this run:\n")
            f.write("size: " + str(size) + "\n")
            f.write("Np0: " + str(Np0) + "\n")
            f.write("Npm0: " + str(Npm0) + "\n")
            f.write("L: " + str(L) + "\n")
            f.write("S: " + str(S) + "\n")
            f.write("Pinit: " + str(Pinit) + "\n")
            f.write("P: " + str(P) + "\n")
            f.write("Nnorm: " + str(Nnorm) + "\n")
            f.write("total_steps: " + str(total_steps) + "\n")
            f.write("aa: " + str(aa) + "\n")
            f.write("Ne: " + str(Ne) + "\n")
            f.write("Ns: " + str(Ns) + "\n")
            f.write("Delta_theta: " + str(Delta_theta) + "\n")
            f.write("OUTDIR: " + OUTDIR + "\n")
            f.write("prior_type: " + prior_type + "\n")
            f.write("nsamples_prior: " + str(nsamples_prior) + "\n")
            f.write("radial_selection: " + str(radial_selection) + "\n")
            f.write("selection_params:\n" + str(selection_params) + "\n")
            f.write("survey_mask_path: " + str(survey_mask_path) + "\n")
            f.write("lin_bias: " + str(lin_bias) + "\n")
            f.write("obs_density: " + str(obs_density) + "\n")
            f.write("noise: " + str(noise) + "\n")
            f.write("sim_params: " + str(sim_params) + "\n")

        logger.info("Setting up model parameters done.")

        logger.info("Generating ground truth spectrum...")
        gt_path = modeldir + "theta_gt.npy"
        if not exists(gt_path) or force:
            from pysbmy.power import get_Pk

            theta_gt = get_Pk(k_s, params_cosmo_obs)
            np.save(gt_path, theta_gt)
            del theta_gt
        logger.info("Generating ground truth spectrum done.")

        def theta2P(theta):
            return theta * P_0

        merged_path, indices_steps_cumul, eff_redshifts = handle_time_stepping(
            aa=aa,
            total_steps=total_steps,
            modeldir=modeldir,
            figuresdir=figuresdir,
            sim_params=sim_params,
            force=force,
        )

        # Instantiate the HiddenBox object
        logger.info("Instantiating the HiddenBox...")
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
            TimeStepDistribution=merged_path,
            TimeSteps=indices_steps_cumul,
            eff_redshifts=eff_redshifts,
            seedphase=BASELINE_SEEDPHASE,
            seednoise=BASELINE_SEEDNOISE,
            fixnoise=False,
            seednorm=BASELINE_SEEDNORM,
            reset=True,
            save_frequency=5,
        )
        logger.info("Instantiating the HiddenBox done.")

        logger.info("Generating Simbelmynë parameter files for normalisation...")
        if gravity_on:
            HB_selfi.switch_setup()
        INDENT()
        for i in range(Nnorm):
            logger.diagnostic("Setting Simbelmynë file %d/%d...", i + 1, Nnorm, verbosity=1)
            worker_normalisation_public(HB_selfi, params_planck, Nnorm, i)
            logger.diagnostic("Setting Simbelmynë file %d/%d done.", i + 1, Nnorm, verbosity=1)
        if gravity_on:
            HB_selfi.switch_setup()

        if prior_type == "selfi2019":
            logger.diagnostic("Computing cosmic variance alpha_cv...")
            compute_alpha_cv(
                workdir=modeldir,
                k_s=k_s,
                size=size,
                L=L,
                window_fct_path=wd[:-1] + "/model/select_fct.h5",
                force=True,
            )
            logger.diagnostic("Computing cosmic variance alpha_cv done.")
        UNINDENT()
        logger.info("Generating Simbelmynë parameter files for normalisation done.")

    except OSError as e:
        logger.error("Directory or file access error: %s", str(e))
        raise
    except Exception as e:
        logger.critical("An unexpected error occurred: %s", str(e))
        raise RuntimeError("Pipeline step 0a failed.") from e
    finally:
        gc.collect()
        logger.info("step 0a of the SelfiSys pipeline: done.")
