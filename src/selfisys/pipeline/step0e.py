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
Step 0e of the SelfiSys pipeline.

Generate all the Simbelmyne parameter files to run the simulations at
the expansion point, in all parameter space directions, in order to
linearise the HiddenBox.

Unless the forward model is run in no-gravity mode, the only simulation
actually run here is the one to generate the prior on the initial
spectrum (if using planck2018[_cv]), based on cosmological parameters
drawn from the prior.
"""

import gc
import pickle
import numpy as np
from pathlib import Path
from os.path import exists

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


def worker_fct(params):
    """
    Run a simulation in parallel to linearise the HiddenBox.

    Parameters
    ----------
    params : tuple
        A tuple containing (x, index, selfi_object):
        x : int or float
            Direction index (1..S) or 0 for the expansion point.
        index : int or None
            Simulation index for the expansion point. Expect None if
            direction x is not 0.
        selfi_object : object
            Instance of the selfi object.

    Returns
    -------
    int
        Returns 0 on successful completion.
    """
    from io import BytesIO
    from selfisys.utils.low_level import stdout_redirector
    import gc

    x, idx, selfi_object = params
    logger.debug("Running simulation: x=%s, idx=%s", x, idx)

    # Capture output to avoid cluttering logs
    f = BytesIO()
    with stdout_redirector(f):
        selfi_object.run_simulations(d=x, p=idx)
    f.close()

    # Release memory
    del selfi_object
    gc.collect()
    return 0


parser = ArgumentParser(
    description=(
        "Step 0e of the SelfiSys pipeline. "
        "Generate all the required Simbelmyne parameter files for the simulations "
        "at the expansion point, and compute the prior on the initial spectrum."
    )
)
parser.add_argument("--wd", type=str, help="Absolute path of the working directory.")
parser.add_argument(
    "--N_THREADS",
    type=int,
    default=64,
    help=(
        "Number of threads for computing the prior. Also serves as the number of "
        "parameter files to generate in parallel (note that a distinct HiddenBox "
        "object is instantiated for each)."
    ),
)
parser.add_argument(
    "--prior",
    type=str,
    default="planck2018",
    help=(
        "Prior for the parameters. Possible values: "
        '"selfi2019" (as in [leclercq2019primordial]), '
        '"planck2018" (Planck 2018 cosmology), '
        '"planck2018_cv" (Planck 2018 + cosmic variance).'
    ),
)
parser.add_argument(
    "--nsamples_prior",
    type=int,
    default=int(1e4),
    help=(
        "Number of samples (drawn from the prior on cosmology) to compute the prior "
        "on the primordial power spectrum (when using planck2018[_cv])."
    ),
)
parser.add_argument(
    "--survey_mask_path",
    type=none_or_bool_or_str,
    default=None,
    help="Path to the survey mask for the well-specified model.",
)
parser.add_argument(
    "--name_obs",
    type=none_or_bool_or_str,
    default=None,
    help=(
        "Prefix for the observation files. If None, uses a default name. "
        "Can be used to work with different data vectors."
    ),
)
parser.add_argument(
    "--effective_volume",
    type=bool_sh,
    default=False,
    help="Use the effective volume to compute alpha_cv.",
)
parser.add_argument(
    "--force_recompute_prior",
    type=bool_sh,
    default=False,
    help="Force overwriting the prior.",
)
parser.add_argument(
    "--Ne",
    type=int,
    default=None,
    help=(
        "Number of simulations to keep at the expansion point. "
        "If None, uses the value from the prior steps."
    ),
)
parser.add_argument(
    "--Ns",
    type=int,
    default=None,
    help=(
        "Number of simulations for each gradient component. "
        "If None, uses the value from the prior steps."
    ),
)

args = parser.parse_args()

wd = args.wd
N_THREADS = args.N_THREADS
prior_type = args.prior
nsamples_prior = int(args.nsamples_prior)
survey_mask_path = args.survey_mask_path
name_obs = "_" + args.name_obs if args.name_obs is not None else None
local_mask_prefix = args.name_obs if args.name_obs is not None else None
effective_volume = args.effective_volume
force_recompute_prior = args.force_recompute_prior

modeldir = wd + "model/"
prior_dir = ROOT_PATH + "data/stored_priors/"
Path(prior_dir).mkdir(parents=True, exist_ok=True)

P_0 = np.load(modeldir + "P_0.npy")


def theta2P(theta):
    """
    Convert dimensionless theta to physical P(k).

    Parameters
    ----------
    theta : ndarray
        The dimensionless power-spectrum values.

    Returns
    -------
    ndarray
        The physical power-spectrum values.
    """
    return theta * P_0


if __name__ == "__main__":
    from pysbmy.timestepping import read_timestepping
    from os.path import exists
    from selfisys.hiddenbox import HiddenBox

    try:
        logger.diagnostic("Setting up main parameters...")

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
        Ne = other_params["Ne"] if args.Ne is None else args.Ne
        Ns = other_params["Ns"] if args.Ns is None else args.Ns
        Delta_theta = other_params["Delta_theta"]
        sim_params = other_params["sim_params"]

        isstd = sim_params[:3] == "std"
        splitLPT = sim_params[:8] == "splitLPT"
        gravity_on = sim_params[:6] != "nograv"

        radial_selection = safe_npload(modeldir + "radial_selection.npy")
        selection_params = np.load(modeldir + "selection_params.npy")
        lin_bias = np.load(modeldir + "lin_bias.npy")
        Npop = len(lin_bias) if isinstance(lin_bias, np.ndarray) else 1
        obs_density = safe_npload(modeldir + "obs_density.npy")
        noise = np.load(modeldir + "noise.npy")

        k_max = get_k_max(L, size)  # k_max in h/Mpc

        Pbins_bnd = np.load(modeldir + "Pbins_bnd.npy")
        Pbins = np.load(modeldir + "Pbins.npy")
        k_s = np.load(modeldir + "k_s.npy")
        planck_Pk_EH = np.load(modeldir + "theta_planck.npy")

        INDENT()
        if isstd:
            TimeStepDistribution = None
            eff_redshifts = None
            TimeSteps = None
        elif splitLPT:
            TimeStepDistribution = None
            TimeSteps = [f"pop{i}" for i in range(1, len(aa))]
            eff_redshifts = [1 / a - 1 for a in aa[1:]]
        else:
            logger.info("Setting up the time-stepping...")
            nsteps = [
                round((aa[i + 1] - aa[i]) / (aa[-1] - aa[0]) * total_steps)
                for i in range(len(aa) - 1)
            ]
            if sum(nsteps) != total_steps:
                nsteps[nsteps.index(max(nsteps))] += total_steps - sum(nsteps)
            TimeSteps = list(np.cumsum(nsteps) - 1)
            merged_path = modeldir + "merged.h5"
            TS_merged = read_timestepping(merged_path)

            if sim_params.startswith("custom") or sim_params.startswith("nograv"):
                TimeStepDistribution = merged_path
                eff_redshifts = 1 / aa[-1] - 1
            else:
                raise NotImplementedError("Time-stepping strategy not yet implemented.")
            logger.info("Time-stepping setup done.")
        UNINDENT()
        logger.diagnostic("Setting up main parameters done.")

        # Normalisation constants
        logger.diagnostic("Loading normalisation constants...")
        norm_csts_path = modeldir + "norm_csts.npy"
        if not exists(norm_csts_path):
            raise ValueError(
                "Normalisation constants not found. Please run steps 0c and 0d before 0e."
            )
        norm_csts = np.load(norm_csts_path)
        logger.diagnostic("Normalisation constants loaded.")

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
            norm_csts=norm_csts,
            survey_mask_path=survey_mask_path,
            local_mask_prefix=local_mask_prefix,
            sim_params=sim_params,
            TimeStepDistribution=TimeStepDistribution,
            TimeSteps=TimeSteps,
            eff_redshifts=eff_redshifts,
            seedphase=BASELINE_SEEDPHASE,
            seednoise=BASELINE_SEEDNOISE,
            fixnoise=False,
            seednorm=BASELINE_SEEDNORM,
            reset=False,
            save_frequency=5,
        )
        logger.info("HiddenBox instantiated successfully.")

        logger.diagnostic("Loading the ground truth spectrum...")
        if not exists(modeldir + "theta_gt.npy"):
            raise ValueError("Ground truth cosmology not found.")
        theta_gt = np.load(modeldir + "theta_gt.npy")
        logger.diagnostic("Ground truth spectrum loaded.")

        logger.diagnostic("Loading observations...")
        if not exists(joinstrs([modeldir, "phi_obs", name_obs, ".npy"])):
            raise ValueError("Observation data not found.")
        phi_obs = np.load(joinstrs([modeldir, "phi_obs", name_obs, ".npy"]))
        logger.diagnostic("Observations loaded.")

        logger.info("Setting up the prior and instantiating the selfi object...")
        fname_results = wd + "RESULTS/res.h5"
        pool_prefix = wd + "pool/pool_res_dir_"
        pool_suffix = ".h5"

        from pyselfi.power_spectrum.selfi import power_spectrum_selfi

        if prior_type == "selfi2019":
            from pyselfi.power_spectrum.prior import power_spectrum_prior

            theta_0 = np.ones(S)
            if effective_volume:
                alpha_cv = np.load(modeldir + "alpha_cv_eff.npy")
            else:
                alpha_cv = np.load(modeldir + "alpha_cv.npy")

            prior = power_spectrum_prior(
                k_s, theta_0, THETA_NORM_GUESS, K_CORR_GUESS, alpha_cv, False
            )
            selfi = power_spectrum_selfi(
                fname_results,
                pool_prefix,
                pool_suffix,
                prior,
                HB_selfi,
                theta_0,
                Ne,
                Ns,
                Delta_theta,
                phi_obs,
            )

            selfi.prior.theta_norm = THETA_NORM_GUESS
            selfi.prior.k_corr = K_CORR_GUESS
            selfi.prior.alpha_cv = alpha_cv
        elif prior_type.startswith("planck2018"):
            from selfisys.prior import planck_prior

            theta_planck = np.load(modeldir + "theta_planck.npy")
            theta_0 = theta_planck / P_0

            prior = planck_prior(
                planck_mean,
                planck_cov,
                k_s,
                P_0,
                k_max,
                nsamples=nsamples_prior,
                nthreads=N_THREADS,
                filename=(
                    prior_dir
                    + f"planck_prior_S{S}_L{L}_size{size}_"
                    + f"{nsamples_prior}_{WHICH_SPECTRUM}.npy"
                ),
            )
            selfi = power_spectrum_selfi(
                fname_results,
                pool_prefix,
                pool_suffix,
                prior,
                HB_selfi,
                theta_0,
                Ne,
                Ns,
                Delta_theta,
                phi_obs,
            )
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")

        logger.info("Prior and selfi object created successfully.")

        # Plot the observed summaries
        logger.info("Plotting the observed summaries...")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(k_s, theta_gt / P_0, label=r"$\theta_{\mathrm{gt}}$", color="C0")
        ax1.set_xscale("log")
        ax1.semilogx(
            k_s,
            planck_Pk_EH / P_0,
            label=r"$P_{\mathrm{Planck}}(k)/P_0(k)$",
            color="C1",
            lw=0.5,
        )
        ax1.set_xlabel("$k$ [$h$/Mpc]")
        ax1.set_ylabel("$[\\mathrm{Mpc}/h]^3$")
        ax1.grid(which="both", axis="y", linestyle="dotted", linewidth=0.6)
        for kk in k_s[:-1]:
            ax1.axvline(x=kk, color="green", linestyle="dotted", linewidth=0.6)
        ax1.axvline(
            x=k_s[-1],
            color="green",
            linestyle="dotted",
            linewidth=0.6,
            label=r"$\theta$-bins boundaries",
        )
        ax1.axvline(x=Pbins[0], color="red", linestyle="dashed", linewidth=0.5)
        ax1.axvline(x=Pbins[-1], color="red", linestyle="dashed", linewidth=0.5)
        for kk in Pbins[1:-2]:
            ax1.axvline(x=kk, ymax=0.167, color="red", linestyle="dashed", linewidth=0.5)
        ax1.legend(loc=2)
        ax1.set_xlim(max(1e-4, k_s.min() - 2e-4), k_s.max())
        ax1.set_ylim(7e-1, 1.6e0)

        ax2 = ax1.twinx()
        ax2.axvline(
            x=Pbins[-2],
            ymax=0.333,
            color="red",
            linestyle="dashed",
            linewidth=0.5,
            label=r"$\psi$-bins centers",
        )
        len_obs = len(phi_obs) // np.shape(selection_params)[1]
        cols = ["C4", "C5", "C6", "C7"]
        for i in range(np.shape(selection_params)[1]):
            ax2.plot(
                Pbins,
                phi_obs[i * len_obs : (i + 1) * len_obs],
                marker="x",
                label=rf"Summary $\psi_{{\mathrm{{obs}}}},$ pop {i}",
                linewidth=0.5,
                color=cols[i % len(cols)],
            )
        ax2.legend(loc=1)
        ax2.set_ylabel("Summary values")
        plt.title(
            "Observations generated with the ground truth cosmology and well-specified models"
        )
        plt.savefig(wd + "Figures/summary_obs_step0e.pdf", bbox_inches="tight", dpi=300)
        plt.close()
        logger.info("Plotting the observed summaries done.")

        logger.info("Loading or computing prior...")
        error_str_prior = (
            "Error while computing the prior. For OOM issues, a fix might be to set "
            "os.environ['OMP_NUM_THREADS'] = '1'. Otherwise, refer to the error message."
        )

        if not prior_type.startswith("selfi2019"):
            if not force_recompute_prior:
                try:
                    selfi.prior = selfi.prior.load(selfi.fname)
                    logger.info("Prior loaded from file.")
                except:
                    logger.info("Prior not found in %s, recomputing...", selfi.fname)
                    try:
                        selfi.compute_prior()
                        selfi.save_prior()
                        selfi.prior = selfi.prior.load(selfi.fname)
                    except:
                        logger.critical(error_str_prior)
                        raise RuntimeError("Prior computation failed.")
                    logger.info("Prior computed and saved.")
            else:
                logger.info("Forcing recomputation of the prior (user request).")
                selfi.compute_prior()
                selfi.save_prior()
                selfi.prior = selfi.prior.load(selfi.fname)
        else:
            selfi.compute_prior()
            selfi.save_prior()
            selfi.load_prior()

        from os import cpu_count
        import tqdm.auto as tqdm
        from multiprocessing import Pool

        HB_selfi.switch_recompute_pool()
        if gravity_on:
            HB_selfi.switch_setup()

        list_part_1 = [[0, idx, selfi] for idx in range(Ne)]
        list_part_2 = [[x, None, selfi] for x in range(1, S + 1)]

        ncors = cpu_count()
        nprocess = min(N_THREADS, ncors, len(list_part_1[1:]) + len(list_part_2))
        logger.info("Using %d processes to generate Simbelmynë parameter files.", nprocess)
        gc.collect()

        # Generate parameter files for estimating f0
        logger.info("Generating parameter files for estimating f0...")
        # First poke the HiddenBox once to avoid Pool access issues
        worker_fct(list_part_1[0])
        with Pool(processes=nprocess) as mp_pool:
            pool_results_1 = mp_pool.map(worker_fct, list_part_1[1:])
            for _ in tqdm.tqdm(pool_results_1, total=len(list_part_1[1:])):
                pass
        logger.info("Generating parameter files for the estimation of f0 done.")

        # Generate parameter files for estimating the gradient
        logger.info("Generating parameter files for the gradient...")
        with Pool(processes=nprocess) as mp_pool:
            pool_results_2 = mp_pool.map(worker_fct, list_part_2)
            for _ in tqdm.tqdm(pool_results_2, total=len(list_part_2)):
                pass
        logger.info("Generating parameter files for the gradient done.")

        if gravity_on:
            HB_selfi.switch_setup()
        HB_selfi.switch_recompute_pool()

    except OSError as e:
        logger.error("File or directory access error during Step 0e: %s", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred in Step 0e: %s", str(e))
        raise RuntimeError("Step 0e failed.") from e
    finally:
        gc.collect()
        logger.info("Step 0e of the SelfiSys pipeline: done.")
