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
Third step of the SelfiSys pipeline.

Run the initial matter power spectrum inference, using the simulations
performed in previous steps.

Raises
------
OSError
    If file or directory paths are inaccessible.
RuntimeError
    If unexpected HPC or PySbmy issues occur.
"""

import gc

from selfisys.utils.parser import (
    ArgumentParser,
    none_or_bool_or_str,
    bool_sh,
    joinstrs,
    safe_npload,
)
from selfisys.utils.logger import getCustomLogger, INDENT, UNINDENT

logger = getCustomLogger(__name__)


def worker_fct(params):
    """
    Run a Simbelmynë simulation.

    Parameters
    ----------
    params : tuple
        (x, index, selfi_object)
        x : int or float
            Direction index (1..S), or 0 for the expansion point.
        index : int or None
            Simulation index for the expansion point.
        selfi_object : object
            Instance of the selfi object.

    Returns
    -------
    int
        Returns 0 on successful completion.

    Raises
    ------
    OSError
        If file/directory access fails for .sbmy or logs.
    RuntimeError
        If the simulation fails unexpectedly.
    """
    from io import BytesIO
    from selfisys.utils.low_level import stdout_redirector, stderr_redirector

    x, index, selfi_object = params

    # Check consistency
    if x != 0 and index is not None:
        raise ValueError("Expansion point is not 0 but index is not None.")

    logger.debug("Running simulation: offset=%s, index=%s", x, index)

    # Suppress console output to keep logs clean. Use with caution.
    f = BytesIO()
    g = BytesIO()
    with stdout_redirector(f):
        with stderr_redirector(g):
            selfi_object.run_simulations(d=x, p=index)
        g.close()
    f.close()

    del selfi_object
    gc.collect()
    return 0


from pathlib import Path
from os.path import exists
import numpy as np

from selfisys.global_parameters import *
from selfisys.utils.tools import get_k_max

parser = ArgumentParser(
    description=(
        "Third step of the SelfiSys pipeline. Run the initial matter power spectrum inference."
    )
)
parser.add_argument("--wd", type=str, help="Absolute path of the working directory.")
parser.add_argument("--N_THREADS", type=int, help="1 direction per thread.", default=64)
parser.add_argument(
    "--N_THREADS_PRIOR", type=int, help="Number of threads for computing the prior.", default=64
)
parser.add_argument(
    "--prior",
    type=str,
    default="planck2018",
    help=(
        "Prior for the parameters. Possible values:\n"
        '  - "selfi2019": prior used in [leclercq2019primordial]\n'
        '  - "planck2018": sampling from Planck 2018 cosmology\n'
        '  - "planck2018_cv": Planck 2018 + cosmic variance.\n'
        "Note: 'selfi2019' and 'planck2018_cv' have not been checked with the latest code "
        "version. Use at your own risk."
    ),
)
parser.add_argument(
    "--nsamples_prior",
    type=int,
    default=int(1e4),
    help="Number of samples to compute planck2018[_cv].",
)
parser.add_argument(
    "--survey_mask_path",
    type=none_or_bool_or_str,
    default=None,
    help="Path to the survey mask.",
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
    help=(
        "Prefix for the observation file. If None, uses default name. "
        "Enables working with different data vectors."
    ),
)
parser.add_argument(
    "--params_obs",
    type=none_or_bool_or_str,
    default=None,
    help="Recompute observations with the specified parameters.",
)
parser.add_argument(
    "--force_obs",
    type=bool_sh,
    default=False,
    help="Force re-computation of the observations.",
)
parser.add_argument(
    "--recompute_obs_mock",
    type=bool_sh,
    default=False,
    help="Recompute the observational part of the data vector.",
)
parser.add_argument(
    "--time_steps_obs",
    type=int,
    nargs="*",
    default=None,
    help=(
        "Suffixes of the fields to use for recomputing the observational part of the data vector. "
        "Ignored if `recompute_obs_mock=False`."
    ),
)
parser.add_argument(
    "--force_recompute_prior",
    type=bool_sh,
    default=False,
    help="Force overwriting the prior.",
)
parser.add_argument(
    "--update_obs_phase",
    type=bool_sh,
    default=False,
    help="Change the phase for observations.",
)
parser.add_argument(
    "--recompute_mocks",
    type=none_or_bool_or_str,
    default=False,
    help=(
        "Recompute all mocks in the inference phase, without affecting the dark matter fields. "
        "Possible values: True, 'gradients', 'list', 'listdd', 'list_exp'."
    ),
)
parser.add_argument(
    "--list_of_pps",
    type=float,
    nargs="*",
    default=None,
    help="Indices to recompute mocks for gradient, if 'list' is chosen.",
)
parser.add_argument("--pp_min", type=int, help="Min index for recompute_mocks in gradient mode.")
parser.add_argument("--pp_max", type=int, help="Max index for recompute_mocks in gradient mode.")
parser.add_argument("--dd_min", type=int, help="Min index for recompute_mocks in listdd mode.")
parser.add_argument("--dd_max", type=int, help="Max index for recompute_mocks in listdd mode.")
parser.add_argument(
    "--perform_score_compression",
    type=bool_sh,
    default=False,
    help="Perform score compression stage.",
)
parser.add_argument(
    "--force_score_compression",
    type=bool_sh,
    default=False,
    help="Force re-computation of the score compression.",
)
parser.add_argument(
    "--test_gravity",
    type=bool_sh,
    default=False,
    help="Flag to test gravity parameters. If True, norm_csts_path must be given.",
)
parser.add_argument(
    "--neglect_lightcone",
    type=bool_sh,
    default=False,
    help="Neglect lightcone effects even if multiple snapshots are available.",
)
parser.add_argument(
    "--norm_csts_path",
    type=none_or_bool_or_str,
    default=None,
    help="Path to external normalisation constants (needed for test_gravity=True).",
)
parser.add_argument(
    "--Ne",
    type=int,
    default=None,
    help="Number of simulations at the expansion point (override if not None).",
)
parser.add_argument(
    "--Ns",
    type=int,
    default=None,
    help="Number of simulations per gradient component (override if not None).",
)
parser.add_argument(
    "--prefix_mocks",
    type=none_or_bool_or_str,
    default=None,
    help="Prefix for mock files, if any.",
)
parser.add_argument(
    "--selection_params",
    type=float,
    nargs="*",
    default=None,
    help="Selection function parameters for the well-specified model.",
)
parser.add_argument(
    "--reset_window_function",
    type=bool_sh,
    default=False,
    help="Reset the window function if True.",
)
parser.add_argument(
    "--obs_density",
    type=float,
    default=None,
    help="Observed density override for the well-specified model.",
)
parser.add_argument(
    "--lin_bias",
    type=float,
    nargs="*",
    default=None,
    help="Linear bias override; uses stored value otherwise.",
)
parser.add_argument(
    "--figdir_suffix",
    type=none_or_bool_or_str,
    default=None,
    help="Suffix for the figures directory.",
)
parser.add_argument(
    "--noise_dbg",
    type=float,
    default=None,
    help="Manually specify noise level (for debugging). "
    "Normalisation constants won't reflect this override: use with caution.",
)

args = parser.parse_args()

wd = args.wd
N_THREADS = args.N_THREADS
N_THREADS_PRIOR = args.N_THREADS_PRIOR
prior_type = args.prior
nsamples_prior = int(args.nsamples_prior)
survey_mask_path = args.survey_mask_path
effective_volume = args.effective_volume
force_recompute_prior = args.force_recompute_prior
update_obs_phase = args.update_obs_phase
recompute_mocks = args.recompute_mocks
list_of_pps = args.list_of_pps
pp_min = args.pp_min
pp_max = args.pp_max
ddmin = args.dd_min
ddmax = args.dd_max
prefix_mocks = args.prefix_mocks
params_obs = args.params_obs
name_obs = "_" + args.name_obs if args.name_obs is not None else None
force_obs = args.force_obs
recompute_obs_mock = args.recompute_obs_mock if not force_obs else True
time_steps_obs = args.time_steps_obs
local_mask_prefix = args.name_obs if args.name_obs is not None else None
reset_window_function = args.reset_window_function
perform_score_compression = args.perform_score_compression
force_score_compression = args.force_score_compression
test_gravity = args.test_gravity
neglect_lightcone = args.neglect_lightcone
norm_csts_path = args.norm_csts_path
figdir_suffix = args.figdir_suffix + "/" if args.figdir_suffix is not None else ""
noise_dbg = args.noise_dbg

# Consistency checks
if pp_min is not None or pp_max is not None:
    if pp_min is None or pp_max is None:
        raise ValueError("both pp_min and pp_max should be specified if one of them is.")
    elif list_of_pps is not None:
        raise ValueError("pp_min and pp_max should not be specified if list_of_pps is specified.")
    else:
        list_of_pps = range(pp_min, pp_max + 1)

if list_of_pps is not None and recompute_mocks:
    if recompute_mocks[:4] != "list":
        raise ValueError("To use list_of_pps, set recompute_mocks to 'list'.")

if test_gravity:
    if force_obs:
        raise ValueError("test_gravity and force_obs cannot both be True.")
    if norm_csts_path is None:
        raise ValueError("norm_csts_path should be specified if test_gravity is True.")

if time_steps_obs is not None and not recompute_obs_mock:
    raise ValueError(
        "time_steps_obs can't be specified if recompute_obs_mock is False. "
        "If you want to recompute observations, set recompute_obs_mock=True."
    )

# Load parameters
modeldir = wd + "model/"
if prefix_mocks is not None:
    modeldir_refined = modeldir + prefix_mocks + "/"
else:
    modeldir_refined = modeldir
if args.selection_params is None:
    selection_params = np.load(modeldir + "selection_params.npy")
else:
    selection_params = np.reshape(np.array(args.selection_params), (3, -1))

if args.norm_csts_path is None:
    norm_csts_path = modeldir + "norm_csts.npy"

radial_selection = safe_npload(modeldir + "radial_selection.npy")

# If user hasn't specified new obs_density, load it from file
if args.obs_density is None:
    obs_density = safe_npload(modeldir + "obs_density.npy")
else:
    obs_density = args.obs_density

# If user hasn't specified new linear bias, load it from file
if args.lin_bias is None:
    lin_bias = np.load(modeldir + "lin_bias.npy")
else:
    if not (recompute_mocks or force_obs):
        raise ValueError(
            "lin_bias shouldn't be specified if neither recompute_mocks nor force_obs is True."
        )
    lin_bias = args.lin_bias
    if not isinstance(lin_bias, float):
        lin_bias = np.array(args.lin_bias)

Npop = len(lin_bias) if isinstance(lin_bias, np.ndarray) else 1

Path(wd + "Figures/").mkdir(exist_ok=True)
modeldir_obs = wd + "model/"

if prefix_mocks is not None:
    resultsdir = wd + "RESULTS/" + prefix_mocks + "/"
    figuresdir = joinstrs([wd, "Figures/", prefix_mocks, "/", figdir_suffix])
    resultsdir_obs = wd + "RESULTS/" + prefix_mocks + "/"
    scoredir = wd + "score_compression/" + prefix_mocks + "/"
    fname_results = wd + "RESULTS/" + prefix_mocks + "/res.h5"
else:
    resultsdir = wd + "RESULTS/"
    figuresdir = joinstrs([wd, "Figures/", figdir_suffix])
    resultsdir_obs = wd + "RESULTS/"
    scoredir = wd + "score_compression/"
    fname_results = wd + "RESULTS/res.h5"

Path(resultsdir).mkdir(parents=True, exist_ok=True)
Path(figuresdir).mkdir(parents=True, exist_ok=True)
Path(resultsdir_obs).mkdir(parents=True, exist_ok=True)
Path(modeldir_obs).mkdir(parents=True, exist_ok=True)
Path(modeldir_refined).mkdir(parents=True, exist_ok=True)
Path(scoredir).mkdir(parents=True, exist_ok=True)

logger.diagnostic("> Loading normalisation constants from: %s", norm_csts_path)
if not exists(norm_csts_path):
    raise ValueError("Normalisation constants not found. Please run steps 0, 1, 2 before step 3.")
else:
    norm_csts = np.load(norm_csts_path)
logger.diagnostic("Normalisation constants loaded successfully.")

# Write all parameters to a text file for reference
with open(figuresdir + "parameters.txt", "w") as f:
    f.write("wd = " + wd + "\n")
    f.write("N_THREADS = " + str(N_THREADS) + "\n")
    f.write("N_THREADS_PRIOR = " + str(N_THREADS_PRIOR) + "\n")
    f.write("prior_type = " + prior_type + "\n")
    f.write("nsamples_prior = " + str(nsamples_prior) + "\n")
    f.write("survey_mask_path = " + str(survey_mask_path) + "\n")
    f.write("effective_volume = " + str(effective_volume) + "\n")
    f.write("force_recompute_prior = " + str(force_recompute_prior) + "\n")
    f.write("update_obs_phase = " + str(update_obs_phase) + "\n")
    f.write("recompute_mocks = " + str(recompute_mocks) + "\n")
    f.write("list_of_pps = " + str(list_of_pps) + "\n")
    f.write("pp_min = " + str(pp_min) + "\n")
    f.write("pp_max = " + str(pp_max) + "\n")
    f.write("dd_min = " + str(ddmin) + "\n")
    f.write("dd_max = " + str(ddmax) + "\n")
    f.write("prefix_mocks = " + str(prefix_mocks) + "\n")
    f.write("params_obs = " + str(params_obs) + "\n")
    f.write("name_obs = " + str(name_obs) + "\n")
    f.write("force_obs = " + str(force_obs) + "\n")
    f.write("recompute_obs_mock = " + str(recompute_obs_mock) + "\n")
    f.write("local_mask_prefix = " + str(local_mask_prefix) + "\n")
    f.write("reset_window_function = " + str(reset_window_function) + "\n")
    f.write("perform_score_compression = " + str(perform_score_compression) + "\n")
    f.write("force_score_compression = " + str(force_score_compression) + "\n")
    f.write("test_gravity = " + str(test_gravity) + "\n")
    f.write("norm_csts_path = " + str(norm_csts_path) + "\n")
    f.write("norm_csts = " + str(norm_csts) + "\n")
    f.write("noise_dbg = " + str(noise_dbg) + "\n")

P_0 = np.load(modeldir + "P_0.npy")


def theta2P(theta):
    """
    Convert dimensionless theta to physical P(k).

    Parameters
    ----------
    theta : ndarray
        Dimensionless power-spectrum values.

    Returns
    -------
    ndarray
        Physical power-spectrum values (P_0 multiplied).
    """
    return theta * P_0


if __name__ == "__main__":
    import gc
    from pickle import load
    from pysbmy.timestepping import read_timestepping
    from selfisys.hiddenbox import HiddenBox

    try:
        logger.diagnostic(
            "Loading main parameters from other_params.pkl in modeldir: %s", modeldir
        )
        with open(modeldir + "other_params.pkl", "rb") as f:
            other_params = load(f)
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
        Pinit = other_params["Pinit"]
        Ne = other_params["Ne"] if args.Ne is None else args.Ne
        Ns = other_params["Ns"] if args.Ns is None else args.Ns
        Delta_theta = other_params["Delta_theta"]
        sim_params = other_params["sim_params"]
        isstd = sim_params[:3] == "std"
        splitLPT = sim_params[:8] == "splitLPT"

        noise = np.load(modeldir + "noise.npy") if noise_dbg is None else noise_dbg

        k_max = get_k_max(L, size)  # k_max in h/Mpc

        params_planck_EH = params_planck_kmax_missing.copy()
        params_planck_EH["k_max"] = k_max
        params_cosmo_obs = params_cosmo_obs_kmax_missing.copy()
        params_cosmo_obs["k_max"] = k_max

        Pbins_bnd = np.load(modeldir + "Pbins_bnd.npy")
        Pbins = np.load(modeldir + "Pbins.npy")
        k_s = np.load(modeldir + "k_s.npy")
        planck_Pk = np.load(modeldir + "theta_planck.npy")

        logger.diagnostic("Successfully loaded input data.")

        if isstd:
            TimeStepDistribution = None
            eff_redshifts = None
            TimeSteps = None
        elif splitLPT:
            TimeStepDistribution = None
            TimeSteps = [f"pop{i}" for i in range(1, len(aa))]
            eff_redshifts = [1 / a - 1 for a in aa[1:]]
        else:
            logger.info("Setting up time-stepping...")
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

        logger.info("Instantiating the HiddenBox...")
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
            TimeSteps=TimeSteps,
            eff_redshifts=eff_redshifts,
            seedphase=BASELINE_SEEDPHASE,
            seednoise=BASELINE_SEEDNOISE,
            fixnoise=False,
            seednorm=BASELINE_SEEDNORM,
            reset=reset_window_function,
            save_frequency=5,
        )
        logger.info("HiddenBox instantiated successfully.")

        modeldir_obs = wd + "model/"
        if force_obs:
            from pysbmy.power import get_Pk

            logger.info("Forceful re-computation of the ground truth.")
            theta_gt = get_Pk(k_s, params_cosmo_obs)
            np.save(joinstrs([modeldir_obs, "theta_gt", name_obs]), theta_gt)
            logger.info("Ground truth recomputed successfully.")
        elif not exists(joinstrs([modeldir_obs, "theta_gt", name_obs, ".npy"])):
            raise ValueError("Ground truth cosmology not found. Please run prior steps.")
        else:
            theta_gt = np.load(joinstrs([modeldir_obs, "theta_gt", name_obs, ".npy"]))

        if (
            not exists(joinstrs([modeldir, "phi_obs", name_obs, ".npy"]))
            and not recompute_obs_mock
        ):
            raise ValueError(
                "Observations not found. Please re-run previous steps or set --recompute_obs_mock."
            )
        elif recompute_obs_mock:
            logger.info("Recomputing the observed data vector...")
            if exists(joinstrs([modeldir_obs, "phi_obs", name_obs, ".npy"])) and not (
                update_obs_phase
            ):
                d_obs = -2
            else:
                d_obs = -1
            if time_steps_obs is not None:
                BB_selfi.update(TimeSteps=time_steps_obs)
            if params_obs is not None:
                BB_selfi.update(sim_params=params_obs)

            BB_selfi.switch_recompute_pool()
            phi_obs = BB_selfi.make_data(
                cosmo=params_cosmo_obs,
                id=joinstrs([BASEID_OBS, name_obs]),
                seedphase=SEEDPHASE_OBS,
                seednoise=SEEDNOISE_OBS,
                d=d_obs,
                force_powerspectrum=force_obs,
                force_parfiles=force_obs,
                force_sim=force_obs,
                force_cosmo=force_obs,
            )
            BB_selfi.switch_recompute_pool()

            if params_obs is not None:
                BB_selfi.update(sim_params=sim_params)
            if time_steps_obs is not None:
                BB_selfi.update(TimeSteps=TimeSteps)
            np.save(joinstrs([modeldir_obs, "phi_obs", name_obs, ".npy"]), phi_obs)
            logger.info("Observations recomputed successfully.")
        else:
            phi_obs = np.load(joinstrs([modeldir_obs, "phi_obs", name_obs, ".npy"]))

        logger.info("Setting up the prior and instantiating the selfi object...")
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
                BB_selfi,
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
                nthreads=N_THREADS_PRIOR,
                filename=joinstrs(
                    [
                        ROOT_PATH,
                        "data/stored_priors/planck_prior_S",
                        str(S),
                        "_L",
                        str(L),
                        "_size",
                        str(size),
                        "_",
                        str(nsamples_prior),
                        "_",
                        str(WHICH_SPECTRUM),
                        ".npy",
                    ]
                ),
            )
            selfi = power_spectrum_selfi(
                fname_results,
                pool_prefix,
                pool_suffix,
                prior,
                BB_selfi,
                theta_0,
                Ne,
                Ns,
                Delta_theta,
                phi_obs,
            )
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        logger.info("Prior set up and selfi object instantiated successfully.")

        if prior_type != "selfi2019":
            logger.info("Computing / loading prior...")
            if not force_recompute_prior:
                try:
                    selfi.prior = selfi.prior.load(selfi.fname)
                except:
                    logger.warning(
                        "Prior not found in %s, computing from scratch. Possible incomplete step 1?",
                        selfi.fname,
                    )
                    try:
                        logger.info("Computing prior from scratch.")
                        selfi.compute_prior()
                        selfi.save_prior()
                        selfi.prior = selfi.prior.load(selfi.fname)
                    except:
                        logger.error(
                            "Error whilst recomputing the prior. For OOM or another error encountered while computing prior. "
                            "For OOM issues, a simple fix might be setting os.environ['OMP_NUM_THREADS'] = '1'. "
                            "Otherwise, refer the the error message."
                        )
            else:
                try:
                    logger.warning("Forcefully recomputing prior from scratch: %s", selfi.fname)
                    selfi.compute_prior()
                    selfi.save_prior()
                    selfi.prior = selfi.prior.load(selfi.fname)
                except:
                    logger.error(
                        "OOM or another error while forcing prior computation. "
                        "For OOM issues, a simple fix might be setting os.environ['OMP_NUM_THREADS'] = '1'. "
                        "Otherwise, refer the the error message."
                    )
            logger.info("Prior computed / loaded successfully.")
        else:
            selfi.compute_prior()
            selfi.save_prior()
            selfi.load_prior()

        if recompute_mocks:
            if neglect_lightcone:
                BB_selfi.update(_force_neglect_lightcone=True)
            logger.info("Recomputing mocks...")
            INDENT()

            from multiprocessing import Pool
            from os import cpu_count
            import tqdm.auto as tqdm

            if recompute_mocks is True:
                list_part_1 = [[0, idx, selfi] for idx in range(Ne)]
                list_part_2 = [[x, None, selfi] for x in range(1, S + 1)]
                liste = list_part_1 + list_part_2
            elif recompute_mocks == "gradients":
                liste = [[x, None, selfi] for x in range(1, S + 1)]
            elif recompute_mocks == "list":
                liste = [[x, p, selfi] for x in range(1, S + 1) for p in list_of_pps]
            elif recompute_mocks == "listdd":
                liste = [[d, None, selfi] for d in range(ddmin, ddmax + 1)]
            elif recompute_mocks == "list_exp":
                liste = [[0, p, selfi] for p in list_of_pps]
            else:
                raise ValueError("recompute_mocks can't be {}".format(recompute_mocks))

            ncors = cpu_count()
            nprocess = min(N_THREADS, ncors, len(liste))
            logger.info("Using %s processes to compute the mocks", nprocess)
            gc.collect()
            BB_selfi.switch_recompute_pool(prefix_mocks=prefix_mocks)
            with Pool(processes=nprocess) as mp_pool:
                import tqdm.auto as tqdm

                pool = mp_pool.imap(worker_fct, liste)
                for contrib_to_grad in tqdm.tqdm(pool, total=len(liste)):
                    pass
            BB_selfi.switch_recompute_pool(prefix_mocks=None)
            UNINDENT()
            logger.info("Mocks recomputed successfully.")

        logger.diagnostic("Loading simulations...")
        BB_selfi.update(_prefix_mocks=prefix_mocks)
        selfi.likelihood.switch_bar()
        selfi.run_simulations(Ne=Ne, Ns=Ns)  # > Load < the simulations
        selfi.likelihood.switch_bar()
        BB_selfi.update(_prefix_mocks=None)
        logger.diagnostic("Simulations loaded successfully.")

        from selfisys.utils.plot_utils import *

        logger.diagnostic("Plotting the observed summaries...")
        plot_observations(
            k_s,
            theta_gt,
            planck_Pk,
            P_0,
            Pbins,
            phi_obs,
            np.shape(selection_params)[1],
            path=figuresdir + f"observations_{name_obs}.png",
        )
        gc.collect()
        logger.diagnostic("Plotting the observed summaries done.")

        logger.info("Running the inference...")
        INDENT()
        logger.info("Computing likelihoods and posteriors...")
        selfi.compute_likelihood()
        selfi.save_likelihood()
        selfi.load_likelihood()
        selfi.compute_posterior()
        selfi.save_posterior()
        selfi.load_posterior()
        logger.info("Computing likelihoods and posteriors done.")

        logger.diagnostic("Preparing SELFI outputs for plotting...")
        C_0 = selfi.likelihood.C_0
        grad_f = selfi.likelihood.grad_f
        Phi_0 = selfi.likelihood.Phi_0.Phi
        f_0 = selfi.likelihood.f_0
        f_16 = selfi.likelihood.f_8
        f_32 = selfi.likelihood.f_16
        f_48 = selfi.likelihood.f_24
        # f_16 = selfi.likelihood.f_16  # DBG. TODO: put this back.
        # f_32 = selfi.likelihood.f_32
        # f_48 = selfi.likelihood.f_48
        grad_f_16 = (f_16 - f_0) / Delta_theta
        grad_f_32 = (f_32 - f_0) / Delta_theta
        grad_f_48 = (f_48 - f_0) / Delta_theta
        X0, Y0 = np.meshgrid(Pbins, Pbins)
        X1, Y1 = np.meshgrid(k_s, Pbins)
        N = Ne

        np.save(resultsdir + "Phi_0.npy", Phi_0)
        np.save(resultsdir + "grad_f.npy", grad_f)
        np.save(resultsdir + "f_0.npy", f_0)
        np.save(resultsdir + "f_16.npy", f_16)
        np.save(resultsdir + "f_32.npy", f_32)
        np.save(resultsdir + "f_48.npy", f_48)
        np.save(resultsdir + "C_0.npy", C_0)
        logger.diagnostic("Preparing SELFI outputs for plotting done.")

        logger.diagnostic("Plotting the mocks and covariance matrices...")
        CovarianceMap = create_colormap("CovarianceMap")
        plot_mocks(
            1,
            N,
            P,
            Pbins,
            phi_obs,
            Phi_0,
            np.mean(Phi_0, axis=0),
            C_0,
            X0,
            Y0,
            CovarianceMap,
            savepath=figuresdir + "covariance_matrix.png",
        )
        plot_mocks_compact(
            1,
            N,
            P,
            Pbins,
            phi_obs,
            Phi_0,
            np.mean(Phi_0, axis=0),
            C_0,
            savepath=figuresdir + "mocks.pdf",
        )
        logger.diagnostic("Plotting the mocks and covariance matrices done.")

        logger.diagnostic("Plotting the full covariance matrix...")
        FullCovarianceMap = create_colormap("FullCovarianceMap")
        plot_C(
            C_0,
            X0,
            Y0,
            Pbins,
            FullCovarianceMap,
            binning=False,
            suptitle="Full covariance matrix",
            savepath=figuresdir + "full_covariance_matrix.png",
        )
        logger.diagnostic("Plotting the full covariance matrix done.")

        logger.diagnostic("Plotting the estimated gradients...")
        plot_gradients(
            Pbins,
            P,
            grad_f_16,
            grad_f_32,
            grad_f_48,
            grad_f,
            k_s,
            X1,
            Y1,
            fixscale=True,
            suptitle="Estimated gradients at expansion point for all populations of galaxies",
            savepath=figuresdir + "gradients.png",
        )
        logger.diagnostic("Plotting the estimated gradients done.")

        if prior_type == "selfi2019":
            # Optimise the prior hyperparameters
            theta_norm_min = 0.01
            theta_norm_max = 0.1
            k_corr_min = 0.005
            k_corr_max = 0.015

            theta_norm = THETA_NORM_GUESS
            k_corr = K_CORR_GUESS
            # The routine `Nbin_min` in "power_spectrum/prior.py" finds
            # the index of the minimal wavenumber given k_min, that is,
            # minimal index such that k_s[Nbin_min] >= k_min
            Nbin_min, Nbin_max = selfi.prior.Nbin_min(0), selfi.prior.Nbin_max(k_max)

            from selfisys.prior import perform_prior_optimisation_and_plot

            theta_fiducial = planck_Pk / P_0
            logger.info("Performing hyperparameters optimisation for the prior...")
            theta_norm, k_corr = perform_prior_optimisation_and_plot(
                selfi,
                theta_fiducial,
                k_opt_min=0.0,
                k_opt_max=max(k_s),
                theta_norm_mean=0.2,
                theta_norm_std=0.3,
                k_corr_mean=0.020,
                k_corr_std=0.015,
                theta_norm_min=theta_norm_min,
                theta_norm_max=theta_norm_max,
                k_corr_min=k_corr_min,
                k_corr_max=k_corr_max,
                meshsize=50,
                Nbin_min=Nbin_min,
                Nbin_max=Nbin_max,
                theta_norm=theta_norm,
                k_corr=k_corr,
                alpha_cv=alpha_cv,
                plot=True,
                verbose=False,
                savepath=wd + "Figures/prior_optimisation.png",
            )
            logger.info("Performing hyperparameters optimisation for the prior done.")

        prior_theta_mean = selfi.prior.mean
        prior_theta_covariance = selfi.prior.covariance
        Nbin_min, Nbin_max = 0, len(k_s)  # keep all scales
        k_s = k_s[Nbin_min:Nbin_max]
        P_0 = P_0[Nbin_min:Nbin_max]
        prior_theta_mean = prior_theta_mean[Nbin_min:Nbin_max]
        prior_theta_covariance = prior_theta_covariance[Nbin_min:Nbin_max, Nbin_min:Nbin_max]
        posterior_theta_mean, posterior_theta_covariance, posterior_theta_icov = (
            selfi.restrict_posterior(Nbin_min, Nbin_max)
        )

        X2, Y2 = np.meshgrid(k_s, k_s)
        prior_covariance = np.diag(P_0).dot(prior_theta_covariance).dot(np.diag(P_0))
        np.save(resultsdir + "prior_theta_mean.npy", prior_theta_mean)
        np.save(resultsdir + "prior_theta_covariance.npy", prior_theta_covariance)
        np.save(resultsdir_obs + "posterior_theta_mean.npy", posterior_theta_mean)
        np.save(resultsdir_obs + "posterior_theta_covariance.npy", posterior_theta_covariance)

        logger.diagnostic("Plotting the prior and posterior...")
        plot_prior_and_posterior_covariances(
            X2,
            Y2,
            k_s,
            prior_theta_covariance,
            prior_covariance,
            posterior_theta_covariance,
            P_0,
            suptitle="Prior and posterior covariance matrices",
            savepath=figuresdir + "prior_and_posterior_covariances.png",
        )
        logger.diagnostic("Plotting the prior and posterior done.")

        logger.diagnostic("Plotting the reconstruction...")
        plot_reconstruction(
            k_s,
            Pbins,
            prior_theta_mean,
            prior_theta_covariance,
            posterior_theta_mean,
            posterior_theta_covariance,
            theta_gt,
            P_0,
            suptitle="Posterior primordial matter power spectrum",
            savepath=figuresdir + "reconstruction.png",
        )
        logger.diagnostic("Plotting the reconstruction done.")
        UNINDENT()
        logger.info("Inference done.")

        if perform_score_compression:
            logger.info("Performing score compression...")
            INDENT()

            from selfisys.utils.tools import *
            from selfisys.utils.workers import evaluate_gradient_of_Symbelmyne

            logger.info("Computing the gradient of CLASS wrt the cosmological parameters...")
            delta = 1e-3
            if not exists(wd + "score_compression/grads_class.npy") or force_score_compression:
                coeffs = [4 / 5, -1 / 5, 4 / 105, -1 / 280]

                grad = np.zeros((len(planck_mean), len(k_s)))
                for i in range(len(planck_mean)):
                    if i == 0:  # workaround to correctly evaluate the gradient with respect to h
                        delta *= 10
                    logger.diagnostic("Evaluating gradient of CLASS wrt parameter %d" % i)
                    deltas_x = delta * np.linspace(1, len(coeffs), len(coeffs))
                    grad[i, :] = evaluate_gradient_of_Symbelmyne(
                        planck_mean,
                        i,
                        k_s,
                        coeffs=coeffs,
                        deltas_x=deltas_x,
                        delta=delta,
                        kmax=max(k_s),
                    )
                    if i == 0:
                        delta /= 10
                np.save(wd + "score_compression/grads_class.npy", grad)
            else:
                grad = np.load(wd + "score_compression/grads_class.npy")
            logger.info("Computing the gradient of CLASS wrt the cosmological parameters done.")

            grad_class = grad.T

            if not exists(wd + "score_compression/planck_Pk.npy") or force_score_compression:
                from pysbmy.power import get_Pk

                logger.info("Computing the Planck spectrum...")
                planck_Pk = get_Pk(k_s, params_planck_EH)
                np.save(wd + "score_compression/planck_Pk", planck_Pk)
                logger.info("Computing the Planck spectrum done.")
            else:
                planck_Pk = np.load(wd + "score_compression/planck_Pk.npy")

            logger.diagnostic("Plotting the gradients of CLASS...")
            plt.figure(figsize=(14, 10))
            names_of_parameters = [
                r"$h$",
                r"$\Omega_b$",
                r"$\Omega_m$",
                r"$n_s$",
                r"$\sigma_8$",
            ]
            fig, ax = plt.subplots(3, 2, figsize=(14, 15))
            u, v = (-1, -1)
            ax[u, v].loglog(k_s, planck_Pk)
            ax[u, v].set_xlabel(r"$k$ [h/Mpc]")
            ax[u, v].set_ylabel(r"$P(k)$")
            ax[u, v].set_title(r"$P=\mathcal{T}(\omega_{\rm Planck})$")
            for k in k_s:
                ax[u, v].axvline(k, color="k", alpha=0.1, linewidth=0.5)
            for i in range(len(planck_mean)):
                u = i // 2
                v = i % 2
                for k in k_s:
                    ax[u, v].axvline(k, color="k", alpha=0.1, linewidth=0.5)
                ax[u, v].plot(k_s, grad[i])
                ax[u, v].set_xscale("log")
                ax[u, v].set_xlabel(r"$k$ [h/Mpc]")
                ax[u, v].set_ylabel(r"$\partial P(k)/\partial$" + names_of_parameters[i])
                ax[u, v].set_title("Gradient wrt " + names_of_parameters[i])
            plt.suptitle("Gradient of Simbelmynë wrt cosmological parameters", fontsize=20)
            plt.tight_layout()
            plt.savefig(
                figuresdir + "gradient_class.png",
                bbox_inches="tight",
                dpi=300,
                transparent=True,
            )
            plt.savefig(figuresdir + "gradient_class.pdf", bbox_inches="tight", dpi=300)
            plt.close()
            logger.diagnostic("Plotting the gradients of CLASS done.")

            if prefix_mocks is None:
                logger.info("Computing Fisher matrix for the well specified model...")
            else:
                logger.info(
                    "Computing Fisher matrix for the misspecified model %s...", prefix_mocks
                )
            params_ids_fisher = np.linspace(0, 4, 5, dtype=int)

            dw_f0 = selfi.likelihood.grad_f.dot(grad_class)[:, params_ids_fisher]
            C0_inv = np.linalg.inv(selfi.likelihood.C_0)
            F0 = dw_f0.T.dot(C0_inv).dot(dw_f0)

            if not exists(scoredir + "dw_f0.npy") or force_score_compression:
                np.save(scoredir + "dw_f0.npy", dw_f0)
            if not exists(scoredir + "C0_inv.npy") or force_score_compression:
                np.save(scoredir + "C0_inv.npy", C0_inv)
            if not exists(scoredir + "F0.npy") or force_score_compression:
                np.save(scoredir + "F0.npy", F0)

            f0 = selfi.likelihood.f_0
            np.save(scoredir + "f0_expansion.npy", f0)

            plot_fisher(
                F0,
                names_of_parameters,
                title="Fisher matrix",
                path=figuresdir + "fisher.png",
            )
            logger.info("Fisher matrix for the well specified model done.")
            UNINDENT()
            logger.info("Score compression done.")

        logger.info("Computing additional statistics...")
        INDENT()

        logger.info("Computing the Mahalanobis distances...")
        INDENT()
        prior_theta_icov = selfi.prior.inv_covariance
        diff = posterior_theta_mean - prior_theta_mean
        Mahalanobis_distance = np.sqrt(diff.dot(prior_theta_icov).dot(diff))
        logger.info(
            "Mahalanobis distance between the posterior and the prior: %s", Mahalanobis_distance
        )
        np.savetxt(resultsdir_obs + "Mahalanobis_distances.txt", [Mahalanobis_distance])

        diff = theta_gt / P_0 - posterior_theta_mean
        Mahalanobis_distance_gt = np.sqrt(diff.dot(posterior_theta_icov).dot(diff))
        logger.info(
            "Mahalanobis distance between the groundtruth and the posterior: %s",
            Mahalanobis_distance_gt,
        )
        np.savetxt(resultsdir_obs + "Mahalanobis_distances_gt.txt", [Mahalanobis_distance_gt])

        NMahal = 2000
        logger.info(
            "Computing the mean Mahalanobis distance between the prior and %s samples...", NMahal
        )
        Mahalanobis_distances = []
        for _ in range(NMahal):
            theta = np.random.multivariate_normal(prior_theta_mean, prior_theta_covariance)
            diff = theta - prior_theta_mean
            mahal = np.sqrt(diff.dot(prior_theta_icov).dot(diff))
            Mahalanobis_distances.append(mahal)
        mean_Mahalanobis_distance = np.mean(Mahalanobis_distances)
        logger.info(
            "Mean Mahalanobis distance between the prior and %s samples: %s",
            NMahal,
            mean_Mahalanobis_distance,
        )
        np.savetxt(resultsdir_obs + "mean_Mahalanobis_distances.txt", [mean_Mahalanobis_distance])
        UNINDENT()
        logger.info("Computing the Mahalanobis distances done.")

        logger.diagnostic("Plotting the Mahalanobis distances...")
        plot_histogram(
            Mahalanobis_distances,
            Mahalanobis_distance,
            suptitle="Mahalanobis distances between the prior and samples",
            savepath=figuresdir + "Mahalanobis_distances.png",
        )
        logger.diagnostic("Plotting the Mahalanobis distances done.")
        UNINDENT()

    except OSError as e:
        logger.error("Directory or file access error: %s", str(e))
        raise
    except Exception as e:
        logger.critical("An unexpected error occurred: %s", str(e))
        raise RuntimeError("Pipeline step 3 failed.") from e
    finally:
        gc.collect()
        logger.info("step 3 of the SelfiSys pipeline: done.")
