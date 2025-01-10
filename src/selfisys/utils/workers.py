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
Routines for parameter inference and gradient evaluation in the SelfiSys
pipeline.
"""

import gc
from typing import Any, Tuple, List

from selfisys.utils.logger import getCustomLogger

logger = getCustomLogger(__name__)


def Simbelmyne_worker(args) -> Tuple[float, Any]:
    """
    Worker function used for implicit likelihood inference of
    cosmological parameters.

    Parameters
    ----------
    args : tuple
        A tuple of arguments to be unpacked for the worker routine:
        (index, param_val, param_id, fsimdir, k_s, Pbins_bnd,
        selection_params, norm_csts, P_ss_obj_path, obs_density,
        lin_bias, noise, survey_mask_path, G_sim_path, G_ss_path, Np0,
        Npm0, seedphase_init, seednoise_init, size, L,
        radial_selection, sim_params, wd, batch_idx, dbg, modeldir,
        local_mask_prefix, TimeStepDistribution, indices_steps_cumul,
        eff_redshifts, poolname_abc, setup_only, prefix_mocks).

    Returns
    -------
    tuple
        (param_val, Phi) where param_val is the parameter value used,
        and Phi is the resulting summary from evaluating the model.

    Raises
    ------
    OSError
        If file I/O (reading or writing mock data) fails.
    RuntimeError
        For unexpected errors in the worker routine.
    """
    import os
    from pathlib import Path

    try:
        (
            index,
            param_val,
            param_id,
            fsimdir,
            k_s,
            Pbins_bnd,
            selection_params,
            norm_csts,
            P_ss_obj_path,
            obs_density,
            lin_bias,
            noise,
            survey_mask_path,
            G_sim_path,
            G_ss_path,
            Np0,
            Npm0,
            seedphase_init,
            seednoise_init,
            size,
            L,
            radial_selection,
            sim_params,
            wd,
            batch_idx,
            dbg,
            modeldir,
            local_mask_prefix,
            TimeStepDistribution,
            indices_steps_cumul,
            eff_redshifts,
            poolname_abc,
            setup_only,
            prefix_mocks,
        ) = args

        spectrum_name = int(str(seedphase_init + index) + str(seednoise_init + index))
        pooldir = (
            fsimdir + "/pool/d" if not poolname_abc else fsimdir + "/pool/" + poolname_abc + "/d"
        )
        simdir_d = pooldir + str(spectrum_name) + "/"
        Path(simdir_d).mkdir(parents=True, exist_ok=True)

        if prefix_mocks is None:
            fname_mocks = (
                simdir_d + "mocks_d" + str(spectrum_name) + "_p" + str(batch_idx + index) + ".h5"
            )
        else:
            fname_mocks = (
                simdir_d
                + prefix_mocks
                + "_mocks_d"
                + str(spectrum_name)
                + "_p"
                + str(batch_idx + index)
                + ".h5"
            )

        if os.path.exists(fname_mocks):
            from h5py import File

            logger.debug("Mock file %s found, loading existing data...", fname_mocks)
            with File(fname_mocks, "r") as f:
                Phi = f["Phi"][:]
        else:
            logger.debug("No existing mock file at %s, generating new data...", fname_mocks)
            from numpy.random import normal
            from numpy import shape, max
            from selfisys.global_parameters import BASELINE_SEEDNORM, omegas_gt
            from selfisys.utils.tools import get_k_max
            from selfisys.hiddenbox import HiddenBox
            from selfisys.utils.tools import get_summary

            P = len(Pbins_bnd) - 1
            try:
                BB_selfi = HiddenBox(
                    k_s=k_s,
                    P_ss_path=P_ss_obj_path,
                    Pbins_bnd=Pbins_bnd,
                    theta2P=None,
                    P=P * shape(selection_params)[1],  # P * Npop
                    size=size,
                    L=L,
                    G_sim_path=G_sim_path,
                    G_ss_path=G_ss_path,
                    Np0=Np0,
                    Npm0=Npm0,
                    fsimdir=wd[:-1],
                    modeldir=modeldir,
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
                    seedphase=seedphase_init,
                    seednoise=seednoise_init,
                    fixnoise=False,
                    seednorm=BASELINE_SEEDNORM,
                    reset=False,
                    save_frequency=5,
                    verbosity=2,
                )
                k_max = get_k_max(L, size)
            except Exception as e:
                logger.critical("Error instantiating HiddenBox: %s", str(e))
                raise RuntimeError("Failed to set up HiddenBox.") from e

            # Evaluate the param -> 'theta' using some get_summary logic
            try:
                theta = get_summary(param_val, param_id, omegas_gt, k_s, kmax=k_max)
            except Exception:
                max_tries = 10
                perturb_std = 1e-8
                param_val_init = param_val
                logger.warning(
                    "get_summary failed for param_val=%s. Trying small perturbations...", param_val
                )
                for i in range(max_tries):
                    param_val = normal(param_val_init, perturb_std)
                    logger.diagnostic("Attempt #%d: param_val=%s", i + 1, param_val)
                    try:
                        theta = get_summary(param_val, param_id, omegas_gt, k_s, kmax=k_max)
                        logger.diagnostic(
                            "Success with param_val=%s on attempt #%d", param_val, i + 1
                        )
                        break
                    except Exception:
                        if i == max_tries - 1:
                            logger.critical(
                                "All attempts to get_summary failed for param_val=%s",
                                param_val_init,
                            )
                            raise RuntimeError("get_summary repeatedly failed.")
                        continue

            from io import BytesIO
            from selfisys.utils.low_level import stderr_redirector, stdout_redirector

            cosmo_vect = omegas_gt
            cosmo_vect[param_id] = param_val

            logger.debug("Evaluating model with HPC redirection, setup_only=%s", setup_only)
            f = BytesIO()
            g = BytesIO()
            try:
                with stderr_redirector(f):
                    with stdout_redirector(g):
                        if setup_only:
                            BB_selfi.switch_setup()
                        else:
                            BB_selfi.switch_recompute_pool(prefix_mocks=prefix_mocks)

                        Phi = BB_selfi.evaluate(
                            theta,
                            spectrum_name,
                            seedphase_init + index,
                            seednoise_init + index,
                            i=batch_idx + index,
                            thetaIsP=True,
                            remove_sbmy=True,
                            force_powerspectrum=dbg,
                            force_parfiles=dbg,
                            check_output=dbg,
                            abc=poolname_abc,
                            cosmo_vect=cosmo_vect,
                        )
                        if setup_only:
                            BB_selfi.switch_setup()
                        else:
                            BB_selfi.switch_recompute_pool(prefix_mocks=prefix_mocks)

            except Exception as e:
                logger.critical("Error while evaluating model: %s", str(e))
                raise RuntimeError("Simbelmyne_worker model evaluation failed.") from e
            finally:
                g.close()
                f.close()

        logger.debug("Returning param_val=%s with resulting Phi of shape %s", param_val, Phi.shape)
        return param_val, Phi

    except OSError as e:
        logger.error("File I/O error in Simbelmyne_worker: %s", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error in Simbelmyne_worker: %s", str(e))
        raise RuntimeError("Simbelmyne_worker HPC run failed.") from e
    finally:
        gc.collect()


def worker_gradient_Symbelmyne(
    coeff: float,
    delta_x: float,
    omega,
    param_index: int,
    k_s,
    delta: float,
    kmax: float,
):
    """
    Worker function for evaluating the gradient of the power spectrum
    using finite differences.

    Parameters
    ----------
    coeff : float
        Coefficient for the finite difference.
    delta_x : float
        Step size in the parameter space.
    omega : ndarray
        Base cosmological parameter vector.
    param_index : int
        Index of the parameter being varied.
    k_s : ndarray
        Array of wavenumbers.
    delta : float
        Denominator for finite differences (scaled).
    kmax : float
        Maximum wavenumber for power spectrum.

    Returns
    -------
    ndarray
        The gradient of the power spectrum wrt the specified parameter.

    Raises
    ------
    RuntimeError
        If the gradient evaluation fails.
    """
    import numpy as np
    from pysbmy.power import get_Pk
    from selfisys.utils.tools import cosmo_vector_to_Simbelmyne_dict

    omega_new = omega.copy()
    try:
        omega_new[param_index] += delta_x
        ps = get_Pk(k_s, cosmo_vector_to_Simbelmyne_dict(omega_new, kmax=kmax))
        contrib_to_grad = (coeff * ps) / delta
        return np.array(contrib_to_grad)
    except Exception as e:
        logger.critical("Error in worker_gradient_Symbelmyne: %s", str(e))
        raise RuntimeError("worker_gradient_Symbelmyne failed.") from e
    finally:
        gc.collect()


def evaluate_gradient_of_Symbelmyne(
    omega,
    param_index: int,
    k_s,
    coeffs: List[float] = [2 / 3.0, -1 / 12.0],
    deltas_x: List[float] = [0.01, 0.02],
    delta: float = 1e-2,
    kmax: float = 1.4,
):
    """
    Estimate the gradient of CLASS with respect to the cosmological
    parameters using central finite differences of arbitrary order.

    Parameters
    ----------
    omega : ndarray
        Base cosmological parameter vector.
    param_index : int
        Index of the parameter to differentiate against.
    k_s : ndarray
        Wavenumbers for the power spectrum.
    coeffs : list of float, optional
        Coefficients for the finite-difference scheme, typically
        [2/3, -1/12] etc. Default is [2/3.0, -1/12.0].
    deltas_x : list of float, optional
        Step sizes. The corresponding negative steps are generated
        automatically. Default is [0.01, 0.02].
    delta : float, optional
        Scale for the finite difference in the denominator. Default is
        1e-2.
    kmax : float, optional
        Maximum wavenumber for the power spectrum. Default is 1.4.

    Returns
    -------
    ndarray
        The gradient of the power spectrum wrt the specified parameter.

    Raises
    ------
    RuntimeError
        If the gradient evaluation fails.
    """
    import numpy as np
    from multiprocessing import Pool

    try:
        grad = np.zeros(len(k_s))
        full_coeffs = np.concatenate((-np.array(coeffs)[::-1], coeffs))
        deltas_x_full = np.concatenate((-np.array(deltas_x)[::-1], deltas_x))

        tasks = [
            (c, dx, omega, param_index, k_s, delta, kmax)
            for c, dx in zip(full_coeffs, deltas_x_full)
        ]
        logger.diagnostic("Starting parallel HPC for gradient, tasks=%d", len(tasks))

        with Pool() as mp_pool:
            results = mp_pool.starmap(worker_gradient_Symbelmyne, tasks)
            for contrib in results:
                grad += contrib

        logger.diagnostic("Gradient evaluation completed. Shape=%s", grad.shape)
        return grad
    except Exception as e:
        logger.critical("Unexpected error in evaluate_gradient_of_Symbelmyne: %s", str(e))
        raise RuntimeError("evaluate_gradient_of_Symbelmyne failed.") from e
    finally:
        gc.collect()
