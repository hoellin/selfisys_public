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
Priors for the SelfiSys pipeline. This module provides:
- a Planck2018-based prior class (`planck_prior`) compatible with
pySelfi, adapted for the logic of the SelfiSys pipeline;
- wrappers for the selfi2019 prior from [leclercq2019primordial].


Raises
------
OSError
    If file or directory paths are inaccessible.
RuntimeError
    If unexpected HPC or multi-processing errors arise.
"""

import gc

from selfisys.utils.logger import getCustomLogger

logger = getCustomLogger(__name__)


def get_summary(x, bins, normalisation=None, kmax=1.4):
    """
    Compute a power-spectrum summary for given cosmological parameters.

    Parameters
    ----------
    x : array-like
        Cosmological parameters [h, Omega_b, Omega_m, n_s, sigma_8].
    bins : array-like
        Wavenumber bins.
    normalisation : float or None, optional
        Normalisation constant to scale the resulting spectrum.
    kmax : float, optional
        Maximum wavenumber for get_Pk.

    Returns
    -------
    theta : ndarray
        The computed power-spectrum values, optionally normalised.

    Raises
    ------
    RuntimeError
        If the power-spectrum computation fails unexpectedly.
    """
    from numpy import array
    from pysbmy.power import get_Pk
    from selfisys.utils.tools import cosmo_vector_to_Simbelmyne_dict

    try:
        theta = get_Pk(bins, cosmo_vector_to_Simbelmyne_dict(x, kmax=kmax))
        if normalisation is not None:
            theta /= normalisation
        return array(theta)
    except Exception as e:
        logger.critical("Unexpected error in get_summary: %s", str(e))
        raise RuntimeError("Failed to compute power spectrum summary.") from e
    finally:
        gc.collect()


def worker_class(params):
    """
    Worker function to compute power spectra with CLASS, compatible with
    Python multiprocessing.

    Parameters
    ----------
    params : tuple
        (x, bins, normalisation, kmax) where x is an array-like of
        cosmological parameters, bins is the wavenumber array,
        normalisation is a float or None, and kmax is a float.

    Returns
    -------
    theta : ndarray
        Power-spectrum summary from `get_summary`.
    """
    x, bins, normalisation, kmax = params
    return get_summary(x, bins, normalisation, kmax)


class planck_prior:
    """
    Custom prior for the SelfiSys pipeline. This is the prior used in
    [hoellinger2024diagnosing], based on the Planck 2018 cosmological
    parameters.

    This class provides methods to compute a power-spectrum prior from a
    prior distribution of cosmological parameters, using a Gaussian fit.
    See equation (7) in [hoellinger2024diagnosing].

    Parameters
    ----------
    Omega_mean : array-like
        Mean of the prior distribution on cosmological parameters.
    Omega_cov : array-like
        Covariance matrix of the prior distribution on cosmological
        parameters.
    bins : array-like
        Wavenumbers where the power spectrum is evaluated.
    normalisation : float or None
        If not None, divide the power spectra by the normalisation.
    kmax : float
        Maximum wavenumber for computations.
    nsamples : int, optional
        Number of samples drawn from the prior on the cosmological
        parameters. Default is 10,000.
    nthreads : int, optional
        Number of CPU threads for parallel tasks. Default is -1, that
        is, auto-detect the number of available threads.
    EPS_K : float, optional
        Regularisation parameter for covariance inversion. Default 1e-7.
    EPS_residual : float, optional
        Additional cutoff for matrix inversion. Default 1e-3.
    filename : str or None, optional
        Path to a .npy file to store or load precomputed power spectra.

    Attributes
    ----------
    mean : ndarray
        Mean of the computed power spectra.
    covariance : ndarray
        Covariance matrix of the computed power spectra.
    inv_covariance : ndarray
        Inverse of the covariance matrix.

    Raises
    ------
    OSError
        If file reading or writing fails.
    RuntimeError
        For unexpected HPC or multi-processing errors.
    """

    def __init__(
        self,
        Omega_mean,
        Omega_cov,
        bins,
        normalisation,
        kmax,
        nsamples=10000,
        nthreads=-1,
        EPS_K=1e-7,
        EPS_residual=1e-3,
        filename=None,
    ):
        from numpy import where
        from multiprocessing import cpu_count

        self.Omega_mean = Omega_mean
        self.Omega_cov = Omega_cov
        self.bins = bins
        self.normalisation = normalisation
        self.kmax = kmax
        self.nsamples = nsamples
        self.EPS_K = EPS_K
        self.EPS_residual = EPS_residual
        self.filename = filename

        if nthreads == -1:
            # Use #CPU - 1 or fallback to 1 if a single CPU is available
            self.nthreads = cpu_count() - 1 or 1
        else:
            self.nthreads = nthreads

        self._Nbin_min = where(self.bins >= 0.01)[0].min()
        self._Nbin_max = where(self.bins <= self.kmax)[0].max() + 1

        # Attributes set after compute()
        self.mean = None
        self.covariance = None
        self.inv_covariance = None
        self.thetas = None

    @property
    def Nbin_min(self, k_min):
        """Index of the minimal wavenumber given k_min."""
        return self._Nbin_min

    @property
    def Nbin_max(self, k_min):
        """Index of the maximal wavenumber given self.kmax."""
        return self._Nbin_max

    def compute(self):
        """
        Compute the prior (mean, covariance, and inverse covariance).

        If `self.filename` exists, tries to load the prior. Otherwise,
        samples from the prior distribution on cosmological parameters
        and evaluates the power spectra in parallel.

        Raises
        ------
        OSError
            If self.filename is not writable/accessible.
        RuntimeError
            If multi-processing or power-spectra computations fail.
        """
        from os.path import exists
        import numpy as np

        try:
            if self.filename and exists(self.filename):
                logger.info("Loading precomputed thetas from %s", self.filename)
                self.thetas = np.load(self.filename)
            else:
                from time import time
                from multiprocessing import Pool
                import tqdm.auto as tqdm

                logger.info("Sampling %d cosmological parameter sets...", self.nsamples)
                OO = np.random.multivariate_normal(
                    np.array(self.Omega_mean), np.array(self.Omega_cov), self.nsamples
                )
                eps = 1e-5
                OO = np.clip(OO, eps, 1 - eps)

                liste = [(o, self.bins, self.normalisation, self.kmax) for o in OO]

                logger.info(
                    "Computing prior power spectra in parallel using %d threads...", self.nthreads
                )
                start = time()
                with Pool(self.nthreads) as pool:
                    thetas = []
                    for theta in tqdm.tqdm(pool.imap(worker_class, liste), total=len(liste)):
                        thetas.append(theta)
                thetas = np.array(thetas)
                end = time()
                logger.info("Done computing power spectra in %.2f seconds.", end - start)

                self.thetas = thetas
                if self.filename:
                    logger.info("Saving thetas to %s", self.filename)
                    np.save(self.filename, thetas)

            # Compute stats
            self.mean = np.mean(self.thetas, axis=0)
            self.covariance = np.cov(self.thetas.T)

            logger.info("Regularising and inverting the prior covariance matrix.")
            from pyselfi.utils import regular_inv

            self.inv_covariance = regular_inv(self.covariance, self.EPS_K, self.EPS_residual)

        except OSError as e:
            logger.error("File I/O error: %s", str(e))
            raise
        except Exception as e:
            logger.critical("Error during prior computation: %s", str(e))
            raise RuntimeError("planck_prior computation failed.") from e
        finally:
            gc.collect()

    def logpdf(self, theta, theta_mean, theta_covariance, theta_icov):
        """
        Return the log prior probability at a given point in parameter
        space.

        Parameters
        ----------
        theta : ndarray
            Evaluation point in parameter space.
        theta_mean : ndarray
            Prior mean vector.
        theta_covariance : ndarray
            Prior covariance matrix.
        theta_icov : ndarray
            Inverse of the prior covariance matrix.

        Returns
        -------
        float
            Log prior probability value.
        """
        import numpy as np

        diff = theta - theta_mean
        val = -0.5 * diff.dot(theta_icov).dot(diff)
        val -= 0.5 * np.linalg.slogdet(2 * np.pi * theta_covariance)[1]
        return val

    def sample(self, seedsample=None):
        """
        Draw a random sample from the prior distribution.

        Parameters
        ----------
        seedsample : int, optional
            Seed for the random number generator.

        Returns
        -------
        ndarray
            A single sample from the prior distribution.
        """
        from numpy.random import seed, multivariate_normal

        if seedsample is not None:
            seed(seedsample)
        return multivariate_normal(self.mean, self.covariance)

    def save(self, fname):
        """
        Save the prior to an output file.

        Parameters
        ----------
        fname : str
            Output HDF5 filename to store the prior data.

        Raises
        ------
        OSError
            If the file cannot be accessed or written.
        """
        import h5py
        from ctypes import c_double
        from pyselfi.utils import PrintMessage, save_replace_dataset, save_replace_attr

        try:
            PrintMessage(3, f"Writing prior in data file '{fname}'...")
            with h5py.File(fname, "r+") as hf:

                def save_to_hf(name, data, **kwargs):
                    save_replace_dataset(hf, f"/prior/{name}", data, dtype=c_double, **kwargs)

                # Hyperparameters
                save_to_hf("thetas", self.thetas, maxshape=(None, None))
                save_to_hf("Omega_mean", self.Omega_mean, maxshape=(None,))
                save_to_hf("Omega_cov", self.Omega_cov, maxshape=(None, None))
                save_to_hf("bins", self.bins, maxshape=(None,))

                save_replace_attr(hf, "/prior/normalisation", self.normalisation, dtype=c_double)
                save_replace_attr(hf, "/prior/kmax", self.kmax, dtype=c_double)

                # Mandatory attributes
                save_to_hf("mean", self.mean, maxshape=(None,))
                save_to_hf("covariance", self.covariance, maxshape=(None, None))
                save_to_hf("inv_covariance", self.inv_covariance, maxshape=(None, None))

            PrintMessage(3, f"Writing prior in data file '{fname}' done.")
        except OSError as e:
            logger.error("Failed to save prior to '%s': %s", fname, str(e))
            raise
        finally:
            gc.collect()

    @classmethod
    def load(cls, fname):
        """
        Load the prior from input file.

        Parameters
        ----------
        fname : str
            Input HDF5 filename.

        Returns
        -------
        prior
            The prior object.

        Raises
        ------
        OSError
            If the file cannot be read or is invalid.
        """
        from h5py import File
        from numpy import array
        from ctypes import c_double
        from pyselfi.utils import PrintMessage

        try:
            PrintMessage(3, f"Reading prior in data file '{fname}'...")
            with File(fname, "r") as hf:
                # Load constructor parameters
                Omega_mean = array(hf.get("/prior/Omega_mean"), dtype=c_double)
                Omega_cov = array(hf.get("/prior/Omega_cov"), dtype=c_double)
                bins = array(hf.get("/prior/bins"), dtype=c_double)
                normalisation = hf.attrs["/prior/normalisation"]
                kmax = hf.attrs["/prior/kmax"]

                # Instantiate class
                prior = cls(Omega_mean, Omega_cov, bins, normalisation, kmax)

                # Load mandatory arrays
                prior.mean = array(hf.get("prior/mean"), dtype=c_double)
                prior.covariance = array(hf.get("/prior/covariance"), dtype=c_double)
                prior.inv_covariance = array(hf.get("/prior/inv_covariance"), dtype=c_double)

            PrintMessage(3, f"Reading prior in data file '{fname}' done.")
            return prior
        except OSError as e:
            logger.error("Failed to read prior from '%s': %s", fname, str(e))
            raise
        finally:
            gc.collect()


def logposterior_hyperparameters_parallel(
    selfi,
    theta_fiducial,
    Nbin_min,
    Nbin_max,
    theta_norm,
    k_corr,
    alpha_cv,
):
    """
    Compute the log-posterior for the hyperparameters of the prior from
    [leclercq2019primordial], for use within the SelfiSys pipeline.

    Parameters
    ----------
    selfi : object
        The selfi object.
    theta_fiducial : ndarray
        Fiducial spectrum.
    Nbin_min : int
        Minimum bin index for the wavenumber range.
    Nbin_max : int
        Maximum bin index for the wavenumber range.
    theta_norm : float
        Hyperparameter controlling the overall uncertainty.
    k_corr : float
        Hyperparameter controlling correlation scale.
    alpha_cv : float
        Cosmic variance strength.

    Returns
    -------
    float
        The log-posterior value for the given hyperparameters.

    Raises
    ------
    RuntimeError
        If the log-posterior computation fails unexpectedly.
    """
    try:
        return selfi.logposterior_hyperparameters(
            theta_fiducial, Nbin_min, Nbin_max, theta_norm, k_corr, alpha_cv
        )
    except Exception as e:
        logger.critical("Unexpected error in logposterior_hyperparameters_parallel: %s", str(e))
        raise RuntimeError("logposterior_hyperparameters_parallel failed.") from e
    finally:
        gc.collect()


def perform_prior_optimisation_and_plot(
    selfi,
    theta_fiducial,
    theta_norm_mean=0.1,
    theta_norm_std=0.3,
    k_corr_mean=0.020,
    k_corr_std=0.015,
    k_opt_min=0.0,
    k_opt_max=1.4,
    theta_norm_min=0.04,
    theta_norm_max=0.12,
    k_corr_min=0.012,
    k_corr_max=0.02,
    meshsize=30,
    Nbin_min=0,
    Nbin_max=100,
    theta_norm=0.05,
    k_corr=0.015,
    alpha_cv=0.00065,
    plot=True,
    savepath=None,
):
    """
    Optimise the hyperparameters for the selfi2019 prior (from
    [leclercq2019primordial]).

    Parameters
    ----------
    selfi : object
        The selfi object.
    theta_fiducial : ndarray
        Fiducial spectrum.
    theta_norm_mean : float, optional
        Mean of the Gaussian hyperprior on theta_norm. Default 0.1.
    theta_norm_std : float, optional
        Standard deviation of the hyperprior on theta_norm. Default 0.3.
    k_corr_mean : float, optional
        Mean of the Gaussian hyperprior on k_corr. Default 0.020.
    k_corr_std : float, optional
        Standard deviation of the hyperprior on k_corr. Default 0.015.
    k_opt_min : float, optional
        Minimum wavenumber for the prior optimisation. Default 0.0.
    k_opt_max : float, optional
        Maximum wavenumber for the prior optimisation. Default 1.4.
    theta_norm_min : float, optional
        Lower bound for theta_norm in the mesh. Default 0.04.
    theta_norm_max : float, optional
        Upper bound for theta_norm in the mesh. Default 0.12.
    k_corr_min : float, optional
        Lower bound for k_corr in the mesh. Default 0.012.
    k_corr_max : float, optional
        Upper bound for k_corr in the mesh. Default 0.02.
    meshsize : int, optional
        Number of points in each dimension of the plot mesh. Default 30.
    Nbin_min : int, optional
        Minimum bin index for restricting the prior. Default 0.
    Nbin_max : int, optional
        Maximum bin index for restricting the prior. Default 100.
    theta_norm : float, optional
        Initial or default guess of theta_norm. Default 0.05.
    k_corr : float, optional
        Initial or default guess of k_corr. Default 0.015.
    alpha_cv : float, optional
        Cosmic variance term or similar. Default 0.00065.
    plot : bool, optional
        If True, generate and show/save a 2D contour plot. Default True.
    savepath : str, optional
        File path to save the plot. If None, the plot is displayed.

    Returns
    -------
    tuple
        (theta_norm, k_corr) after optimisation.

    Raises
    ------
    OSError
        If file operations fail during saving the prior or posterior.
    RuntimeError
        If the optimisation fails unexpectedly.
    """
    try:
        if plot:
            from selfisys.utils.plot_utils import get_contours
            from numpy import meshgrid, linspace, zeros, exp, array
            from joblib import Parallel, delayed

            logger.info("Preparing the hyperparameter grid for plotting (meshsize=%d).", meshsize)

            X0, Y0 = meshgrid(
                linspace(theta_norm_min, theta_norm_max, meshsize),
                linspace(k_corr_min, k_corr_max, meshsize),
            )
            Z = zeros((meshsize, meshsize))

            # Evaluate log-posterior on the grid in parallel
            Z = array(
                Parallel(n_jobs=-1)(
                    delayed(logposterior_hyperparameters_parallel)(
                        selfi,
                        theta_fiducial,
                        Nbin_min,
                        Nbin_max,
                        X0[i][j],
                        Y0[i][j],
                        alpha_cv,
                    )
                    for i in range(meshsize)
                    for j in range(meshsize)
                )
            ).reshape(meshsize, meshsize)
            Z -= Z.max()
            Z = exp(Z)

            Z_contours = get_contours(Z, meshsize)
            logger.info("Grid evaluations complete.")

        logger.info("Performing the prior hyperparameter optimisation...")
        selfi.prior.theta_norm = theta_norm
        selfi.prior.k_corr = k_corr
        selfi.prior.alpha_cv = alpha_cv

        # Perform the prior optimisation
        x0 = [theta_norm, k_corr]
        selfi.optimize_prior(
            theta_fiducial,
            k_opt_min,
            k_opt_max,
            x0=x0,
            theta_norm_min=theta_norm_min,
            theta_norm_max=theta_norm_max,
            theta_norm_mean=theta_norm_mean,
            theta_norm_std=theta_norm_std,
            k_corr_min=k_corr_min,
            k_corr_max=k_corr_max,
            k_corr_mean=k_corr_mean,
            k_corr_std=k_corr_std,
            options={
                "maxiter": 30,
                "ftol": 1e-10,
                "gtol": 1e-10,
                "eps": 1e-6,
                "disp": False,
            },
        )

        logger.info("Saving prior and posterior after optimisation.")
        selfi.save_prior()
        selfi.save_posterior()

        theta_norm = selfi.prior.theta_norm
        k_corr = selfi.prior.k_corr

        prior_theta_mean, prior_theta_covariance = selfi.prior.mean, selfi.prior.covariance
        prior_theta_mean = prior_theta_mean[Nbin_min:Nbin_max]
        prior_theta_covariance = prior_theta_covariance[Nbin_min:Nbin_max, Nbin_min:Nbin_max]

        posterior_theta_mean, posterior_theta_covariance, posterior_theta_icov = (
            selfi.restrict_posterior(Nbin_min, Nbin_max)
        )

        logger.info("Optimised hyperparameters: theta_norm=%.5f, k_corr=%.5f", theta_norm, k_corr)

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax.xaxis.set_tick_params(which="major", length=6, labelsize=17)
            ax.xaxis.set_tick_params(which="minor", length=4)
            ax.yaxis.set_tick_params(which="both", direction="in", width=1.0)
            ax.yaxis.set_tick_params(which="major", length=6, labelsize=17)

            pcm = ax.pcolormesh(X0, Y0, Z, cmap="Greys", shading="gouraud")
            ax.grid(linestyle=":")
            ax.contour(
                Z,
                Z_contours,
                extent=[theta_norm_min, theta_norm_max, k_corr_min, k_corr_max],
                colors="C9",
            )
            ax.plot(
                [x0[0], x0[0]],
                [k_corr_min, k_corr_max],
                color="C3",
                linestyle=":",
                label="Before optimisation",
            )
            ax.plot([theta_norm_min, theta_norm_max], [x0[1], x0[1]], color="C3", linestyle=":")
            ax.plot(
                [theta_norm, theta_norm],
                [k_corr_min, k_corr_max],
                linestyle="--",
                color="C3",
                label="After optimisation",
            )
            ax.plot([theta_norm_min, theta_norm_max], [k_corr, k_corr], linestyle="--", color="C3")

            ax.set_xlabel(r"$\theta_\mathrm{norm}$", size=19)
            ax.set_ylabel(r"$k_\mathrm{corr}$ [$h$/Mpc]", size=19)
            ax.legend()

            if savepath is None:
                plt.show()
            else:
                fig.savefig(savepath, bbox_inches="tight", dpi=300, format="png", transparent=True)
                fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
            plt.close(fig)

        return theta_norm, k_corr

    except OSError as e:
        logger.error("File access or I/O error: %s", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error in perform_prior_optimisation_and_plot: %s", str(e))
        raise RuntimeError("perform_prior_optimisation_and_plot failed.") from e
    finally:
        gc.collect()
