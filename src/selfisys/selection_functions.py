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

"""Selection functions to simulate galaxy populations.
"""

import os
from gc import collect
import numpy as np
import h5py


class LognormalSelection:
    """Class to generate radial selection functions."""

    def __init__(
        self,
        L=None,
        selection_params=None,
        survey_mask_path=None,
        local_select_path=None,
        size=None,
    ):
        """
        Initialise the LognormalSelection object.

        Parameters
        ----------
        L : float
            Size of the simulation box (in Mpc/h). If not provided, it
            must be set before calling init_selection and using
            grid-dependent methods.
        selection_params : tuple of arrays
            Parameters for the selection functions (ss, mm, rr).
            Required for calling init_selection.
        survey_mask_path : str or None
            Path to the survey mask file. Required for calling
            init_selection.
        local_select_path : str
            Path where the selection function will be saved. Required
            for calling init_selection.
        size : int, optional
            Number of grid points along each axis. If not provided, it
            must be set before using grid-dependent methods.
        """
        self.L = L
        self.selection_params = selection_params
        self.survey_mask_path = survey_mask_path
        self.local_select_path = local_select_path
        self.size = size

    def r_grid(self):
        """Compute the grid of radial distances in the simulation box.

        Returns
        -------
        ndarray
            3D array of radial distances from the origin.

        Raises
        ------
        AttributeError
            If the 'size' attribute is not defined.
        """
        if self.size is None:
            raise AttributeError(
                "The attribute 'size' must be defined to compute the radial grid."
            )
        if self.L is None:
            raise AttributeError("The attribute 'L' must be defined to compute the radial grid.")

        range1d = np.linspace(0, self.L, self.size, endpoint=False)
        xx, yy, zz = np.meshgrid(range1d, range1d, range1d)
        x0 = y0 = z0 = 0.0
        r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2) + 1e-10
        return r

    @staticmethod
    def one_lognormal(x, std, mean, rescale=None):
        """Rescaled log-normal distribution.

        Parameters
        ----------
        x : ndarray
            Input array.
        std : float
            Standard deviation of the distribution.
        mean : float
            Mean of the distribution.
        rescale : float, optional
            Rescaling factor. If None, the distribution is normalised
            such that its maximum value is 1.

        Returns
        -------
        ndarray
            Log-normal distribution evaluated at x.
        """
        mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        sig2 = np.log(1 + std**2 / mean**2)
        lognorm = (1 / (np.sqrt(2 * np.pi) * np.sqrt(sig2) * x)) * np.exp(
            -((np.log(x) - mu) ** 2 / (2 * sig2))
        )
        if rescale is None:
            return lognorm / np.max(lognorm)
        else:
            return lognorm * rescale

    def multiple_lognormal(self, x, mask, ss, ll, rr):
        """Compute multiple log-normal distributions.

        Parameters
        ----------
        x : ndarray
            Input array.
        mask : ndarray or None
            Survey mask C(n).
        ss : array_like
            Standard deviations for each distribution.
        ll : array_like
            Means for each distribution.
        rr : array_like
            Rescaling factors for each distribution.

        Returns
        -------
        list of ndarray
            List of log-normal distributions.
        """
        if mask is None:
            mask = np.ones_like(x)
        return [self.one_lognormal(x, s, l, r) * mask for s, l, r in zip(ss, ll, rr)]

    @staticmethod
    def one_lognormal_z(x, sig2, mu, rescale=None):
        """Compute a log-normal distribution in redshift.

        Parameters
        ----------
        x : ndarray
            Input array.
        sig2 : float
            Variance of the distribution.
        mu : float
            Mean of the distribution.
        rescale : float, optional
            Rescaling factor.

        Returns
        -------
        ndarray
            Log-normal distribution evaluated at x.
        """
        lognorm = (1 / (np.sqrt(2 * np.pi) * np.sqrt(sig2) * x)) * np.exp(
            -((np.log(x) - mu) ** 2 / (2 * sig2))
        )
        return lognorm * rescale if rescale is not None else lognorm

    def multiple_lognormal_z(self, x, mask, ss, mm, rr):
        """
        Compute multiple rescaled lognormal distributions as functions
        of redshift.

        Parameters
        ----------
        x : ndarray
            Input array (redshifts).
        mask : ndarray or None
            Survey mask C(n).
        ss : array_like
            Standard deviations of the lognormal distributions.
        mm : array_like
            Means of the lognormal distributions.
        rr : array_like
            Rescaling factors for each distribution.

        Returns
        -------
        list of ndarray
            List of log-normal distributions.
        """
        if mask is None:
            mask = np.ones_like(x)
        res = []
        maxima = []
        for s, m, r in zip(ss, mm, rr):
            mu = np.log(m**2 / np.sqrt(s**2 + m**2))
            sig2 = np.log(1 + s**2 / m**2)
            maxima.append(np.exp(sig2 / 2 - mu) / (np.sqrt(2 * np.pi * sig2)))
            res.append(self.one_lognormal_z(x, sig2, mu, rescale=r) * mask)
        max = np.max(maxima)
        res = [r / max for r in res]
        return res

    def lognormals_z_to_x(self, xx, mask, params, spline):
        """Convert log-normal distributions from redshift to distance.

        Parameters
        ----------
        xx : array-like
            Comoving distances at which to evaluate the distributions.
        mask : ndarray or None
            Survey mask C(n).
        params : tuple of arrays
            Parameters for the distributions (ss, mm, rr).
        spline : UnivariateSpline
            Linear interpolator for the distance-redshift relation.

        Returns
        -------
        tuple
            Tuple containing redshifts and list of distributions.
        """
        ss, mm, rr = params
        zs = np.maximum(1e-4, spline(xx))
        res = self.multiple_lognormal_z(zs, mask, ss, mm, rr)
        return zs, res

    def init_selection(self, reset=False):
        """Initialise the radial selection functions.

        Parameters
        ----------
        reset : bool, optional
            Whether to reset the selection function.

        Raises
        ------

        """
        if any([self.survey_mask_path is None, self.local_select_path is None]):
            raise AttributeError(
                "Some attributes are missing to initialise the selection function."
            )

        if not os.path.exists(self.local_select_path) or reset:
            from scipy.interpolate import UnivariateSpline
            from classy import Class
            from astropy.cosmology import FlatLambdaCDM
            from selfisys.utils.tools import cosmo_vector_to_class_dict
            from selfisys.global_parameters import omegas_gt
            from selfisys.utils.plot_utils import plot_selection_functions

            # Redshift-distance relation
            redshifts_upper_bound = 3.0
            zz = np.linspace(0, redshifts_upper_bound, 10_000)
            cosmo = FlatLambdaCDM(H0=100 * omegas_gt[0], Ob0=omegas_gt[1], Om0=omegas_gt[2])
            d = cosmo.comoving_distance(zz).value / 1e3  # -> Gpc/h
            spline = UnivariateSpline(d, zz, k=1, s=0)

            # Plot the selection functions
            L = self.L / 1e3
            Lcorner = np.sqrt(3) * L
            zcorner = zz[np.argmin(np.abs(d - Lcorner))]

            # Get linear growth factor from CLASS
            cosmo_dict = cosmo_vector_to_class_dict(omegas_gt)
            cosmo_class = Class()
            cosmo_class.set(cosmo_dict)
            cosmo_class.compute()
            Dz = cosmo_class.get_background()["gr.fac. D"]
            redshifts = cosmo_class.get_background()["z"]
            cosmo_class.struct_cleanup()
            cosmo_class.empty()

            # Define the axis for the plot
            xx = np.linspace(1e-5, Lcorner, 1000)
            zz, res = self.lognormals_z_to_x(
                xx,
                None,
                self.selection_params,
                spline,
            )

            # Call auxiliary plotting routine
            plot_selection_functions(
                xx,
                res,
                None,
                self.selection_params,
                L,
                np.sqrt(3) * L,
                zz=zz,
                zcorner=zcorner,
                path=self.local_select_path[:-3] + ".png",
            )

            # Compute the selection function and save it to disk
            survey_mask = np.load(self.survey_mask_path) if self.survey_mask_path else None
            r = self.r_grid() / 1e3  # Convert to Gpc/h
            _, select_fct = self.lognormals_z_to_x(r, survey_mask, self.selection_params, spline)
            with h5py.File(self.local_select_path, "w") as f:
                f.create_dataset("select_fct", data=select_fct)

            del survey_mask, r, d, zz, spline, select_fct
            collect()
