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

"""This module provides the HiddenBox class, which is used to define
standalone stochastic forward models of large-scale spectroscopic
galaxy surveys.

The HiddenBox class is compatible with the pySELFI package.
"""

import os
from gc import collect
from typing import Callable, Any, List, Optional, Union

import h5py
import numpy as np
from selfisys.global_parameters import DEFAULT_VERBOSE_LEVEL
from selfisys.utils.parser import joinstrs_only


class HiddenBox:
    """This class represents custom forward data model of large-scale
    spectroscopic galaxy surveys."""

    def __init__(
        self,
        k_s: Any,
        P_ss_path: str,
        Pbins_bnd: Any,
        theta2P: Callable,
        P: int,
        size: int,
        L: float,
        G_sim_path: str,
        G_ss_path: str,
        Np0: int,
        Npm0: int,
        fsimdir: str,
        noise_std: Optional[float] = None,
        radial_selection: Optional[str] = None,
        selection_params: Optional[List[Any]] = None,
        observed_density: Optional[float] = None,
        linear_bias: Optional[Union[float, List[float]]] = None,
        norm_csts: Optional[Union[float, List[float]]] = None,
        survey_mask_path: Optional[str] = None,
        local_mask_prefix: Optional[str] = None,
        sim_params: Optional[str] = None,
        eff_redshifts: Optional[List[float]] = None,
        TimeSteps: Optional[List[int]] = None,
        TimeStepDistribution: Optional[str] = None,
        seedphase: Optional[int] = None,
        seednoise: Optional[int] = None,
        fixnoise: bool = False,
        seednorm: Optional[int] = None,
        save_frequency: Optional[int] = None,
        reset: bool = False,
        verbosity: int = DEFAULT_VERBOSE_LEVEL,
        **kwargs,
    ):
        """
        Initialise the HiddenBox.

        Parameters
        ----------
        k_s : array-like
            Vector of input support wavenumbers.
        P_ss_path : str
            Path to the spectrum used to normalise the outputs.
        Pbins_bnd : array-like
            Vector of bin boundaries for the summary statistics.
        theta2P : Callable
            Function to convert theta vectors to initial power spectra.
        P : int
            Dimension of the output summary statistics.
        size : int
            Side length of the simulation box in voxels.
        L : float
            Side length of the simulation box in Mpc/h.
        G_sim_path : str
            Path to the simulation grid.
        G_ss_path : str
            Path to the summary grid.
        Np0 : int
            Number of dark matter particles per spatial dimension.
        Npm0 : int
            Side length of the particle-mesh grid in voxels.
        fsimdir : str
            Output directory.
        noise_std : float, optional
            Standard deviation for the Gaussian noise. Default is
            `None`.
        radial_selection : str, optional
            Type of radial selection mask. Default is `None`,
            corresponding to no radial selection.

            Available options:
                - 'multiple_lognormal': Use multiple log-normal radial
                selection functions.
        selection_params : list, optional
            List of parameters for the radial selection mask, for each
            population. Default is `None`.

            For the 'multiple_lognormal' radial selection,
            `selection_params` is a shape `(3, N_pop)` array comprising
            the three following parameters for each population:
                - selection_std : (float)
                Standard deviation of the distribution, e.g.,
                constant × (1 + z).
                - selection_mean : (float)
                Mean of the distribution in Gpc/h.
                - selection_rescale : (float, optional)
                Individually rescale the distributions by the given
                value. If `None`, the global maximum is normalised to
                `1`.
        observed_density : float, optional
            Mean galaxy density. Default is `None`.
        linear_bias : float or list of float, optional
            First-order linear galaxy biases. If `None`, use the dark
            matter density to compute the summaries. Default is `None`.
        norm_csts : float or list of float, optional
            If not `None`, normalise the output of the hidden box
            accordingly.

            For `radial_selection == 'multiple_lognormal'`, `norm_csts`
            must be a list of `N_pop` values. Default is `None`.
        survey_mask_path : str, optional
            If not `None`, apply the corresponding survey mask to the
            observed field. Default is `None`.
        local_mask_prefix : str, optional
            Prefix for the local copy of the survey mask. If `None`, use
            the default name. Default is `None`.
        sim_params : str, optional
            Set of simulation parameters to be used for Simbelmynë.
            Check `setup_sbmy_parfiles` in `selfisys.sbmy_parser` for
            details. Default is `None`.
        eff_redshifts : list, optional
            Effective redshifts for the time steps. Default is `None`.
        TimeSteps : list, optional
            Number of time steps to reach the corresponding effective
            redshifts. Default is `None`.
        TimeStepDistribution : str, optional
            Path to the Simbelmynë time step distribution file. Default
            is `None`.
        seedphase : int, optional
            Seed to generate the initial white noise realisation.
            Default is `None`.
        seednoise : int, optional
            Initial state of the RNG to generate the noise sequence. If
            `fixnoise` is True, the seed used in `compute_pool` is
            always `seednoise`. If `fixnoise` is False, it is
            `seednoise + i` for realisation `i`. Default is `None`.
        fixnoise : bool, optional
            Whether to fix the noise realisation. If `True`, always use
            `seednoise`. Default is `False`.
        seednorm : int, optional
            Seed used for normalisation. Default is `None`.
        save_frequency : int, optional
            Save the outputs of the hidden box to disk every
            `save_frequency` evaluations. Default is `None`.
        reset : bool, optional
            Whether to always force reset the survey box when the hidden
            `HiddenBox` object is instantiated. Default is `False`.
        verbosity : int, optional
            Verbosity level of the hidden box. `0` is silent, `1` is
            minimal, `2` is verbose. Default is `DEFAULT_VERBOSE_LEVEL`.
        **kwargs
            Additional optional keyword arguments.
        """
        # Mandatory attributes
        self.k_s = k_s
        self.P_ss_path = P_ss_path
        self.Pbins_bnd = Pbins_bnd
        self.theta2P = theta2P
        self.P = P
        self.size = size
        self.L = L
        self.G_sim_path = G_sim_path
        self.G_ss_path = G_ss_path
        self.Np0 = Np0
        self.Npm0 = Npm0
        self.fsimdir = fsimdir

        # Optional attributes
        self.noise_std = noise_std
        self.radial_selection = radial_selection
        self.selection_params = selection_params
        self.observed_density = observed_density
        self.linear_bias = linear_bias
        self.norm_csts = norm_csts
        self.survey_mask_path = survey_mask_path
        self.local_mask_prefix = local_mask_prefix
        self.sim_params = sim_params
        self.eff_redshifts = eff_redshifts
        self.TimeSteps = TimeSteps
        self.TimeStepDistribution = TimeStepDistribution
        self.seedphase = seedphase
        self.seednoise = seednoise
        self.fixnoise = fixnoise
        self.seednorm = seednorm
        self.save_frequency = save_frequency
        self.reset = reset
        self.verbosity = verbosity

        # Additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Compute default values
        self._set_defaults()

        # Create the window function W(n, r) = C(n) * R(r)
        self._init_survey_mask()
        self._init_radial_selection()

    def reset_survey(self):
        """Re-initialise the survey mask C(n) and radial selection
        function R(r).
        """
        self._init_survey_mask(reset=True)
        self._init_radial_selection(reset=True)

    def update(self, **kwargs):
        """Updates the given parameter(s) of the hidden box with the
        given value(s).

        Parameters
        ----------
        **kwargs : dict
            dictionary of parameters to update

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def make_data(
        self,
        cosmo,
        id,
        seedphase,
        seednoise,
        force_powerspectrum=False,
        force_parfiles=False,
        force_sim=False,
        force_cosmo=False,
        force_phase=False,
        d=-1,
        remove_sbmy=True,
        verbosity=None,
        return_g=False,
        RSDs=True,
        prefix_mocks=None,
    ):
        """Generate simulated data based on the given cosmological
        parameters.

        Parameters
        ----------
        cosmo : dict
            Cosmological and infrastructure parameters.
        id : int or str, optional
            Identifier used as a suffix in the file names. Default is
            `0`.
        seedphase : int or list of int
            Seed to generate the initial white noise in Fourier space.
        seednoise : int or list of int
            Seed to generate the noise realisation.
        force_powerspectrum : bool, optional
            Force recomputing the input spectrum. Default is `False`.
        force_parfiles : bool, optional
            Force recomputing the parameter files. Default is `False`.
        force_sim : bool, optional
            Force recomputing the simulation. Default is `False`.
        force_cosmo : bool, optional
            Force recomputing the cosmological parameters. Default is
            `False`.
        force_phase : bool, optional
            Force recomputing the initial phase. Default is `False`.
        d : int, optional
            Direction in the parameter space. Default is `-1`.
        remove_sbmy : bool, optional
            Whether to remove most Simbelmynë output files after use for
            disk space management. Default is `True`.
        verbosity : int, optional
            Verbosity level. If `None`, use self.verbosity. Default is
            `None`.
        return_g : bool, optional
            Whether to return the full field alongside the summary.
            Default is `False`.
        RSDs : bool, optional
            Whether to compute the redshift-space distortions. Default
            is `True`.
        prefix_mocks : str, optional
            Prefix for the mock data. If None, use self.prefix_mocks.
            Default is `None`.

        Returns
        -------
        Phi : ndarray
            Vector of summary statistics.
        g : ndarray or list of ndarray or None
            Observed field(s) if return_g is `True`; otherwise `None`.
        """
        from selfisys.utils.path_utils import get_file_names
        from selfisys.sbmy_interface import get_power_spectrum_from_cosmo

        self._PrintMessage(0, f"Making mock data...", verbosity)
        self._indent()

        names = get_file_names(
            self.fsimdir,
            id,
            self.sim_params,
            self.TimeSteps,
            prefix_mocks or self.prefix_mocks,
            self.gravity_on,
            return_g,
        )

        # Save cosmological parameters
        self._save_cosmo(cosmo, names["fname_cosmo"], force_cosmo)

        # Generate the input initial matter power spectrum
        self._PrintMessage(1, f"Computing initial power spectrum...", verbosity)
        get_power_spectrum_from_cosmo(
            self.L, self.size, cosmo, names["fname_power_spectrum"], force_powerspectrum
        )
        self._PrintMessage(1, f"Computing initial power spectrum done.", verbosity)

        # Generate the simulated galaxy survey data
        self._PrintMessage(1, f"Running the forward model...", verbosity)
        Phi, g = self._aux_hiddenbox(
            d,
            seedphase,
            seednoise,
            names,
            force_parfiles=force_parfiles,
            force_sim=force_sim,
            force_phase=force_phase,
            return_g=return_g,
            RSDs=RSDs,
        )
        self._PrintMessage(1, f"Running the forward model done.", verbosity)

        if remove_sbmy and self.gravity_on:
            self._clean_output(names["fname_outputinitialdensity"])

        self._unindent()
        self._PrintMessage(1, f"Making mock data done.", verbosity)

        return (Phi, g) if return_g else Phi

    def evaluate(
        self,
        theta,
        d,
        seedphase,
        seednoise,
        i=0,
        N=0,
        force_powerspectrum=False,
        force_parfiles=False,
        force_sim=False,
        remove_sbmy=False,
        theta_is_p=False,
        simspath=None,
        check_output=False,
        RSDs=True,
        abc=False,
        cosmo_vect=None,
    ):
        """Evaluate the hidden box for a given input power spectrum.

        The result is deterministic (the phase is fixed), except as it
        is modified by nuisance parameters if any.

        This routine is used by `pySELFI` to compute the gradient of the
        hidden box with respect to the power spectrum, and by ABC-PMC to
        evaluate the forward model.

        Parameters
        ----------
        theta : ndarray
            Power spectrum values at the support wavenumbers.
        d : int
            Direction in parameter space, from `0` to `S`.
        seedphase : int or list of int
            Seed to generate the initial white noise in Fourier space.
        seednoise : int or list of int
            Seed to generate the noise realisation.
        i : int, optional
            Current evaluation index of the hidden box. Default is `0`.
        N : int, optional
            Total number of evaluations of the hidden box. Default is
            `0`.
        force_powerspectrum : bool, optional
            If `True`, force recomputation of the power spectrum at the
            values of the Fourier grid. Default is `False`.
        force_parfiles : bool, optional
            If `True`, overwrite existing parameter files. Default is
            `False`.
        force_sim : bool, optional
            If `True`, rerun the simulation even if the output density
            already exists. Default is `False`.
        remove_sbmy : bool, optional
            If `True`, remove Simbelmynë output files from disk. Default
            is `False`.
        theta_is_p : bool, optional
            Set to `True` if `theta` is already an unnormalised power
            spectrum. Default is `False`.
        simspath : str, optional
            Path to the simulations directory. Default is `None`.
        check_output : bool, optional
            If `True`, check the integrity of the output file and
            recompute if corrupted. Default is `False`.
        RSDs : bool, optional
            Whether to compute redshift-space distortions. Default is
            `True`.
        abc : bool or str, optional
            If not False, remove most output files after evaluation.
            Default is `False`.
        cosmo_vect : ndarray, optional
            Cosmological parameters. Required if `abc` is `True`.

        Returns
        -------
        Phi : ndarray
            Vector of summary statistics.
        """
        from selfisys.utils.path_utils import file_names_evaluate

        if abc and cosmo_vect is None:
            raise ValueError("cosmo_vect must be provided when using ABC.")
        if not abc and cosmo_vect is not None:
            raise ValueError("cosmo_vect must not be provided when not using ABC.")

        if not theta_is_p:
            self._PrintMessage(
                1,
                f"Direction {d}. Evaluating hidden box (index {i}/{N - 1})...",
            )
            self._indent()

        simdir = simspath or self.fsimdir
        simdir_d = os.path.join(simdir, joinstrs_only(["pool/", abc, "d", str(d)]))

        # Get the file names for the current evaluation
        names = file_names_evaluate(
            simdir,
            simdir_d,
            d,
            i,
            self.sim_params,
            self.TimeSteps,
            self.prefix_mocks,
            abc,
            self.gravity_on,
        )
        fname_power_spectrum = names["fname_power_spectrum"]
        fname_simparfile = names["fname_simparfile"]
        fname_whitenoise = names["fname_whitenoise"]
        fname_outputinitialdensity = names["fname_outputinitialdensity"]

        # Interpolate the power spectrum values over the Fourier grid
        self._PrintMessage(1, f"Interpolating spectrum over the Fourier grid...")
        self._power_spectrum_from_theta(
            theta, fname_power_spectrum, theta_is_p, force=force_powerspectrum
        )
        self._PrintMessage(1, f"Interpolating spectrum over the Fourier grid done.")

        # Compute the simulated data
        self._PrintMessage(1, f"Generating observed summary...")
        Phi, _ = self._aux_hiddenbox(
            d,
            seedphase,
            seednoise,
            names,
            force_parfiles,
            force_sim=force_sim,
            force_phase=False,
            return_g=False,
            check_output=check_output,
            RSDs=RSDs,
            sample=cosmo_vect,
        )
        self._PrintMessage(1, f"Generating observed summary done.")

        # Clean up the output files
        if remove_sbmy and self.gravity_on:
            self._clean_output(fname_outputinitialdensity)
        if abc:
            self._clean_output(fname_power_spectrum)
            self._clean_output(fname_simparfile)
            self._clean_output(fname_whitenoise)
            for f in os.listdir(simdir_d):
                if self.TimeSteps is not None:
                    if f.startswith(f"output_realdensity_d{d}_p{i}_{self.TimeSteps[0]}"):
                        self._clean_output(os.path.join(simdir_d, f))
                else:
                    if f.startswith(f"output_realdensity_d{d}_p{i}"):
                        self._clean_output(os.path.join(simdir_d, f))
                if f.startswith(f"output_density_d{d}_p{i}"):
                    self._clean_output(os.path.join(simdir_d, f))
            if not self.gravity_on:
                self._clean_output(fname_outputinitialdensity)

        if not theta_is_p:
            self._unindent()
            self._PrintMessage(
                1,
                f"Direction {d}. Evaluation done (index {i}/{N - 1}).",
            )

        return Phi

    def switch_recompute_pool(self, prefix_mocks=None):
        """Toggle recomputation of the pool for future `compute_pool`
        calls.

        Parameters
        ----------
        prefix_mocks : str, optional
            Prefix for the future simulation files. Default is `None`.
        """
        self._force_recompute_mocks = not self.force_recompute_mocks
        self._prefix_mocks = prefix_mocks if prefix_mocks is not None else None

    def switch_setup(self):
        """Toggle the setup-only mode."""
        self._setup_only = not self.setup_only

    def compute_pool(
        self,
        theta,
        d,
        pool_fname,
        N,
        index=None,
        force_powerspectrum=False,
        force_parfiles=False,
        force_sim=False,
        remove_sbmy=False,
        theta_is_p=False,
        simspath=None,
        bar=False,
    ):
        """Compute a pool of realisations of the hidden box compatible
        with `pySELFI`.

        Parameters
        ----------
        theta : ndarray
            Power spectrum values at the support wavenumbers.
        d : int
            Direction in parameter space, from 0 to S.
        pool_fname : str
            Filename for the pool.
        N : int
            Number of realisations required at the given direction.
        index : int, optional
            Index of a single simulation to run. Default is `None`.
        force_powerspectrum : bool, optional
            If True, force recomputation of the power spectrum. Default
            is `False`.
        force_parfiles : bool, optional
            If True, overwrite existing parameter files. Default is
            `False`.
        force_sim : bool, optional
            If True, rerun the simulation even if the output density
            already exists. Default is `False`.
        remove_sbmy : bool, optional
            If True, remove Simbelmynë output files from disk. Default
            is `False`.
        theta_is_p : bool, optional
            Set to True when `theta` is already an unnormalised power
            spectrum. Default is `False`.
        simspath : str, optional
            Path indicating where to store the simulations. Default is
            `None`.
        bar : bool, optional
            If True, display a progress bar. Default is `False`.

        Returns
        -------
        p : Pool
            Simulation pool object.
        """
        import tqdm.auto as tqdm
        from pyselfi.pool import pool as Pool

        self._PrintMessage(1, f"Computing a pool of realisations of the hidden box...")

        pool_fname = str(pool_fname)
        if self.force_recompute_mocks:
            if os.path.exists(pool_fname):
                os.remove(pool_fname)

        p = Pool(pool_fname, N, retro=False)
        ids = list(range(N)) if index is None else [index]

        def worker(i):
            this_seedphase = self._get_current_seed(self.__global_seedphase, False, i)
            this_seednoise = self._get_current_seed(self.__global_seednoise, self.fixnoise, i)
            Phi = self.evaluate(
                theta,
                d,
                this_seedphase,
                this_seednoise,
                i=i,
                N=N,
                force_powerspectrum=force_powerspectrum,
                force_parfiles=force_parfiles,
                force_sim=force_sim,
                remove_sbmy=remove_sbmy,
                theta_is_p=theta_is_p,
                simspath=simspath,
            )
            p.add_sim(Phi, i)

        iterator = tqdm.tqdm(ids, desc=f"Direction {d}/{self.S}") if bar else ids
        for i in iterator:
            worker(i)

        if index is None:
            p.load_sims()
            p.save_all()
        return p

    def load_pool(self, pool_fname, N):
        """Load a pool of realisations of the hidden box.

        Parameters
        ----------
        pool_fname : str
            Filename of the pool to load.
        N : int
            Number of realisations in the pool.

        Returns
        -------
        p : Pool
            Simulation pool object.
        """
        from pyselfi.pool import pool as Pool

        pool_fname = str(pool_fname)
        p = Pool(pool_fname, N, retro=False)
        p.load_sims()
        p.save_all()
        return p

    @property
    def Npop(self):
        """Number of populations."""
        return self._Npop

    @Npop.setter
    def Npop(self, _):
        """Compute the number of populations."""
        if self.radial_selection == "multiple_lognormal":
            if self.selection_params is not None:
                self._Npop = len(self.selection_params[0])
                if self.linear_bias is not None and len(self.linear_bias) != self._Npop:
                    raise ValueError("Length of linear_bias must match the number of populations.")
                if self.norm_csts is not None and len(self.norm_csts) != self._Npop:
                    raise ValueError("Length of norm_csts must match the number of populations.")
            else:
                raise ValueError(
                    "Selection parameters are required for multiple_lognormal radial selection."
                )
        else:
            self._Npop = 1

    @property
    def gravity_on(self):
        """Whether gravity is enabled."""
        return self._gravity_on

    @gravity_on.setter
    def gravity_on(self, _):
        """Compute and set the gravity status."""
        if self.sim_params:
            self._gravity_on = not self.sim_params.startswith("nograv")
        else:
            self._gravity_on = True

    @property
    def Ntimesteps(self):
        """Number of time steps."""
        return self._Ntimesteps

    @Ntimesteps.setter
    def Ntimesteps(self, _):
        """Compute and set the number of time steps."""
        if self.sim_params:
            if self.sim_params.startswith("splitLPT"):
                self._Ntimesteps = len(self.eff_redshifts)
            elif self.sim_params.startswith("split"):
                self._Ntimesteps = len(self.TimeStepDistribution)
            elif self.sim_params.startswith("nograv"):
                self._Ntimesteps = None
            else:
                self._Ntimesteps = None
        else:
            self._Ntimesteps = None

    @property
    def force_recompute_mocks(self):
        """Whether to force recomputation of mocks."""
        return self._force_recompute_mocks

    @property
    def setup_only(self):
        """Whether to only set up the hidden box."""
        return self._setup_only

    @property
    def prefix_mocks(self):
        """Prefix for the mocks."""
        return self._prefix_mocks

    @property
    def modified_selfi(self):
        """Whether to use the modified selfi."""
        return self._modified_selfi

    @property
    def force_neglect_lightcone(self):
        """Whether to force neglecting the lightcone."""
        return self._force_neglect_lightcone

    @property
    def Psingle(self):
        """Number of summary statistics for each population."""
        return self._Psingle

    @Psingle.setter
    def Psingle(self, _):
        """Dimension of the summary statistics for a single population,
        if relevant.
        """
        if self.radial_selection == "multiple_lognormal":
            self._Psingle = self.P // self.Npop
        else:
            self._Psingle = self.P

    @property
    def __global_seedphase(self):
        return self.seedphase

    @property
    def __global_seednoise(self):
        return self.seednoise

    @property
    def __global_seednorm(self):
        """Global seed for the normalisation constants."""
        return self.seednorm

    def _set_defaults(self):
        """Set default values and ensure consistency of the hidden box
        configuration.
        """
        if not hasattr(self, "modeldir"):
            self.modeldir = os.path.join(self.fsimdir, "model")

        self._force_recompute_mocks = False
        self._setup_only = False
        self._prefix_mocks = None
        self._modified_selfi = True
        self._force_neglect_lightcone = False

        self.S = len(self.k_s)
        # The following attributes are set by the setters
        self.Ntimesteps = None
        self.gravity_on = True
        self.Npop = len(self.linear_bias) or 1
        self.Psingle = None

    def _init_survey_mask(self, reset=False):
        """Initialise the survey mask and save it to disk in binary
        format.

        The survey mask C(n) represents the angular selection function
        of the survey.
        """
        if self.local_mask_prefix:
            mask_filename = f"{self.local_mask_prefix}_survey_mask_binary.h5"
        else:
            mask_filename = "survey_mask_binary.h5"
        self.local_mask_path = os.path.join(self.modeldir, mask_filename)

        if not os.path.exists(self.local_mask_path) or reset or self.reset:
            if self.survey_mask_path is not None:
                survey_mask = np.load(self.survey_mask_path)
                survey_mask_binary = (survey_mask > 0).astype(int)
                del survey_mask
            else:
                survey_mask_binary = np.ones([self.size] * 3, dtype=int)

            with h5py.File(self.local_mask_path, "w") as f:
                f.create_dataset("survey_mask_binary", data=survey_mask_binary)
            del survey_mask_binary
            collect()

    def _init_radial_selection(self, reset=False):
        """Initialise the radial selection function R(r) and save it to
        disk.
        """
        if self.radial_selection == "multiple_lognormal":
            from selfisys.selection_functions import LognormalSelection

            # Set the normalisation constants
            if self.norm_csts is None:
                self.norm_csts = [1.0] * self.Npop

            # Initialise the radial selection functions
            if self.local_mask_prefix:
                filename = f"{self.local_mask_prefix}_select_fct.h5"
            else:
                filename = "select_fct.h5"
            self.local_select_path = os.path.join(self.modeldir, filename)
            LogNorm = LognormalSelection(
                self.L,
                self.selection_params,
                survey_mask_path=self.survey_mask_path,
                local_select_path=self.local_select_path,
                size=self.size,
            )
            LogNorm.init_selection(reset=self.reset or reset)
            del LogNorm
        elif self.radial_selection is None:
            if self.norm_csts is None:
                self.norm_csts = 1.0
        else:
            raise ValueError(
                f"Unknown or unimplemented selection function: {self.radial_selection}"
            )

    def _PrintMessage(self, required_verbosity, message, verbosity=None):
        """Print a message to standard output using PrintMessage from
        pyselfi.utils.
        """
        from selfisys.selfi_interface import PrintMessage

        PrintMessage(required_verbosity, message, verbosity or self.verbosity)

    def _indent(self):
        """Indents the standard output using INDENT from
        pyselfi.utils.
        """
        from selfisys.selfi_interface import indent

        indent()

    def _unindent(self):
        """Unindents the standard output using UNINDENT from
        pyselfi.utils.
        """
        from selfisys.selfi_interface import unindent

        unindent()

    def _save_cosmo(self, cosmo, fname_cosmo, force_cosmo=False):
        """Save cosmological parameters in JSON format.

        Parameters
        ----------
        cosmo : dict
            Cosmological parameters (and infrastructure parameters) to
            be saved.
        fname_cosmo : str
            Name of the output JSON file.
        force_cosmo : bool, optional
            If True, overwrite the file if it already exists. Default is
            `False`.
        """
        from os.path import exists

        if not exists(fname_cosmo) or force_cosmo:
            from json import dump

            with open(fname_cosmo, "w") as fp:
                dump(cosmo, fp)

    def _add_noise(self, g, seednoise, field=None):
        """Add noise to a realisation in physical space.

        Parameters
        ----------
        g : ndarray
            Field to which the noise is added (modified in place).
        seednoise : int or list of int
            Seed to generate the noise realisation.
        field : ndarray, optional
            Selection function to apply to the input field. Default is
            `None`.

        Returns
        -------
        None
        """
        if self.noise_std > 0:
            from numpy import random, sqrt, ones_like
            from h5py import File

            if seednoise is not None:
                rng = random.default_rng(seednoise)
            else:
                raise ValueError("Seednoise must be provided and cannot be None.")

            if field is None:
                field = ones_like(g)

            N = self.observed_density if self.observed_density is not None else 1.0
            noise = rng.normal(
                size=(self.size, self.size, self.size),
                scale=self.noise_std * sqrt(N * field),
            )

            with File(self.local_mask_path, "r") as f:
                mask = f["survey_mask_binary"][:]
                noise *= mask

            g += noise
            del noise
            collect()

    def _get_density_field(self, delta_g_dm, bias):
        """Apply galaxy bias to a dark matter overdensity field.

        Parameters
        ----------
        delta_g_dm : ndarray
            Dark matter density contrast in physical space.
        bias : float
            Linear bias factor.

        Returns
        -------
        delta_g : ndarray
            Galaxy density or overdensity field.
        """
        if bias is None:
            bias = 1.0
        if not isinstance(bias, float):
            raise TypeError("Bias must be a float.")

        if self.observed_density is None:
            delta_g = bias * delta_g_dm
        else:
            delta_g = self.observed_density * (1 + bias * delta_g_dm)

        return delta_g

    def _repaint_and_get_Phi(
        self,
        g_obj,
        norm,
        seednoise,
        bias=None,
        field=None,
        return_g=False,
        AliasingCorr=True,
    ):
        """Repaint a realisation in physical space and compute its
        summary statistics.

        Parameters
        ----------
        g_obj : Field
            Input field object.
        norm : ndarray
            Normalisation constants for the summary statistics.
        seednoise : int or list of int
            Seed to generate the noise realisation.
        bias : float, optional
            Bias to apply to the input field. Default is `None`.
        field : ndarray, optional
            Selection function. Reused as output to save memory
            allocations. Default is `None`.
        return_g : bool, optional
            If True, returns the full field. Default is `False`.
        AliasingCorr : bool, optional
            Whether to apply aliasing correction. Default is `True`.

        Returns
        -------
        Phi : ndarray
            Vector of summary statistics.
        delta_g : ndarray or None
            Realisation in physical space if return_g is True; None
            otherwise.
        """
        import copy
        from selfisys.sbmy_interface import compute_Phi

        if bias is not None and not isinstance(bias, float):
            raise TypeError("Bias must be a float.")

        g_obj_local = copy.deepcopy(g_obj)
        g_obj_local.data = self._get_density_field(g_obj_local.data, bias)
        if field is not None:
            g_obj_local.data *= field

        self._add_noise(g_obj_local.data, seednoise=seednoise, field=field)
        Phi = compute_Phi(
            self.G_ss_path,
            self.P_ss_path,
            g_obj_local,
            norm,
            AliasingCorr,
            self.verbosity,
        )

        delta_g = copy.deepcopy(g_obj_local.data) if return_g else None
        del g_obj_local
        collect()
        return Phi, delta_g

    def _apply_selection(self, fnames_outputdensity, seednoise, return_g=False, AliasingCorr=True):
        """Apply the selection function to a realisation in physical
        space.

        Parameters
        ----------
        fnames_outputdensity : list of str
            Filenames of the output density fields.
        seednoise : int or list of int
            Seed to generate the noise realisation.
        return_g : bool, optional
            If True, returns the full field(s). Default is `False`.
        AliasingCorr : bool, optional
            Whether to apply aliasing correction. Default is `True`.

        Returns
        -------
        Phi_tot : ndarray
            Concatenated summary statistics for each population.
        gs : list of ndarray or None
            List of full fields in physical space if return_g is True;
            None otherwise.
        """
        from pysbmy.field import read_basefield

        split = any(self.sim_params.startswith(s) for s in ["split", "custom"])
        if not split:
            g_obj = read_basefield(fnames_outputdensity[0])
        elif self.force_neglect_lightcone:
            g_obj = read_basefield(fnames_outputdensity[0])
        if self.radial_selection is not None:
            Phi_tot = []
            gs = []
            for ifct in range(self.Npop):
                if split and not self.force_neglect_lightcone:
                    g_obj = read_basefield(fnames_outputdensity[ifct])
                with h5py.File(self.local_select_path, "r") as f:
                    field = f["select_fct"][ifct].astype(float)
                bias = float(self.linear_bias[ifct]) if self.linear_bias is not None else None
                Phi, g_out = self._repaint_and_get_Phi(
                    g_obj,
                    self.norm_csts[ifct],
                    seednoise=seednoise,
                    bias=bias,
                    field=field,
                    return_g=return_g,
                    AliasingCorr=AliasingCorr,
                )
                Phi_tot = np.concatenate([Phi_tot, Phi])
                if return_g:
                    gs.append(g_out)
                del g_out
            result = Phi_tot, gs if return_g else None
        else:
            from h5py import File

            if self.local_mask_path is not None:
                with File(self.local_mask_path, "r") as f:
                    field = f["survey_mask_binary"][:]
            else:
                field = None
            if not split:
                bias = self.linear_bias if self.linear_bias is not None else None
                Phi_tot, delta_g = self._repaint_and_get_Phi(
                    g_obj,
                    self.norm_csts,
                    seednoise=seednoise,
                    bias=bias,
                    field=field,
                    return_g=return_g,
                    AliasingCorr=AliasingCorr,
                )
                result = (Phi_tot, [delta_g]) if return_g else (Phi_tot, None)
            else:
                Phi_tot = []
                gs = []
                for ifct in range(len(self.TimeSteps)):
                    bias = (
                        self.linear_bias
                        if isinstance(self.linear_bias, float)
                        else (self.linear_bias[ifct] if self.linear_bias is not None else None)
                    )
                    g_obj = read_basefield(fnames_outputdensity[ifct])
                    Phi, g_out = self._repaint_and_get_Phi(
                        g_obj,
                        self.norm_csts,
                        seednoise=seednoise,
                        bias=bias,
                        field=field,
                        return_g=return_g,
                        AliasingCorr=AliasingCorr,
                    )
                    Phi_tot = np.concatenate([Phi_tot, Phi])
                    if return_g:
                        gs.append(g_out)
                del g_out
                result = Phi_tot, gs if return_g else None
        del field, g_obj
        collect()
        return result

    def _setup_parfiles(
        self,
        d,
        cosmology,
        file_names,
        force=False,
    ):
        """Sets up Simbelmynë parameter file given the necessary inputs
        (please refer to the Simbelmynë documentation for more details).

        Parameters
        ----------
        d : int
            index giving the direction in parameter space:
            -1 for mock data, 0 for the expansion point, or from 1 to S
        cosmology : array, double, dimension=5
            cosmological parameters
        file_names : dict
            Dictionary containing the names of the input and output
            files.
        force : bool, optional, default=False
            overwrite if files already exists?

        """
        from selfisys.sbmy_interface import setup_sbmy_parfiles

        hiddenbox_params = {
            "Npop": self.Npop,
            "TimeSteps": self.TimeSteps,
            "eff_redshifts": self.eff_redshifts,
            "sim_params": self.sim_params,
            "Ntimesteps": self.Ntimesteps,
            "TimeStepDistribution": self.TimeStepDistribution,
            "modified_selfi": self.modified_selfi,
            "Np0": self.Np0,
            "Npm0": self.Npm0,
            "size": self.size,
            "L": self.L,
            "fsimdir": self.fsimdir,
        }
        setup_sbmy_parfiles(
            d,
            cosmology,
            file_names,
            hiddenbox_params,
            force,
        )

    def _run_sim(
        self,
        fname_simparfile,
        fname_simlogs,
        fnames_outputdensity,
        force_sim=False,
        check_output=False,
    ):
        """Run a simulation with Simbelmynë.

        Parameters
        ----------
        fname_simparfile : str
            Name of the input parameter file.
        fname_simlogs : str
            Name of the output Simbelmynë logs.
        fnames_outputdensity : list of str
            Names of the output density fields to be written.
        force_sim : bool, optional
            If True, force recomputation if output density already
            exists.
            Default is `False`.
        check_output : bool, optional
            If True, check the integrity of the output files and
            recompute if corrupted. Default is `False`.
        """
        from glob import glob
        from selfisys.utils.parser import check_files_exist

        split = self.sim_params.startswith("split")
        if not check_files_exist(fnames_outputdensity) or force_sim:
            from pysbmy import pySbmy

            if not split:
                pySbmy(f"{fname_simparfile}_{self.Npop}.sbmy", fname_simlogs)
            else:
                for i in range(self.Ntimesteps):
                    pySbmy(f"{fname_simparfile}_pop{i}.sbmy", fname_simlogs)
        elif check_output:
            from pysbmy.field import read_basefield

            try:
                for fname in fnames_outputdensity:
                    g = read_basefield(fname)
                del g
                collect()
            except Exception:
                from pysbmy import pySbmy

                for fname in fnames_outputdensity:
                    os.remove(fname)
                pySbmy(fname_simparfile, fname_simlogs)

        # Workaround to remove unwanted Simbelmynë outputs from disk
        temp_files = glob(os.path.join(self.fsimdir, "data", "cola_kick_*.h5"))
        for f in temp_files:
            os.remove(f)

        if split:
            temp_files = glob(os.path.join(self.fsimdir, "data", "cola_snapshot_*.h5"))
            temp_files += glob(os.path.join(self.fsimdir, "data", "lpt_psi*.h5"))
            for f in temp_files:
                os.remove(f)

    def _compute_mocks(
        self,
        seednoise,
        fnames_input,
        fname_mocks,
        return_g,
        fname_g=None,
        AliasingCorr=True,
    ):
        """Apply galaxy bias, observational effects, and compute the
        summary statistics.

        Parameters
        ----------
        seednoise : int or list of int
            Seed to generate the noise realisation.
        fnames_input : list of str
            Filenames of the input density fields.
        fname_mocks : str
            Filename to save the computed summary statistics.
        return_g : bool
            If True, return the observed field(s).
        fname_g : str, optional
            Filename to save the observed field(s) if return_g is True.
        AliasingCorr : bool, optional
            Whether to apply aliasing correction. Default is `True`.

        Returns
        -------
        Phi : ndarray
            Vector of summary statistics.
        g_out : ndarray or list of ndarray or None
            Observed field(s) if return_g is True; otherwise None.
        """

        if return_g and fname_g is None:
            raise ValueError("Filename for the observed field must be provided.")

        if (not os.path.exists(fname_mocks)) or self.force_recompute_mocks:
            Phi, g_out = self._apply_selection(
                fnames_input,
                seednoise=seednoise,
                return_g=return_g,
                AliasingCorr=AliasingCorr,
            )
            with h5py.File(fname_mocks, "w") as f:
                f.create_dataset("Phi", data=Phi)
            if return_g:
                with h5py.File(fname_g, "w") as f:
                    f.create_dataset("g", data=g_out)
        else:
            self._PrintMessage(1, f"Using existing mock file: {fname_mocks}")
            with h5py.File(fname_mocks, "r") as f:
                Phi = f["Phi"][:]
            if return_g:
                with h5py.File(fname_g, "r") as f:
                    g_out = f["g"][:]
            else:
                g_out = None

        return Phi, g_out

    def _sample_omega(self, seed):
        """Sample cosmological parameters from the prior.

        Parameters
        ----------
        seed : int or list of int
            Seed for the random number generator.

        Returns
        -------
        omega_sample : ndarray
            Sampled cosmological parameters.
        """
        from selfisys.utils.tools import sample_omega_from_prior
        from selfisys.global_parameters import planck_mean, planck_cov

        ids = list(range(len(planck_mean)))
        return sample_omega_from_prior(1, planck_mean, planck_cov, ids, seed=seed)[0]

    def _aux_hiddenbox(
        self,
        d,
        seedphase,
        seednoise,
        file_names,
        force_parfiles=False,
        force_sim=False,
        force_phase=False,
        return_g=False,
        check_output=False,
        RSDs=True,
        sample=None,
    ):
        """Generate observations from the input initial matter power
        spectrum.

        Parameters
        ----------
        d : int
            Index indicating the direction in parameter space (-1 for
            mock data, 0 for the expansion point, or from 1 to S for the
            gradient directions).
        seedphase : int or list of int
            Seed to generate the initial white noise in Fourier space.
        seednoise : int or list of int
            Seed to generate the observational noise realisation.
        file_names : dict
            Dictionary containing the names of the input and output
            files.
        force_parfiles : bool, optional
            If True, force recomputation of the parameter files. Default
            is `False`.
        force_sim : bool, optional
            If True, force recomputation of the simulation. Default is
            `False`.
        force_phase : bool, optional
            If True, force recomputation of the phase. Default is
            `False`.
        return_g : bool, optional
            If True, return the full realisation in physical space.
            Default is `False`.
        check_output : bool, optional
            If True, check the integrity of the output file and
            recompute if corrupted. Default is `False`.
        RSDs : bool, optional
            If True, include redshift-space distortions. Default is
            `True`.
        sample : ndarray, optional
            Cosmological parameters sample. If `None`, sample from the
            prior. Default is `None`.

        Returns
        -------
        result : tuple
            A tuple containing:
            - Phi : (ndarray)
                Summary statistics for each population, concatenated.
            - g : (ndarray or list of ndarray or None)
                List of observed fields if return_g is True; None
                otherwise.
        """
        fname_power_spectrum = file_names["fname_power_spectrum"]
        fname_simparfile = file_names["fname_simparfile"]
        fname_whitenoise = file_names["fname_whitenoise"]
        seedname_whitenoise = file_names["seedname_whitenoise"]
        fname_outputinitialdensity = file_names["fname_outputinitialdensity"]
        fnames_outputrealspacedensity = file_names["fnames_outputrealspacedensity"]
        fnames_outputdensity = file_names["fnames_outputdensity"]
        fname_simlogs = file_names["fname_simlogs"]
        fname_mocks = file_names["fname_mocks"]
        fname_g = file_names["fname_g"] if return_g else None

        sample = self._sample_omega(seedphase) if sample is None else sample

        if self.gravity_on:
            from selfisys.sbmy_interface import generate_white_noise_Field

            self._setup_parfiles(
                d,
                sample,
                file_names,
                force_parfiles,
            )
            generate_white_noise_Field(
                self.L,
                self.size,
                seedphase,
                fname_whitenoise,
                seedname_whitenoise,
                force_phase,
            )

        if not self.setup_only:
            if self.gravity_on:
                self._run_sim(
                    fname_simparfile,
                    fname_simlogs,
                    fnames_outputdensity,
                    force_sim,
                    check_output,
                )
                fnames = fnames_outputdensity if RSDs else fnames_outputrealspacedensity
            else:
                from selfisys.grf import primordial_grf

                primordial_grf(
                    self.L,
                    self.size,
                    seedphase,
                    fname_power_spectrum,
                    fname_outputinitialdensity,
                    force_sim,
                    False,
                    self.verbosity,
                )
                fnames = [fname_outputinitialdensity]

            result = self._compute_mocks(seednoise, fnames, fname_mocks, return_g, fname_g)
        else:
            result = [], []  # lists are mandatory for compatibility

        return result

    def _power_spectrum_from_theta(
        self, theta, fname_power_spectrum, theta_is_p=False, force=False
    ):
        """Compute the power spectrum values using spline interpolation.

        Parameters
        ----------
        theta : ndarray
            Vector of power spectrum values at the support wavenumbers.
        fname_power_spectrum : str
            Name of the input/output power spectrum file.
        theta_is_p : bool, optional
            If True, theta is already an unnormalised power spectrum.
            Default is `False`.
        force : bool, optional
            If True, force recomputation. Default is `False`.
        """
        from pysbmy.power import PowerSpectrum

        if (not os.path.exists(fname_power_spectrum)) or force:
            from scipy.interpolate import InterpolatedUnivariateSpline
            from pysbmy.power import FourierGrid

            PP = theta if theta_is_p else self.theta2P(theta)
            Spline = InterpolatedUnivariateSpline(self.k_s, PP, k=5)
            G_sim = FourierGrid.read(self.G_sim_path)
            power_spectrum = Spline(G_sim.k_modes)
            power_spectrum[0] = 0.0
            P = PowerSpectrum.from_FourierGrid(G_sim, powerspectrum=power_spectrum)
            P.write(fname_power_spectrum)
            del G_sim
            collect()

    def _clean_output(self, fname_output):
        """Remove a file from disk if it exists.

        Parameters
        ----------
        fname_output : str
            Name of the file to remove.
        """
        if fname_output is not None and os.path.exists(fname_output):
            os.remove(fname_output)

    def _get_current_seed(self, parent_seed, fixed_seed, i):
        """Return the current seed for the i-th realisation.

        Parameters
        ----------
        parent_seed : int or list of int
            The parent seed.
        fixed_seed : bool
            If True, use the parent seed directly.
        i : int
            Index of the current realisation.

        Returns
        -------
        this_seed : int or list of int
            The seed for the current realisation.
        """
        if fixed_seed:
            this_seed = parent_seed
        else:
            this_seed = [i, parent_seed]
        return this_seed
