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

"""Simbelmynë-related functions for the SelfiSys pipeline.
"""

import os
import gc
from typing import Optional, List, Tuple

from selfisys.utils.logger import getCustomLogger, INDENT, UNINDENT

logger = getCustomLogger(__name__)


def get_power_spectrum_from_cosmo(
    L,
    size,
    cosmo,
    fname_power_spectrum,
    force=False,
):
    """
    Compute a power spectrum from cosmological parameters and save it to
    disk.

    Parameters
    ----------
    L : float
        Size of the simulation box (in Mpc/h).
    size : int
        Number of grid points along each axis.
    cosmo : dict
        Cosmological parameters (and infrastructure parameters).
    fname_power_spectrum : str
        Name (including path) of the power spectrum file to read/write.
    force : bool, optional
        If True, forces recomputation even if the file exists. Default
        is False.

    Raises
    ------
    OSError
        If file writing fails or the directory path is invalid.
    RuntimeError
        For unexpected issues during power spectrum computation.
    """
    if not os.path.exists(fname_power_spectrum) or force:
        from pysbmy.power import PowerSpectrum

        try:
            logger.debug("Computing power spectrum for L=%.2f, size=%d", L, size)
            P = PowerSpectrum(L, L, L, size, size, size, cosmo)
            P.write(fname_power_spectrum)
            logger.debug("Power spectrum written to %s", fname_power_spectrum)
        except OSError as e:
            logger.error("File write error at %s: %s", fname_power_spectrum, str(e))
            raise
        except Exception as e:
            logger.critical("Unexpected error in power spectrum computation: %s", str(e))
            raise RuntimeError("get_power_spectrum_from_cosmo failed.") from e
        finally:
            gc.collect()


def compute_Phi(
    G_ss_path,
    P_ss_path,
    g_obj,
    norm,
    AliasingCorr=True,
    verbosity=1,
):
    """
    Compute the summary statistics from a field object, based on a
    provided summary-statistics Fourier grid and baseline spectrum.

    Parameters
    ----------
    G_ss_path : str
        Path to the FourierGrid file used for summary-statistics.
    P_ss_path : str
        Path to the baseline power spectrum file for normalisation.
    g_obj : Field
        Input field object from which to compute summary statistics.
    norm : ndarray
        Normalisation constants for the summary statistics.
    AliasingCorr : bool, optional
        Whether to apply aliasing correction. Default is True.
    verbosity : int, optional
        Verbosity level (0=quiet, 1=normal, 2=debug). Default 1.

    Returns
    -------
    Phi : ndarray
        Vector of summary statistics.

    Raises
    ------
    OSError
        If file reading fails at G_ss_path or P_ss_path.
    RuntimeError
        If unexpected issues occur during computation.
    """
    from pysbmy.correlations import get_autocorrelation
    from pysbmy.power import FourierGrid, PowerSpectrum
    from pysbmy import c_double
    from io import BytesIO

    try:
        logger.debug("Reading FourierGrid from %s", G_ss_path)
        G_ss = FourierGrid.read(G_ss_path)

        if verbosity > 1:
            Pk, _ = get_autocorrelation(g_obj, G_ss, AliasingCorr=AliasingCorr)
        else:
            from selfisys.utils.low_level import stdout_redirector

            f = BytesIO()
            with stdout_redirector(f):
                Pk, _ = get_autocorrelation(g_obj, G_ss, AliasingCorr=AliasingCorr)
            f.close()

        logger.debug("Reading baseline PowerSpectrum from %s", P_ss_path)
        P_ss = PowerSpectrum.read(P_ss_path)
        Phi = Pk / (norm * P_ss.powerspectrum)

        del G_ss, P_ss
        gc.collect()
        return Phi.astype(c_double)
    except OSError as e:
        logger.error("File not found or inaccessible: %s", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error in compute_Phi: %s", str(e))
        raise RuntimeError("compute_Phi failed.") from e
    finally:
        gc.collect()


def generate_white_noise_Field(
    L,
    size,
    seedphase,
    fname_whitenoise,
    seedname_whitenoise,
    force_phase=False,
):
    """
    Generate a white noise realisation in physical space and write it to
    disk.

    Parameters
    ----------
    L : float
        Size of the simulation box (in Mpc/h).
    size : int
        Number of grid points along each axis.
    seedphase : int or list of int
        User-provided seed to generate the initial white noise.
    fname_whitenoise : str
        File path to write the white noise realisation.
    seedname_whitenoise : str
        File path to write the seed state of the RNG.
    force_phase : bool, optional
        If True, forces regeneration of the random phases. Default is
        False.

    Raises
    ------
    OSError
        If file writing fails or directory paths are invalid.
    RuntimeError
        For unexpected issues.
    """
    if not os.path.exists(fname_whitenoise) or force_phase:
        import numpy as np
        from pysbmy.field import BaseField

        try:
            logger.debug("Generating white noise for L=%.2f, size=%d", L, size)
            rng = np.random.default_rng(seedphase)

            logger.debug("Saving RNG state to %s", seedname_whitenoise)
            np.save(seedname_whitenoise, rng.bit_generator.state)
            with open(seedname_whitenoise + ".txt", "w") as f:
                f.write(str(rng.bit_generator.state))

            data = rng.standard_normal(size=size**3)
            wn = BaseField(L, L, L, 0, 0, 0, 1, size, size, size, data)
            del data

            wn.write(fname_whitenoise)
            logger.debug("White noise field written to %s", fname_whitenoise)
            del wn
        except OSError as e:
            logger.error("Writing white noise failed at '%s': %s", fname_whitenoise, str(e))
            raise
        except Exception as e:
            logger.critical("Unexpected error in generate_white_noise_Field: %s", str(e))
            raise RuntimeError("generate_white_noise_Field failed.") from e
        finally:
            gc.collect()


def setup_sbmy_parfiles(
    d,
    cosmology,
    file_names,
    hiddenbox_params,
    force=False,
):
    """Set up Simbelmynë parameter file (please refer to the Simbelmynë
    documentation for more details).

    Parameters
    ----------
    d : int
        Index (from 1 to S) specifying a direction in parameter space, 0
        for the expansion point, or -1 for mock data.
    cosmology : array, double, dimension=5
        Cosmological parameters.
    file_names : dict
        Dictionary containing the names of the input/output files for
        the simulation.
    hiddenbox_params : dict
        See the `HiddenBox` class for more details.
    force : bool, optional, default=False
        If True, forces recompute the simulation parameter files.

    """
    from os.path import exists

    fname_simparfile = file_names["fname_simparfile"]
    fname_power_spectrum = file_names["fname_power_spectrum"]
    fname_whitenoise = file_names["fname_whitenoise"]
    fname_outputinitialdensity = file_names["fname_outputinitialdensity"]
    fnames_outputrealspacedensity = file_names["fnames_outputrealspacedensity"]
    fnames_outputdensity = file_names["fnames_outputdensity"]
    fnames_outputLPTdensity = file_names["fnames_outputLPTdensity"]

    Npop = hiddenbox_params["Npop"]
    Np0 = hiddenbox_params["Np0"]
    Npm0 = hiddenbox_params["Npm0"]
    size = hiddenbox_params["size"]
    L = hiddenbox_params["L"]
    Ntimesteps = hiddenbox_params["Ntimesteps"]
    sim_params = hiddenbox_params["sim_params"]
    eff_redshifts = hiddenbox_params["eff_redshifts"]
    TimeSteps = hiddenbox_params["TimeSteps"]
    TimeStepDistribution = hiddenbox_params["TimeStepDistribution"]
    modified_selfi = hiddenbox_params["modified_selfi"]
    fsimdir = hiddenbox_params["fsimdir"]

    if not exists(fname_simparfile + "_{}.sbmy".format(Npop)) or force:
        from pysbmy import param_file
        from re import search
        from selfisys.global_parameters import BASEID_OBS

        if TimeSteps is not None and eff_redshifts is None:
            raise ValueError("TimeSteps must be provided if eff_redshifts is None.")

        regex = r"([a-zA-Z]+)(\d+)?([a-zA-Z]+)?(\d+)?([a-zA-Z]+)?"
        m = search(regex, sim_params)
        if m.group(1) == "std":
            # Single LPT+COLA/PM Simbelmynë data card with linear time
            # stepping
            if m.group(2)[0] == "0":
                RedshiftLPT = float("0." + m.group(2)[1:])
            else:
                RedshiftLPT = int(m.group(2))

            RedshiftFCs = 0.0
            WriteLPTSnapshot = 0
            WriteLPTDensity = 0
            match m.group(3):
                case "RSD":
                    ModulePMCOLA = 0
                    EvolutionMode = 2
                    NumberOfTimeSteps = 0
                    RedshiftFCs = RedshiftLPT
                    NonLinearRSD = 1
                case "PM":
                    ModulePMCOLA = 1
                    EvolutionMode = 1
                    NumberOfTimeSteps = m.group(4)
                    NonLinearRSD = 0 if (m.group(5) and m.group(5)[:3] == "lin") else 1
                case "COLA":
                    ModulePMCOLA = 1
                    EvolutionMode = 2
                    NumberOfTimeSteps = m.group(4)
                    NonLinearRSD = 0 if (m.group(5) and m.group(5)[:3] == "lin") else 1
                case _:
                    raise ValueError("sim_params = {} not valid".format(sim_params))
            NumberOfTimeSteps = int(m.group(4)) if m.group(4) is not None else 0
        elif m.group(1) == "custom":
            # Single LPT+COLA/PM Simbelmynë card with user-provided time
            # stepping object
            RedshiftLPT = int(m.group(2))
            match m.group(3):
                case None:
                    ModulePMCOLA = 0
                    EvolutionMode = 2
                case "PM":
                    ModulePMCOLA = 1
                    EvolutionMode = 1
                    NonLinearRSD = 0 if (m.group(5) and m.group(5)[:3] == "lin") else 1
                case "COLA":
                    ModulePMCOLA = 1
                    EvolutionMode = 2
                    NonLinearRSD = 0 if (m.group(5) and m.group(5)[:3] == "lin") else 1
                case _:
                    raise ValueError("sim_params = {} not valid".format(sim_params))
            if TimeStepDistribution is None:
                raise ValueError("TimeStepDistribution must be provided for 'custom'.")
        elif m.group(1) == "splitLPT":
            # Use as many Simbelmynë data cards as there are populations
            # of galaxies
            if eff_redshifts is None:
                raise ValueError("eff_redshifts must be provided for 'splitLPT'.")
            elif len(eff_redshifts) != Ntimesteps:
                raise ValueError("len(eff_redshifts) != Ntimesteps")
        elif m.group(1) == "split":
            # Use as many Simbelmynë data cards as there are populations
            # of galaxies
            if TimeStepDistribution is None:
                raise ValueError("TimeStepDistribution must be for 'split'.")
            if eff_redshifts is None:
                raise ValueError("eff_redshifts must be provided for 'split'.")
            elif len(eff_redshifts) != Ntimesteps:
                raise ValueError("len(eff_redshifts) != Ntimesteps")

            RedshiftLPT = int(m.group(2))
            match m.group(3):
                case "RSD":
                    ModulePMCOLA = 1
                    EvolutionMode = 2
                    NonLinearRSD = 1
                case _:
                    raise ValueError("sim_params = {} not valid".format(sim_params))
            NumberOfTimeSteps = int(m.group(4)) if m.group(4) is not None else 0
        else:
            raise ValueError("sim_params = {} not valid" + sim_params)

        if sim_params[-3:] == BASEID_OBS:
            from selfisys.global_parameters import (
                h_obs as h,
                Omega_b_obs as Omega_b,
                Omega_m_obs as Omega_m,
                nS_obs as nS,
                sigma8_obs as sigma8,
            )
        else:
            if modified_selfi:
                # Treat the cosmological parameters as nuisance
                # parameters within the hidden box forward model
                h, Omega_b, Omega_m, nS, sigma8 = cosmology
            else:
                # Fix the fiducial cosmology within the hidden box
                from selfisys.global_parameters import (
                    h_planck as h,
                    Omega_b_planck as Omega_b,
                    Omega_m_planck as Omega_m,
                    nS_planck as nS,
                    sigma8_planck as sigma8,
                )

        if d < 0:  # -1 for mock data, -2 to recompute the observations
            WriteInitialConditions = 1
            WriteDensities = 1  # also write real space density fields
        else:  # d=0 for expansion point or d>0 for the gradients
            WriteInitialConditions = 0
            WriteDensities = 1  # also write real space density fields

        if m.group(1) == "std":
            S = param_file(  ## Module LPT ##
                ModuleLPT=1,
                # Basic setup:
                Particles=Np0,
                Mesh=size,
                BoxSize=L,
                corner0=0.0,
                corner1=0.0,
                corner2=0.0,
                # Initial conditions:
                ICsMode=1,
                WriteICsRngState=0,
                WriteInitialConditions=WriteInitialConditions,
                InputWhiteNoise=fname_whitenoise,
                OutputInitialConditions=fname_outputinitialdensity,
                # Power spectrum:
                InputPowerSpectrum=fname_power_spectrum,
                # Final conditions for LPT:
                RedshiftLPT=RedshiftLPT,
                WriteLPTSnapshot=WriteLPTSnapshot,
                WriteLPTDensity=WriteLPTDensity,
                OutputLPTDensity=fnames_outputLPTdensity,
                ####################
                ## Module PM/COLA ##
                ####################
                ModulePMCOLA=ModulePMCOLA,
                EvolutionMode=EvolutionMode,  # 1 for PM, 2 for COLA
                ParticleMesh=Npm0,
                NumberOfTimeSteps=NumberOfTimeSteps,
                # Final snapshot:
                RedshiftFCs=RedshiftFCs,
                WriteFinalSnapshot=0,
                WriteFinalDensity=WriteDensities,
                OutputFinalDensity=fnames_outputrealspacedensity[0],
                #########
                ## RSD ##
                #########
                ModuleRSD=1,
                WriteIntermediaryRSD=0,
                DoNonLinearMapping=NonLinearRSD,
                WriteRSDensity=1,
                OutputRSDensity=fnames_outputdensity[0],
                #############################
                ## Cosmological parameters ##
                #############################
                h=h,
                Omega_q=1.0 - Omega_m,
                Omega_b=Omega_b,
                Omega_m=Omega_m,
                Omega_k=0.0,
                n_s=nS,
                sigma8=sigma8,
                w0_fld=-1.0,
                wa_fld=0.0,
            )
            S.write(fname_simparfile + "_{}.sbmy".format(Npop))
        elif m.group(1) == "custom":
            RedshiftFCs = eff_redshifts
            fname_outputdensity = (
                fnames_outputdensity[0][: fnames_outputdensity[0].rfind("_")] + ".h5"
            )
            S = param_file(  ## Module LPT ##
                ModuleLPT=1,
                # Basic setup:
                Particles=Np0,
                Mesh=size,
                BoxSize=L,
                corner0=0.0,
                corner1=0.0,
                corner2=0.0,
                # Initial conditions:
                ICsMode=1,
                WriteICsRngState=0,
                WriteInitialConditions=WriteInitialConditions,
                InputWhiteNoise=fname_whitenoise,
                OutputInitialConditions=fname_outputinitialdensity,
                # Power spectrum:
                InputPowerSpectrum=fname_power_spectrum,
                # Final conditions for LPT:
                RedshiftLPT=RedshiftLPT,
                WriteLPTSnapshot=0,
                WriteLPTDensity=0,
                ####################
                ## Module PM/COLA ##
                ####################
                ModulePMCOLA=ModulePMCOLA,
                EvolutionMode=EvolutionMode,  # 1 for PM, 2 for COLA
                ParticleMesh=Npm0,
                OutputKickBase=fsimdir + "/data/cola_kick_",
                # Final snapshot:
                RedshiftFCs=RedshiftFCs,
                WriteFinalSnapshot=0,
                WriteFinalDensity=0,
                OutputFinalDensity=fnames_outputrealspacedensity[0],
                # Intermediate snapshots:
                WriteSnapshots=0,
                WriteDensities=WriteDensities,
                OutputDensitiesBase=fnames_outputrealspacedensity[0][
                    : fnames_outputrealspacedensity[0].rfind("_")
                ]
                + "_",
                OutputDensitiesExt=".h5",
                ############################
                ## Time step distribution ##
                ############################
                TimeStepDistribution=TimeStepDistribution,
                ModifiedDiscretization=1,  # Modified KD discretisation
                n_LPT=-2.5,  # Exponent for the Ansatz in KD operators
                #########
                ## RSD ##
                #########
                ModuleRSD=1,
                WriteIntermediaryRSD=1,
                DoNonLinearMapping=NonLinearRSD,
                WriteRSDensity=1,
                OutputRSDensity=fname_outputdensity,
                #############################
                ## Cosmological parameters ##
                #############################
                h=h,
                Omega_q=1.0 - Omega_m,
                Omega_b=Omega_b,
                Omega_m=Omega_m,
                Omega_k=0.0,
                n_s=nS,
                sigma8=sigma8,
                w0_fld=-1.0,
                wa_fld=0.0,
            )
            S.write(fname_simparfile + "_{}.sbmy".format(Npop))
        elif m.group(1) == "split":
            datadir = fsimdir + "/data/"
            RedshiftFCs = eff_redshifts[0]

            # Write the parameter file for the first simulation
            S = param_file(
                ################
                ## Module LPT ##
                ################
                ModuleLPT=1,
                # Basic setup:
                Particles=Np0,
                Mesh=size,
                BoxSize=L,
                corner0=0.0,
                corner1=0.0,
                corner2=0.0,
                # Initial conditions:
                ICsMode=1,
                WriteICsRngState=0,
                WriteInitialConditions=WriteInitialConditions,
                InputWhiteNoise=fname_whitenoise,
                OutputInitialConditions=fname_outputinitialdensity,
                # Power spectrum:
                InputPowerSpectrum=fname_power_spectrum,
                # Final conditions for LPT:
                RedshiftLPT=RedshiftLPT,
                WriteLPTSnapshot=0,
                WriteLPTDensity=0,
                ####################
                ## Module PM/COLA ##
                ####################
                ModulePMCOLA=ModulePMCOLA,
                EvolutionMode=EvolutionMode,
                ParticleMesh=Npm0,
                OutputKickBase=datadir + "cola_kick_0_",
                # Final snapshot:
                RedshiftFCs=RedshiftFCs,
                WriteFinalSnapshot=1,
                OutputFinalSnapshot=datadir + "cola_snapshot_0.gadget3",
                WriteFinalDensity=1,
                OutputFinalDensity=fnames_outputrealspacedensity[0],
                WriteLPTDisplacements=1,
                OutputPsiLPT1=datadir + "lpt_psi1_0.h5",
                OutputPsiLPT2=datadir + "lpt_psi2_0.h5",
                ############################
                ## Time step distribution ##
                ############################
                TimeStepDistribution=TimeStepDistribution[0],
                ModifiedDiscretization=1,
                #########
                ## RSD ##
                #########
                ModuleRSD=1,
                WriteIntermediaryRSD=0,
                DoNonLinearMapping=NonLinearRSD,
                WriteRSDensity=1,
                OutputRSDensity=fnames_outputdensity[0],
                #############################
                ## Cosmological parameters ##
                #############################
                h=h,
                Omega_q=1.0 - Omega_m,
                Omega_b=Omega_b,
                Omega_m=Omega_m,
                Omega_k=0.0,
                n_s=nS,
                sigma8=sigma8,
                w0_fld=-1.0,
                wa_fld=0.0,
            )
            S.write(fname_simparfile + "_pop0.sbmy")

            for i in range(1, Ntimesteps):
                RedshiftFCs = eff_redshifts[i]

                S = param_file(
                    ModuleLPT=0,
                    # Basic setup:
                    Particles=Np0,
                    Mesh=size,
                    BoxSize=L,
                    corner0=0.0,
                    corner1=0.0,
                    corner2=0.0,
                    InputPsiLPT1=datadir + "lpt_psi1_0.h5",
                    InputPsiLPT2=datadir + "lpt_psi2_0.h5",
                    ####################
                    ## Module PM/COLA ##
                    ####################
                    ModulePMCOLA=ModulePMCOLA,
                    InputPMCOLASnapshot=datadir + "cola_snapshot_{:d}.gadget3".format(i - 1),
                    EvolutionMode=EvolutionMode,
                    ParticleMesh=Npm0,
                    OutputKickBase=datadir + "cola_kick_{:d}_".format(i),
                    # Final snapshot:
                    RedshiftFCs=RedshiftFCs,
                    WriteFinalSnapshot=1,
                    OutputFinalSnapshot=datadir + "cola_snapshot_{:d}.gadget3".format(i),
                    WriteFinalDensity=1,
                    OutputFinalDensity=fnames_outputrealspacedensity[::-1][i],
                    WriteLPTDisplacements=0,
                    ############################
                    ## Time step distribution ##
                    ############################
                    TimeStepDistribution=TimeStepDistribution[i],
                    ModifiedDiscretization=1,
                    #########
                    ## RSD ##
                    #########
                    ModuleRSD=1,
                    WriteIntermediaryRSD=0,
                    DoNonLinearMapping=NonLinearRSD,
                    WriteRSDensity=1,
                    OutputRSDensity=fnames_outputdensity[i],
                    #############################
                    ## Cosmological parameters ##
                    #############################
                    h=h,
                    Omega_q=1.0 - Omega_m,
                    Omega_b=Omega_b,
                    Omega_m=Omega_m,
                    Omega_k=0.0,
                    n_s=nS,
                    sigma8=sigma8,
                    w0_fld=-1.0,
                    wa_fld=0.0,
                )
                S.write(fname_simparfile + "_pop{}.sbmy".format(i))
        elif m.group(1) == "splitLPT":
            datadir = fsimdir + "/data/"
            RedshiftLPT = eff_redshifts[0]
            RedshiftFCs = eff_redshifts[0]

            # Write the parameter file for the first simulation
            S = param_file(
                ################
                ## Module LPT ##
                ################
                ModuleLPT=1,
                # Basic setup:
                Particles=Np0,
                Mesh=size,
                BoxSize=L,
                corner0=0.0,
                corner1=0.0,
                corner2=0.0,
                # Initial conditions:
                ICsMode=1,
                WriteICsRngState=0,
                InputWhiteNoise=fname_whitenoise,
                OutputInitialConditions=fname_outputinitialdensity,
                InputPowerSpectrum=fname_power_spectrum,
                # Final conditions for LPT:
                RedshiftLPT=RedshiftLPT,
                WriteLPTSnapshot=0,
                WriteLPTDensity=0,
                # Final snapshot:
                RedshiftFCs=RedshiftFCs,
                WriteFinalDensity=0,
                WriteLPTDisplacements=0,
                #########
                ## RSD ##
                #########
                ModuleRSD=1,
                WriteIntermediaryRSD=0,
                DoNonLinearMapping=NonLinearRSD,
                WriteRSDensity=1,
                OutputRSDensity=fnames_outputdensity[0],
                #############################
                ## Cosmological parameters ##
                #############################
                h=h,
                Omega_q=1.0 - Omega_m,
                Omega_b=Omega_b,
                Omega_m=Omega_m,
                Omega_k=0.0,
                n_s=nS,
                sigma8=sigma8,
                w0_fld=-1.0,
                wa_fld=0.0,
            )
            S.write(fname_simparfile + "_pop0.sbmy")

            for i in range(1, Ntimesteps):
                RedshiftLPT = eff_redshifts[i]
                RedshiftFCs = eff_redshifts[i]

                S = param_file(
                    ################
                    ## Module LPT ##
                    ################
                    ModuleLPT=1,
                    # Basic setup:
                    Particles=Np0,
                    Mesh=size,
                    BoxSize=L,
                    corner0=0.0,
                    corner1=0.0,
                    corner2=0.0,
                    # Initial conditions:
                    ICsMode=1,
                    WriteICsRngState=0,
                    InputWhiteNoise=fname_whitenoise,
                    OutputInitialConditions=fname_outputinitialdensity,
                    InputPowerSpectrum=fname_power_spectrum,
                    # Final conditions for LPT:
                    RedshiftLPT=RedshiftLPT,
                    WriteLPTDensity=0,
                    WriteLPTDisplacements=0,
                    #########
                    ## RSD ##
                    #########
                    ModuleRSD=1,
                    WriteIntermediaryRSD=0,
                    DoNonLinearMapping=NonLinearRSD,
                    WriteRSDensity=1,
                    OutputRSDensity=fnames_outputdensity[i],
                    #############################
                    ## Cosmological parameters ##
                    #############################
                    h=h,
                    Omega_q=1.0 - Omega_m,
                    Omega_b=Omega_b,
                    Omega_m=Omega_m,
                    Omega_k=0.0,
                    n_s=nS,
                    sigma8=sigma8,
                    w0_fld=-1.0,
                    wa_fld=0.0,
                )
                S.write(fname_simparfile + "_pop{}.sbmy".format(i))


def handle_time_stepping(
    aa: List[float],
    total_steps: int,
    modeldir: str,
    figuresdir: str,
    sim_params: str,
    force: bool = False,
) -> Tuple[Optional[List[int]], Optional[float]]:
    """
    Create and merge individual time-stepping objects.

    Parameters
    ----------
    aa : list of float
        List of scale factors in ascending order.
    total_steps : int
        Total number of time steps to distribute among the provided
        scale factors.
    modeldir : str
        Directory path to store generated time-stepping files.
    figuresdir : str
        Directory path to store time-stepping plots.
    sim_params : str
        Simulation parameter string (e.g., "custom", "std", "nograv").
    force : bool, optional
        Whether to force recompute the time-stepping files. Default is
        False.

    Returns
    -------
    merged_path : str
        Path to the merged time-stepping file.
    indices_steps_cumul : list of int or None
        Cumulative indices for the distributed steps. Returns None if
        using splitLPT or if `sim_params` indicates an alternative
        strategy.
    eff_redshifts : float or None
        Effective redshift derived from the final scale factor in
        'custom' or 'nograv' mode. None otherwise.

    Raises
    ------
    NotImplementedError
        If a unsupported time-stepping strategy is used.
    OSError
        If file or directory operations fail.
    RuntimeError
        If unexpected issues occur during time-stepping setup.
    """
    import numpy as np

    from pysbmy.timestepping import StandardTimeStepping, read_timestepping
    from selfisys.utils.plot_utils import reset_plotting, setup_plotting
    from selfisys.utils.timestepping import merge_nTS

    logger.info("Evaluating time-stepping strategy: %s", sim_params)

    indices_steps_cumul = None
    eff_redshifts = None

    isstd = sim_params.startswith("std")
    splitLPT = sim_params.startswith("splitLPT")

    try:
        # Case 1: standard approach with distributed steps
        if not isstd and not splitLPT:
            reset_plotting()  # Revert to default plotting style
            merged_path = modeldir + "merged.h5"

            # Create time-stepping
            if not os.path.exists(merged_path) or force:
                logger.info("Setting up time-stepping...")

                # Distribute steps among the scale factors
                nsteps = [
                    round((aa[i + 1] - aa[i]) / (aa[-1] - aa[0]) * total_steps)
                    for i in range(len(aa) - 1)
                ]
                # Adjust the largest gap if rounding caused a mismatch
                if sum(nsteps) != total_steps:
                    nsteps[nsteps.index(max(nsteps))] += total_steps - sum(nsteps)

                indices_steps_cumul = list(np.cumsum(nsteps) - 1)
                np.save(modeldir + "indices_steps_cumul.npy", indices_steps_cumul)

                INDENT()
                logger.diagnostic("Generating individual time-stepping objects...")

                TS_paths = []
                for i, (ai, af) in enumerate(zip(aa[:-1], aa[1:])):
                    snapshots = np.full((nsteps[i]), False)
                    snapshots[-1] = True  # Mark last step as a snapshot
                    TS = StandardTimeStepping(ai, af, snapshots, 0)
                    TS_path = modeldir + f"ts{i+1}.h5"
                    TS.write(str(TS_path))
                    TS_paths.append(TS_path)

                # Ensure the timestepping object are readable and plot
                for i, path_ts in enumerate(TS_paths):
                    read_timestepping(str(path_ts)).plot(path=str(figuresdir + f"TS{i}.png"))
                logger.diagnostic("Generating individual time-stepping objects done.")

                logger.diagnostic("Merging time-stepping...")
                merge_nTS([str(p) for p in TS_paths], merged_path)
                TS_merged = read_timestepping(merged_path)
                TS_merged.plot(path=str(figuresdir + "TS_merged.png"))

                # Restore the project's plotting style
                setup_plotting()
                logger.diagnostic("Merging time-stepping done.")
                UNINDENT()
                logger.info("Setting up time-stepping done.")
            else:
                logger.diagnostic("Time-stepping objects already computed.")

            # Evaluate final effective redshift
            if sim_params.startswith("custom") or sim_params.startswith("nograv"):
                eff_redshifts = 1 / aa[-1] - 1
            else:
                raise NotImplementedError("Time-stepping strategy not yet implemented.")

        # Case 2: splitted
        elif splitLPT:
            indices_steps_cumul = [f"pop{i}" for i in range(1, len(aa))]
            eff_redshifts = [1 / a - 1 for a in aa[1:]]

        # Case 3: other
        else:
            logger.diagnostic("Standard time-stepping or no special distribution required.")

    except OSError as e:
        logger.error("File or directory access error in handle_time_stepping: %s", str(e))
        raise
    except Exception as e:
        logger.critical("An error occurred during time-stepping setup: %s", str(e))
        raise RuntimeError("Time-stepping setup failed.") from e
    finally:
        gc.collect()

    return merged_path, indices_steps_cumul, eff_redshifts
