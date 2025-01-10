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
SelfiSys Package

A Python package for diagnosing systematic effects in field-based,
implicit likelihood inference (ILI) of cosmological parameters from
large-scale spectroscopic galaxy surveys. The diagnostic utilises the
initial matter power spectrum inferred with pySELFI.

Key functionalities:
- Setup custom models of realistic spectroscopic galaxy surveys,
- Diagnosis of systematic effects model using the initial matter power
spectrum inferred with pySELFI (https://pyselfi.readthedocs.io/),
- Perform inference of cosmological parameters using Approximate
Bayesian Computation (ABC) with a Population Monte Carlo (PMC) sampler
using ELFI (https://elfi.readthedocs.io/).
"""

from .global_parameters import *
