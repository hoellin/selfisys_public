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
Setup script for the SelfiSys package.

SelfiSys enables thorough diagnosis of systematic effects in
field-based, implicit likelihood inference (ILI) of cosmological
parameters from large-scale spectroscopic galaxy surveys.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "../README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="selfisys",
    version="0.1.0",
    author="Tristan Hoellinger",
    author_email="tristan.hoellinger@iap.fr",
    description="Diagnosing systematic effects in implicit likelihood cosmological inferences.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/hoellin/selfisys_public",
    package_data={"selfisys": ["preamble.tex"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.7",
    license="GPLv3",
    keywords="cosmology systematic-effects large-scale-structure systematics implicit-likelihood-inference misspecification robust-inference galaxy-surveys",
)
