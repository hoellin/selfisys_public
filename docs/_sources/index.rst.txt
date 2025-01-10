SelfiSys: Assess the Impact of Systematic Effects in Galaxy Surveys
===================================================================

.. image:: https://img.shields.io/badge/astro--ph.CO-arxiv%3A2412.04443-B31B1B.svg
   :target: https://arxiv.org/abs/2412.04443
   :alt: arXiv

.. image:: https://img.shields.io/github/v/tag/hoellin/selfisys_public.svg?label=version
   :target: https://github.com/hoellin/selfisys_public/releases
   :alt: GitHub Release

.. image:: https://img.shields.io/github/last-commit/hoellin/selfisys_public
   :target: https://github.com/hoellin/selfisys_public/commits/main
   :alt: Last Commit

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://github.com/hoellin/selfisys_public/blob/main/LICENSE
   :alt: License

**SelfiSys** is a Python package designed to address the issue of model misspecification in field-based, implicit likelihood cosmological inference.

It leverages the inferred initial matter power spectrum, enabling a thorough diagnosis of systematic effects in large-scale spectroscopic galaxy surveys.

Key Features
------------

- **Custom hidden-box forward models**
   We provide a `HiddenBox` class to simulate realistic spectroscopic galaxy surveys. It accommodates fully non-linear gravitational evolution, and incorporates multiple systematic effects observed in real-world survey, e.g., misspecified galaxy bias, survey mask, selection functions, dust extinction, line interlopers, or inaccurate gravity solver.
- **Diagnosis of systematic effects**
   Diagnose the impact of systematic effects using the inferred initial matter power spectrum, prior to performing cosmological inference.
- **Cosmological inference**
   Perform inference of cosmological parameters using Approximate Bayesian Computation (ABC) with a Population Monte Carlo (PMC) sampler.

For practical examples demonstrating how to use SelfiSys, visit the `SelfiSys Examples Repository <https://github.com/hoellin/selfisys_examples>`_.

References
----------

If you use the SelfiSys package in your research, please cite the following paper and feel free to `contact the authors <mailto:tristan.hoellinger@iap.fr>`_ for feedback, collaboration opportunities, or other inquiries.

**Diagnosing Systematic Effects Using the Inferred Initial Power Spectrum**
*Hoellinger, T. and Leclercq, F., arXiv e-prints*, 2024
`arXiv:2412.04443 <https://arxiv.org/abs/2412.04443>`_  
`[astro-ph.CO] <https://arxiv.org/abs/2412.04443>`_  
`[ADS] <https://ui.adsabs.harvard.edu/abs/arXiv:2412.04443>`_  
`[pdf] <https://arxiv.org/pdf/2412.04443>`_

Contributors
------------

- **Tristan Hoellinger**  
  `tristan.hoellinger@iap.fr <mailto:tristan.hoellinger@iap.fr>`_
  
  Principal developer and maintainer, Institut d’Astrophysique de Paris (IAP).

License
-------

This software is distributed under the GPLv3 Licence. Please review the `LICENSE <https://github.com/hoellin/selfisys_public/blob/main/LICENSE>`_ file in the repository to understand the terms of use and ensure compliance. By downloading and using this software, you agree to the terms of the licence.

Requirements
------------

The code is written in Python 3.10 and depends on the following packages:

- `pySELFI <https://pyselfi.readthedocs.io/en/latest/>`_: Python implementation of the Simulator Expansion for Likelihood-Free Inference.
- `Simbelmynë <https://simbelmyne.readthedocs.io/en/latest/>`_: A hierarchical probabilistic simulator for generating synthetic galaxy survey data.
- `ELFI <https://elfi.readthedocs.io/en/latest/>`_: A statistical software package for likelihood-free inference, implementing Approximate Bayesian Computation (ABC) with a Population Monte Carlo (PMC) sampler.

A comprehensive list of dependencies, along with installation instructions, will be provided in a future release.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   selfisys.hiddenbox
   selfisys.normalise_hb
   selfisys.prior
   selfisys.selection_functions
   selfisys.selfi_interface
   selfisys.sbmy_interface
   selfisys.grf
   selfisys.utils

.. toctree::
   :maxdepth: 2
   :caption: Contribute

   ../../CONTRIBUTING.md

.. toctree::
   :maxdepth: 2
   :caption: References

   ../../REFERENCES.md