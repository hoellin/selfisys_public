# SelfiSys: Assess the Impact of Systematic Effects in Galaxy Surveys

[![arXiv](https://img.shields.io/badge/astro--ph.CO-arxiv%3A2412.04443-B31B1B.svg?style=flat)](https://arxiv.org/abs/2412.04443)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/hoellin/selfisys_public/blob/main/LICENSE)
[![GitHub version](https://img.shields.io/github/tag/hoellin/selfisys_public.svg?label=version)](https://github.com/hoellin/selfisys_public)
[![GitHub last commit](https://img.shields.io/github/last-commit/hoellin/selfisys_public.svg)](https://github.com/hoellin/selfisys_public/commits/main)


**SelfiSys** is a Python package designed to address the issue of model misspecification in field-based, implicit likelihood cosmological inference.

It leverages the inferred initial matter power spectrum, enabling a thorough diagnosis of systematic effects in large-scale spectroscopic galaxy surveys.

## Key Features

- **Custom hidden-box forward models**

We provide a `HiddenBox` class to simulate realistic spectroscopic galaxy surveys. It accommodates fully non-linear gravitational evolution, and incorporates multiple systematic effects observed in real-world survey, e.g., misspecified galaxy bias, survey mask, selection functions, dust extinction, line interlopers, or inaccurate gravity solver.

- **Diagnosis of systematic effects**

Diagnose the impact of systematic effects using the inferred initial matter power spectrum, prior to performing cosmological inference.

- **Cosmological inference**

Perform inference of cosmological parameters using Approximate Bayesian Computation (ABC) with a Population Monte Carlo (PMC) sampler.

---

## Documentation

The documentation, including a detailed API reference, is available at [hoellin.github.io/selfisys_public](https://hoellin.github.io/selfisys_public/).

For practical examples demonstrating how to use SelfiSys, visit the [SelfiSys Examples Repository](https://github.com/hoellin/selfisys_examples).

## Contributors

- **Tristan Hoellinger**, [tristan.hoellinger@iap.fr](mailto:tristan.hoellinger@iap.fr)  
  Principal developer and maintainer, Institut d’Astrophysique de Paris (IAP).
  
For information on contributing, refer to [CONTRIBUTING.md](CONTRIBUTING.md).

## References

If you use the SelfiSys package in your research, please cite the following paper and feel free to [contact the authors](mailto:tristan.hoellinger@iap.fr) for feedback, collaboration opportunities, or other inquiries.

**Diagnosing Systematic Effects Using the Inferred Initial Power Spectrum**
*Hoellinger, T. and Leclercq, F., 2024*
[arXiv:2412.04443](https://arxiv.org/abs/2412.04443) [[astro-ph.CO]](https://arxiv.org/abs/2412.04443) [[ADS]](https://ui.adsabs.harvard.edu/abs/arXiv:2412.04443) [[pdf]](https://arxiv.org/pdf/2412.04443)

BibTeX entry for citation:
```bibtex
@ARTICLE{hoellinger2024diagnosing,
      author = {Hoellinger, Tristan and Leclercq, Florent},
      title = "{Diagnosing Systematic Effects Using the Inferred Initial Power Spectrum}",
      journal = {arXiv e-prints},
   keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
      month = dec,
         eid = {arXiv:2412.04443},
      pages = {arXiv:2412.04443},
         doi = {10.48550/arXiv.2412.04443},
archivePrefix = {arXiv},
      eprint = {2412.04443},
primaryClass = {astro-ph.CO},
      adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241204443H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Requirements

The code is written in Python 3.10 and depends on the following packages:
- [`pySELFI`](https://pyselfi.readthedocs.io/en/latest/): Python implementation of the Simulator Expansion for Likelihood-Free Inference.
- [`Simbelmynë`](https://simbelmyne.readthedocs.io/en/latest/): A hierarchical probabilistic simulator for generating synthetic galaxy survey data.
- [`ELFI`](https://elfi.readthedocs.io/en/latest/): A statistical software package for likelihood-free inference, implementing in particular Approximate Bayesian Computation (ABC) with a Population Monte Carlo (PMC) sampler.

A comprehensive list of dependencies, including version specifications to ensure reproducibility, will be provided in a yaml file, along with installation instructions, in a future release.

---

## License

This software is distributed under the GPLv3 Licence. Please review the [LICENSE](https://github.com/hoellin/selfisys_public/blob/main/LICENSE) file in the repository to understand the terms of use and ensure compliance. By downloading and using this software, you agree to the terms of the licence.