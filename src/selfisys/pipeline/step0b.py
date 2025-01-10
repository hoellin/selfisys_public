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
Step 0b of the SelfiSys pipeline.

Run all the Simbelmynë simulations required to normalise the HiddenBox.

This script invokes the pySbmy interface to launch Simbelmynë
simulations in parallel.
"""

import os
from pathlib import Path
import gc
import numpy as np

from selfisys.utils.parser import ArgumentParser, bool_sh
from selfisys.utils.logger import getCustomLogger

logger = getCustomLogger(__name__)

parser = ArgumentParser(description="Run Simbelmynë for step 0b of the SelfiSys pipeline.")
parser.add_argument("--pool_path", type=str, help="Path to the pool of simulations.")
parser.add_argument("--ii", type=int, nargs="+", help="Indices of simulations to run.")
parser.add_argument(
    "--npar", type=int, help="Number of simulations to run in parallel.", default=4
)
parser.add_argument("--force", type=bool_sh, help="Force the computations.", default=False)
args = parser.parse_args()

pool_path = args.pool_path
npar = args.npar
force = args.force
ii = np.array(args.ii, dtype=int)

# If 'ii' is [-1], find simulation indices from the pool directory
if len(ii) == 1 and ii[0] == -1:
    try:
        ii = np.array(
            [
                int(f.split("norm__")[1].split("_")[0])
                for f in os.listdir(pool_path)
                if f.startswith("sim_norm") and f.endswith(".sbmy")
            ],
            dtype=int,
        )
    except OSError as e:
        logger.error("Failed to list files in '%s': %s", pool_path, str(e))
        raise

nsim = len(ii)


def worker_norm(i):
    """
    Run a Simbelmynë simulation to normalise the HiddenBox.

    Parameters
    ----------
    i : int
        Index specifying which simulation file to run.

    Raises
    ------
    OSError
        If file or directory access fails.
    RuntimeError
        If pySbmy encounters an unexpected error or the simulation fails.
    """
    from pysbmy import pySbmy
    from selfisys.utils.low_level import stdout_redirector, stderr_redirector
    from io import BytesIO

    file_prefix = "sim_norm__" + str(i)
    try:
        # Find the simulation file corresponding to this index
        suffix = [str(f) for f in os.listdir(pool_path) if f.startswith(file_prefix)][0]
        fname_simparfile = Path(pool_path) / suffix

        # Derive output and logs filenames
        base_out = suffix.split(".")[0].split("sim_")[1]
        fname_output = Path(pool_path) / f"output_density_{base_out}.h5"
        fname_simlogs = Path(pool_path) / f"{file_prefix}.txt"

        # Skip if the output file already exists
        if fname_output.exists() and not force:
            logger.info(
                "Output file %s already exists, skipping simulation index %d...", fname_output, i
            )
            return

        # Suppress or capture stdout/stderr for live monitoring purposes
        f = BytesIO()
        g = BytesIO()
        with stdout_redirector(f):
            with stderr_redirector(g):
                pySbmy(str(fname_simparfile), str(fname_simlogs))
            g.close()
        f.close()

    except OSError as e:
        logger.error("File or directory access error while running index %d: %s", i, str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error in worker_norm (index %d): %s", i, str(e))
        raise RuntimeError(f"Simulation index {i} failed.") from e


if __name__ == "__main__":
    from tqdm import tqdm
    from multiprocessing import Pool

    try:
        logger.info("Running the simulations to normalise the HiddenBox...")
        with Pool(processes=npar) as pool:
            for _ in tqdm(pool.imap(worker_norm, ii), total=nsim):
                pass
        logger.info("Running the simulations to normalise the HiddenBox done.")

    except OSError as e:
        logger.error("Pool or directory error: %s", str(e))
        raise
    except Exception as e:
        logger.critical("An unexpected error occurred during step 0b: %s", str(e))
        raise RuntimeError("Step 0b failed.") from e
    finally:
        gc.collect()
        logger.info("step 0b of the SelfiSys pipeline: done.")
