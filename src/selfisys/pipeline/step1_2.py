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
Steps 1 and 2 of the SelfiSys pipeline.

Run all the Simbelmynë simulations needed to linearise the HiddenBox,
using the .sbmy files generated in step 0. It can run sequentially or in
parallel, depending on the --npar argument.

Raises
------
OSError
    If file or directory paths are inaccessible.
RuntimeError
    If unexpected HPC or PySbmy issues occur.
"""

import os
import gc
import numpy as np

from selfisys.utils.parser import ArgumentParser, none_or_bool_or_str, bool_sh
from selfisys.utils.logger import getCustomLogger


logger = getCustomLogger(__name__)


parser = ArgumentParser(
    description="Run the Simbelmynë simulations required to linearise the HiddenBox."
)
parser.add_argument("--pool_path", type=str, help="Path to the pool of simulations.")
parser.add_argument("--directions", type=int, nargs="+", help="List of directions.")
parser.add_argument("--pp", type=int, nargs="+", help="List of simulation indices p.")
parser.add_argument("--Npop", type=int, help="Number of populations.", default=None)
parser.add_argument("--npar", type=int, help="Number of sim to run in parallel.", default=8)
parser.add_argument(
    "--sim_params",
    type=none_or_bool_or_str,
    default=None,
    help="Parameters for the simulations, e.g., 'splitLPT', 'custom19COLA20' etc.",
)
parser.add_argument("--force", type=bool_sh, help="Force computations.", default=False)

args = parser.parse_args()
pool_path = args.pool_path
sim_params = args.sim_params
splitLPT = sim_params[:8] == "splitLPT" if sim_params is not None else False
force = args.force
directions = np.array(args.directions, dtype=int)
pp = np.array(args.pp, dtype=int)
Npop = args.Npop
npar = args.npar


def run_sim(val):
    """
    Execute a single Simbelmynë simulation.

    Parameters
    ----------
    val : tuple
        A tuple (d, p, ipop) containing:
        d : int
            Direction index.
        p : int
            Simulation index.
        ipop : str or None
            Population identifier for splitLPT (e.g. 'pop0', 'pop1',
            etc), None for other approaches.

    Raises
    ------
    OSError
        If the .sbmy file or output path is invalid.
    RuntimeError
        If the simulation fails unexpectedly.
    """
    from pysbmy import pySbmy

    d, p, ipop = val
    dirpath = f"{pool_path}d{d}/"
    if ipop is not None:
        fname_simparfile = f"{dirpath}sim_d{d}_p{p}_{ipop}.sbmy"
    else:
        fname_simparfile = f"{dirpath}sim_d{d}_p{p}.sbmy"
    fname_output = f"{dirpath}output_density_d{d}_p{p}.h5"
    fname_simlogs = f"{dirpath}logs_sim_d{d}_p{p}.txt"

    if os.path.isfile(fname_output) and not force:
        logger.info("Output file %s already exists, skipping...", fname_output)
        gc.collect()
    else:
        from io import BytesIO
        from selfisys.utils.low_level import stdout_redirector, stderr_redirector

        logger.debug("Running Simbelmynë for d=%d, p=%d, ipop=%s", d, p, ipop)
        # sys.stdout.flush()
        f = BytesIO()
        g = BytesIO()
        with stdout_redirector(f):
            with stderr_redirector(g):
                pySbmy(fname_simparfile, fname_simlogs)
            g.close()
        f.close()
        # sys.stdout.flush()
        gc.collect()
        logger.debug("Simbelmynë run completed for d=%d, p=%d, ipop=%s", d, p, ipop)


if len(pp) == 1 and pp[0] == -1:
    # If simulation indices are not specified, find them in the
    # pool_path directory
    if splitLPT:
        raise ValueError("pp = -1 not supported with splitLPT.")

    pp = np.array(
        [
            int(f.split("_")[2].split(".")[0][1:])
            for f in os.listdir(f"{pool_path}d{directions[0]}")
            if f.startswith("sim_d") and f.endswith(".sbmy")
        ],
        dtype=int,
    )


if __name__ == "__main__":
    import tqdm.auto as tqdm
    from itertools import product

    try:
        if splitLPT:
            if Npop is None:
                raise ValueError("Npop must be specified when using splitLPT mode.")
            pops = [f"pop{i}" for i in range(Npop)]
            vals = list(product(directions, pp, pops))
        else:
            vals = list(product(directions, pp, [Npop]))

        nsim = len(vals)
        logger.info("Found %d simulation tasks to run.", nsim)

        if npar > 1:
            from multiprocessing import Pool

            logger.info("Running simulations using %d processes in parallel.", npar)
            with Pool(processes=npar) as mp_pool:
                for _ in tqdm.tqdm(mp_pool.imap(run_sim, vals), total=nsim):
                    pass
            logger.info("Running simulations done.")
        else:
            logger.info("Running simulations sequentially...")
            for _ in tqdm.tqdm(map(run_sim, vals), total=nsim):
                pass
            logger.info("Running simulations done.")

    except OSError as e:
        logger.error("File or directory access error: %s", str(e))
        raise
    except Exception as e:
        logger.critical("Unexpected error in step X: %s", str(e))
        raise RuntimeError("Simulations failed.") from e
    finally:
        gc.collect()
        logger.info("All simulations completed successfully.")
