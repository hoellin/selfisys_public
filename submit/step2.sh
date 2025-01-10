#!/bin/bash
# ----------------------------------------------------------------------
# Copyright (C) 2024 Tristan Hoellinger
# Distributed under the GNU General Public License v3.0 (GPLv3).
# See the LICENSE file in the root directory for details.
# SPDX-License-Identifier: GPL-3.0-or-later
# ----------------------------------------------------------------------

# Author: Tristan Hoellinger
# Version: 0.1
# Date: 2024
# License: GPLv3

eval "$(conda shell.bash hook)"
conda activate simbel

# Run the Simbelmynë simulations in the directions dir_array of the
# parameter space, (all sims if pp_array=-1, selected sims otherwise).

dir_array=( $(seq 1 32 ) )
# dir_array=( $(seq 62 64 ) )
# dir_array=( $(seq 1 64 ) )

pp_array=-1
# pp_array=( $(seq 0 1 ) )
# pp_array=( $(seq 0 9 ) )

export OMP_NUM_THREADS=1
python $SELFISYS_ROOT_PATH"src/selfisys/pipeline/step1_2.py" \
    --pool_path $SELFISYS_OUTPUT_PATH"dev/643600502/run1/pool/" \
    --directions ${dir_array[@]} \
    --pp ${pp_array[@]} \
    --Npop 3 \
    --npar 8 \
    --force True

exit 0