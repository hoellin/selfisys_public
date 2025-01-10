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

ii_array=-1
# ii_array=( $(seq 0 1 ) )
# ii_array=( $(seq 0 9 ) )

export OMP_NUM_THREADS=8
python $SELFISYS_ROOT_PATH"src/selfisys/pipeline/step0b.py" \
    --pool_path $SELFISYS_OUTPUT_PATH"dev/643600502/run1/data/" \
    --ii ${ii_array[@]} \
    --npar 1 \
    --force False

exit 0