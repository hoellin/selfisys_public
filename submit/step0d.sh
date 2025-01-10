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

export OMP_NUM_THREADS=8
python $SELFISYS_ROOT_PATH"src/selfisys/pipeline/step0d.py" \
    --wd $SELFISYS_OUTPUT_PATH"dev/643600502/run1/" \
    --survey_mask_path $SELFISYS_OUTPUT_PATH"expl_notebooks/surveymask/raw_mask_N64.npy" \
    --name_obs obs \
    --reset_window_function True \
    --force_obs True

exit 0
