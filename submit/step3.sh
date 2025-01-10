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

# Notes:
# - One shall use "recompute_mocks True" when running step3 for the
#   first time within a given run.
# - Afterwards, one can, again, use "recompute_mocks True" for
#   recomputing the mocks with a misspecified model.

export OMP_NUM_THREADS=1
python $SELFISYS_ROOT_PATH"src/selfisys/pipeline/step3.py" \
    --wd $SELFISYS_OUTPUT_PATH"dev/643600502/run1/" \
    --N_THREADS 8 \
    --N_THREADS_PRIOR 8 \
    --prior "planck2018" \
    --nsamples_prior 1000 \
    --survey_mask_path $SELFISYS_OUTPUT_PATH"expl_notebooks/surveymask/raw_mask_N64.npy" \
    --params_obs None \
    --name_obs obs \
    --force_obs False \
    --recompute_obs_mock False \
    --reset_window_function True \
    --selection_params 0.1150 0.1492 0.1818 0.1500 0.4925 0.8182 1 1 1 \
    --Ne 150 \
    --Ns 3 \
    --prefix_mocks None \
    --force_recompute_prior False \
    --lin_bias 1.47 1.99 2.32 \
    --recompute_mocks True \
    --perform_score_compression True \
    --force_score_compression False

    # --lin_bias 1.47 1.99 2.32 \
    # --recompute_mocks True \

    # --noise_dbg 0.1 \

exit 0
