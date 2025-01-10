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
python $SELFISYS_ROOT_PATH"src/selfisys/pipeline/step0a.py" \
    --wd_ext dev/ \
    --name run1 \
    --size 64 \
    --Np0 64 \
    --Npm0 64 \
    --L 3600 \
    --S 32 \
    --Pinit 50 \
    --Nnorm 2 \
    --total_steps 10 \
    --sim_params "custom19COLA20" \
    --aa 0.05 0.1 0.4 1 \
    --Ne 150 \
    --Ns 3 \
    --OUTDIR $SELFISYS_OUTPUT_PATH \
    --radial_selection "multiple_lognormal" \
    --selection_params 0.1150 0.1492 0.1818 0.1500 0.4925 0.8182 1 1 1 \
    --survey_mask_path $SELFISYS_OUTPUT_PATH"expl_notebooks/surveymask/raw_mask_N64.npy" \
    --lin_bias 1.47 1.99 2.32 \
    --prior "planck2018" \
    --nsamples_prior 10000 \
    --noise 0.1 \
    --force True

exit 0
