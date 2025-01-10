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

"""Tools for time-stepping.
"""


def merge_nTS(ts_path_list, merged_path):
    """
    Merge multiple time-stepping objects into a single file.

    Parameters
    ----------
    ts_path_list : list of str
        Paths to the individual time-stepping files to be merged.
    merged_path : str
        Path to save the merged time-stepping file.

    Returns
    -------
    None
    """
    from h5py import File
    from numpy import concatenate
    from pysbmy.timestepping import read_timestepping

    # Read individual time-stepping objects
    ts = [read_timestepping(ts_path) for ts_path in ts_path_list]

    with File(merged_path, "w") as hf:
        # Write scalar attributes
        hf.attrs["/info/scalars/nsteps"] = sum(tsi.nsteps for tsi in ts)
        hf.attrs["/info/scalars/nkicks"] = sum(tsi.nkicks for tsi in ts)
        hf.attrs["/info/scalars/ndrifts"] = sum(tsi.ndrifts for tsi in ts)
        hf.attrs["/info/scalars/ai"] = ts[0].ai
        hf.attrs["/info/scalars/af"] = ts[-1].af

        # Merge and write datasets
        hf.create_dataset("/scalars/forces", data=concatenate([tsi.forces for tsi in ts]))
        hf.create_dataset("/scalars/snapshots", data=concatenate([tsi.snapshots for tsi in ts]))
        hf.create_dataset("/scalars/aKickBeg", data=concatenate([tsi.aKickBeg for tsi in ts]))
        hf.create_dataset("/scalars/aKickEnd", data=concatenate([tsi.aKickEnd for tsi in ts]))
        hf.create_dataset("/scalars/aDriftBeg", data=concatenate([tsi.aDriftBeg for tsi in ts]))
        hf.create_dataset("/scalars/aDriftEnd", data=concatenate([tsi.aDriftEnd for tsi in ts]))
        hf.create_dataset("/scalars/aiKick", data=concatenate([tsi.aiKick for tsi in ts]))
        hf.create_dataset("/scalars/afKick", data=concatenate([tsi.afKick for tsi in ts]))

        # Handle `aDrift` merging with overlap adjustments
        aDrift_data = concatenate(
            [
                [ts[0].aDrift[0]],  # Initial drift
                concatenate(
                    [concatenate([tsi.aDrift[1:], [tsi.aDrift[-1]]]) for tsi in ts[:-1]]
                ),  # Intermediate drifts
                ts[-1].aDrift[1:],  # Final drift
            ]
        )
        hf.create_dataset("/scalars/aDrift", data=aDrift_data)

        # Handle `aSnapshotSave` merging
        aSnapshotSave_data = concatenate(
            [ts[0].aSnapshotSave] + [tsi.aSnapshotSave[1:] for tsi in ts[1:]]
        )
        hf.create_dataset("/scalars/aSnapshotSave", data=aSnapshotSave_data)

        hf.create_dataset("/scalars/aiDrift", data=concatenate([tsi.aiDrift for tsi in ts]))
        hf.create_dataset("/scalars/afDrift", data=concatenate([tsi.afDrift for tsi in ts]))
        hf.create_dataset("/scalars/aKick", data=concatenate([tsi.aKick for tsi in ts]))
