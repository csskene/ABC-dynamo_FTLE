import numpy as np
import time
import h5py

from mpi4py import MPI
import copy
from dedalus import public as de
from dedalus.extras import flow_tools
import os

import logging
logger = logging.getLogger(__name__)

import pathlib
def load_state_partial(solver, path, index=-1):
        """
        Load state from HDF5 file (only some variables)
        Parameters
        ----------
        path : str or pathlib.Path
            Path to Dedalus HDF5 savefile
        index : int, optional
            Local write index (within file) to load (default: -1)
        Returns
        -------
        write : int
            Global write number of loaded write
        dt : float
            Timestep at loaded write
        """
        path = pathlib.Path(path)
        logger.info("Loading solver state from: {}".format(path))
        with h5py.File(str(path), mode='r') as file:
            # Load solver attributes
            write = file['scales']['write_number'][index]
            try:
                dt = file['scales']['timestep'][index]
            except KeyError:
                dt = None
            solver.iteration = solver.initial_iteration = file['scales']['iteration'][index]
            solver.sim_time = solver.initial_sim_time = file['scales']['sim_time'][index]
            # Log restart info
            logger.info("Loading iteration: {}".format(solver.iteration))
            logger.info("Loading write: {}".format(write))
            logger.info("Loading sim time: {}".format(solver.sim_time))
            logger.info("Loading timestep: {}".format(dt))
            # Load fields
            for field in solver.state.fields:
                try:
                    dset = file['tasks'][field.name]
                    # Find matching layout
                    for layout in solver.domain.dist.layouts:
                        if np.allclose(layout.grid_space, dset.attrs['grid_space']):
                            break
                    else:
                        raise ValueError("No matching layout")
                    # Set scales to match saved data
                    scales = dset.shape[1:] / layout.global_shape(scales=1)
                    scales[~layout.grid_space] = 1
                    # Extract local data from global dset
                    dset_slices = (index,) + layout.slices(tuple(scales))
                    local_dset = dset[dset_slices]
                    # Copy to field
                    field_slices = tuple(slice(n) for n in local_dset.shape)
                    field.set_scales(scales, keep_data=False)
                    field[layout][field_slices] = local_dset
                    field.set_scales(solver.domain.dealias, keep_data=True)
                except:
                    logger.info('Skipping: %s' %(field.name))
        return write, dt
