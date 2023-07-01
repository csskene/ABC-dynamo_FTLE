"""
Calculate the magnetic Floquet multipliers for ABC flow

Usage:
    Floquet_magnetic_ABC.py [options]

Options:
    --Rm=<Rm>                  Magnetic Reynolds number [default: 100]
    --eps=<eps>                epsilon [default: 1]
    --Omega=<Omega>            Omega [default: 2.5]
    --resl=<resl>              Resolution [default: 64]
    --mesh=<mesh>              Processor mesh for 3-D runs
    --nev=<nev>                Number of eigenvectors [default: 10]
    --tol=<tol>                Tolerance [default: 1e-4]
    --timestep=<timestep>      Tolerance [default: 1e-2]
"""

import numpy as np
import time

from dedalus import public as d3
from mpi4py import MPI
import logging
from docopt import docopt
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

π = np.pi

# parse arguments
args = docopt(__doc__)

# Parameters
Rm       = float(args['--Rm'])
epsilon  = float(args['--eps'])
Ω        = float(args['--Omega'])
tol      = float(args['--tol'])
resl     = int(args['--resl'])
nev      = int(args['--nev'])
timestep = float(args['--timestep'])

fileName = "eigsMagneticRm{0:5.02e}_eps{1:5.02e}_Om{2:5.02e}_dt{3:5.02e}_resl{4:d}".format(Rm, epsilon, Ω,timestep,resl)

if(rank==0):
    print('Requested timestep =',timestep)
T = 2*np.pi/Ω
N = int(T/timestep)
timestep = T/N
if(rank==0):
    print('Adjusted timestep =',timestep)

Lx, Ly, Lz = (2*π,2*π,2*π)
nx, ny, nz = (resl,resl,resl)

log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

# Create bases and domain
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, mesh=mesh,dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
ybasis = d3.RealFourier(coords['y'], size=ny, bounds=(0, Ly), dealias=3/2)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, Lz), dealias=3/2)

phi = dist.Field(name='phi', bases=(xbasis,ybasis,zbasis))
A = dist.VectorField(coords, name='A', bases=(xbasis,ybasis,zbasis))
u0 = dist.VectorField(coords, name='u0', bases=(xbasis,ybasis,zbasis))

x, y, z = dist.local_grids(xbasis,ybasis,zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)

# ABC flow
t = dist.Field()
tau_phi = dist.Field()

sinzxy = dist.VectorField(coords, bases=(xbasis,ybasis,zbasis))
sinzxy['g'][0]  = np.sin(z)
sinzxy['g'][1]  = np.sin(x)
sinzxy['g'][2]  = np.sin(y)

coszxy = dist.VectorField(coords, bases=(xbasis,ybasis,zbasis))
coszxy['g'][0]  = np.cos(z)
coszxy['g'][1]  = np.cos(x)
coszxy['g'][2]  = np.cos(y)

cosyzx = dist.VectorField(coords, bases=(xbasis,ybasis,zbasis))
cosyzx ['g'][0]  = np.cos(y)
cosyzx ['g'][1]  = np.cos(z)
cosyzx ['g'][2]  = np.cos(x)

sinyzx = dist.VectorField(coords, bases=(xbasis,ybasis,zbasis))
sinyzx ['g'][0]  = np.sin(y)
sinyzx ['g'][1]  = np.sin(z)
sinyzx ['g'][2]  = np.sin(x)

u0 = sinzxy*np.cos(epsilon*np.sin(Ω*t)) + coszxy*np.sin(epsilon*np.sin(Ω*t)) +\
     cosyzx*np.cos(epsilon*np.sin(Ω*t)) - sinyzx*np.sin(epsilon*np.sin(Ω*t))

problem = d3.IVP([phi, A,tau_phi], time=t, namespace=locals())
problem.add_equation("dt(A)  + grad(phi) - 1/Rm*lap(A) = cross(u0,curl(A))")
problem.add_equation("div(A) + tau_phi = 0")

# Gauge condition
problem.add_equation("integ(phi) = 0")

dom = A.domain
local_slice = dom.dist.grid_layout.slices(dom,scales=1)
gshape = dom.dist.grid_layout.global_shape(dom,scales=1)

third = np.prod(gshape)
vecSize = np.prod(gshape)*3
vec = np.ones(vecSize)
def vecToField(solver,vec):
    solver.state[1]['g']
    solver.state[1].change_scales(1)
    solver.state[1]['g'][0] = vec[:third].reshape(gshape)[local_slice]
    solver.state[1]['g'][1] = vec[third:2*third].reshape(gshape)[local_slice]
    solver.state[1]['g'][2] = vec[2*third:3*third].reshape(gshape)[local_slice]

def fieldToVec(solver):
    vecu = np.zeros(gshape)
    vecv = np.zeros(gshape)
    vecw = np.zeros(gshape)
    solver.state[1]['g']
    solver.state[1].change_scales(1)
    vecu[local_slice] = solver.state[1]['g'][0]
    vecv[local_slice] = solver.state[1]['g'][1]
    vecw[local_slice] = solver.state[1]['g'][2]
    vecu = comm.allreduce(vecu,op=MPI.SUM).reshape(third)
    vecv = comm.allreduce(vecv,op=MPI.SUM).reshape(third)
    vecw = comm.allreduce(vecw,op=MPI.SUM).reshape(third)
    vec = np.hstack((vecu,vecv,vecw))
    return vec

def monodromyMult(q0):
    problem.time['g'] = 0 # Reset time
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = 2*π/Ω

    vecToField(solver,q0)
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % 100 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    qT = fieldToVec(solver)
    return qT

import scipy.sparse as sp
from scipy import linalg
Psi = sp.linalg.LinearOperator((vecSize,vecSize),matvec=monodromyMult)

mu, v = sp.linalg.eigs(Psi,k=nev,tol=tol)
comm.barrier()
if(rank==0):
    print(mu)
    np.savez(fileName,eigs=mu,modes=v)
    np.savez(fileName+'_eigsOnly',eigs=mu)
