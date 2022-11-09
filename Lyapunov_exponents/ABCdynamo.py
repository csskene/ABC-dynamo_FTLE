"""
Run ABC flow dynamo, no particles tracked

Usage:
    ABCdynamo.py [options]

Options:
    --Re=<Re>                         Reynolds number [default: 100]
    --Rm=<Rm>                         Magnetic Reynolds number [default: 100]
    --eps=<eps>                       epsilon [default: 1]
    --Omega=<Omega>                   Omega [default: 2.5]
    --resl=<resl>                     Resolution [default: 96]
    --restart_file=<restart_file>     Location of restart file [default: None]
    --restartN=<restartN>             Restart number [default: -1]
    --T=<T>                           Run time [default: 30]
    --mesh=<mesh>                     Processor mesh for 3-D runs
    --U1                              Whether to zero the magnetic
    --U1dynamo                        Restart from U1 flow with a magnetic field
"""

import numpy as np
import time

from mpi4py import MPI
import copy
from dedalus import public as de
from dedalus.extras import flow_tools
import os
from tools import load_state_partial
import logging
logger = logging.getLogger(__name__)
from docopt import docopt

# parse arguments
args = docopt(__doc__)

# Parameters
Reynolds     = float(args['--Re'])
Rm           = float(args['--Rm'])
eps          = float(args['--eps'])
Ω            = float(args['--Omega'])
resl         = int(args['--resl'])
restartN     = int(args['--restartN'])
restart_file = str(args['--restart_file'])
T            = float(args['--T'])

U1       = args['--U1']
U1dynamo = args['--U1dynamo']

comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size
mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
    logger.info("running on processor mesh={}".format(mesh))
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

pathstr = "Re{0:5.02e}_Rm{1:5.02e}_eps{2:5.02e}_Om{3:5.02e}".format(Reynolds, Rm, eps, Ω)

if(restart_file!='None'):
    pathstr = 'restart' + str(restartN) + pathstr
if(U1):
    pathstr = pathstr + 'U1'
if(U1dynamo):
    pathstr = pathstr + 'U1dynamo'

logger.info('Parameters: Re: %e, eps: %e, Omega: %e' %(Reynolds, eps, Ω))
logger.info('Run time  : T:  %e' %(T))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#Aspect ratio
Lx, Ly, Lz = (2*np.pi, 2*np.pi, 2*np.pi)
nx, ny, nz = (resl, resl, resl)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y',ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z',nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64,mesh=mesh)

if(U1):
    logger.info('Purely hydro simulation')
    problem = de.IVP(domain, variables=['u','v','w','p'])
else:
    logger.info('MHD simulation')
    problem = de.IVP(domain, variables=['u','v','w','Ax','Ay','Az','p','phi'])

problem.parameters['ReInv'] = 1./Reynolds
problem.parameters['RmInv'] = 1./Rm
problem.parameters['eps'] = eps
problem.parameters['Ω'] = Ω

problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A)) + dz(dz(A))"
problem.substitutions['ad(A)'] = "u*dx(A) + v*dy(A) + w*dz(A)"
problem.substitutions['Fx'] = "eps*Ω*cos(Ω*t)*(cos(z + eps*sin(Ω*t)) - sin(y + eps*sin(Ω*t))) + ReInv*(sin(z + eps*sin(Ω*t)) + cos(y + eps*sin(Ω*t)))"
problem.substitutions['Fy'] = "eps*Ω*cos(Ω*t)*(cos(x + eps*sin(Ω*t)) - sin(z + eps*sin(Ω*t))) + ReInv*(sin(x + eps*sin(Ω*t)) + cos(z + eps*sin(Ω*t)))"
problem.substitutions['Fz'] = "eps*Ω*cos(Ω*t)*(cos(y + eps*sin(Ω*t)) - sin(x + eps*sin(Ω*t))) + ReInv*(sin(y + eps*sin(Ω*t)) + cos(x + eps*sin(Ω*t)))"

if(not U1):
    problem.substitutions['Bx'] = " dy(Az) - dz(Ay)"
    problem.substitutions['By'] = "-dx(Az) + dz(Ax)"
    problem.substitutions['Bz'] = " dx(Ay) - dy(Ax)"

    problem.substitutions['Jx'] = "dy(Bz) - dz(By)"
    problem.substitutions['Jy'] = "dz(Bx) - dx(Bz)"
    problem.substitutions['Jz'] = "dx(By) - dy(Bx)"

if(U1):
    problem.substitutions['Lorentz_x'] = "0"
    problem.substitutions['Lorentz_y'] = "0"
    problem.substitutions['Lorentz_z'] = "0"
else:
    problem.substitutions['Lorentz_x'] = "Jy*Bz - Jz*By"
    problem.substitutions['Lorentz_y'] = "Jz*Bx - Jx*Bz"
    problem.substitutions['Lorentz_z'] = "Jx*By - Jy*Bx"

# Hydrodynamic
problem.add_equation("dt(u) - ReInv*Lap(u) + dx(p) =  - ad(u) + Fx + Lorentz_x")
problem.add_equation("dt(v) - ReInv*Lap(v) + dy(p) =  - ad(v) + Fy + Lorentz_y")
problem.add_equation("dt(w) - ReInv*Lap(w) + dz(p) =  - ad(w) + Fz + Lorentz_z")
problem.add_equation("dx(u) + dy(v) + dz(w) = 0",condition="(nx != 0) or (ny != 0) or (nz !=0 )")
problem.add_equation("p = 0",condition="(nx == 0) and (ny == 0) and (nz == 0)")

if(not U1):
    # Magnetic
    problem.add_equation("dt(Ax) - RmInv*Lap(Ax) + dx(phi) = Bz*v - By*w")
    problem.add_equation("dt(Ay) - RmInv*Lap(Ay) + dy(phi) = Bx*w - Bz*u")
    problem.add_equation("dt(Az) - RmInv*Lap(Az) + dz(phi) = By*u - Bx*v")
    problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0",condition="(nx != 0) or (ny != 0) or (nz !=0 )")
    problem.add_equation("phi = 0",condition="(nx == 0) and (ny == 0) and (nz == 0)")

ts = de.timesteppers.RK443
solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

u = solver.state['u']
v = solver.state['v']
w = solver.state['w']

if(not U1):
    Ax = solver.state['Ax']
    Ay = solver.state['Ay']
    Az = solver.state['Az']

if(restart_file != 'None'):
    logger.info('Restarting from: %s' % restart_file)
    try:
        write, last_dt = load_state_partial(solver,restart_file, restartN)
    except:
        logger.info('Run non-linear simmulation to get restart states first')
        raise
    T += solver.sim_time

    if(U1dynamo):
        logger.info('Seeding with a small random magnetic')
        # Seed the magnetic field with a small random field
        Ax.set_scales(1)
        Ay.set_scales(1)
        Az.set_scales(1)

        # Zero to get the U1 flow
        Ax['g'] = 1e-5*np.random.rand(*Ax['g'].shape)
        Ay['g'] = 1e-5*np.random.rand(*Ay['g'].shape)
        Az['g'] = 1e-5*np.random.rand(*Az['g'].shape)

        # Trick to smooth the initial condition (get rid of 50% of modes
        Ax.set_scales(0.5, keep_data=True)
        Ax['c']
        Ax['g']
        Ax.set_scales(domain.dealias, keep_data=True)

        Ay.set_scales(0.5, keep_data=True)
        Ay['c']
        Ay['g']
        Ay.set_scales(domain.dealias, keep_data=True)

        Az.set_scales(0.5, keep_data=True)
        Az['c']
        Az['g']
        Az.set_scales(domain.dealias, keep_data=True)
else:
    logger.info('Initialising with an ABC flow')
    u['g'] = np.sin(z) + np.cos(y)
    v['g'] = np.sin(x) + np.cos(z)
    w['g'] = np.sin(y) + np.cos(x)

    if(not U1):
        logger.info('Initialising with a small random magnetic field')
        Ax['g'] = 1e-5*np.random.rand(*Ax['g'].shape)
        Ay['g'] = 1e-5*np.random.rand(*Ay['g'].shape)
        Az['g'] = 1e-5*np.random.rand(*Az['g'].shape)

        # Trick to smooth the initial condition (get rid of 50% of modes
        Ax.set_scales(0.5, keep_data=True)
        Ax['c']
        Ax['g']
        Ax.set_scales(domain.dealias, keep_data=True)

        Ay.set_scales(0.5, keep_data=True)
        Ay['c']
        Ay['g']
        Ay.set_scales(domain.dealias, keep_data=True)

        Az.set_scales(0.5, keep_data=True)
        Az['c']
        Az['g']
        Az.set_scales(domain.dealias, keep_data=True)

solver.stop_sim_time = T
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

initial_dt = 1e-3
cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v','w'))

# Sort out the paths
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('data'):
        os.mkdir('data')
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('dataSnaps'):
        os.mkdir('dataSnaps')

if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(os.path.join('data/',pathstr)):
        os.mkdir(os.path.join('data/',pathstr))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(os.path.join('dataSnaps/',pathstr)):
        os.mkdir(os.path.join('dataSnaps/',pathstr))

analysis = solver.evaluator.add_file_handler(os.path.join('data/',pathstr,'analysis_dynamo'), sim_dt=0.1)

solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly
solver.evaluator.vars['Lz'] = Lz
analysis.add_task("integ(integ(integ(u**2 + v**2 + w**2,'x'),'y'),'z')/(Lx*Ly*Lz)", name='KE')
if(not U1):
    analysis.add_task("integ(integ(integ(Bx**2 + By**2 + Bz**2,'x'),'y'),'z')/(Lx*Ly*Lz)", name='ME')

snapshots = solver.evaluator.add_file_handler(os.path.join('dataSnaps/',pathstr,'snapshots_dynamo'), max_writes=50,sim_dt=50)
snapshots.add_system(solver.state)

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)

    if (solver.iteration-1) % 10 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)

from dedalus.tools import post
post.merge_process_files(os.path.join('data/',pathstr,'analysis_dynamo'), cleanup=True)
post.merge_process_files(os.path.join('dataSnaps/',pathstr,'snapshots_dynamo'), cleanup=True)
