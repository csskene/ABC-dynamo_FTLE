"""
Run ABC flow dynamo and track passive field

Usage:
    ABCdynamoPassive.py [options]

Options:
    --Re=<Re>                         Reynolds number [default: 100]
    --Rm=<Rm>                         Magnetic Reynolds number [default: 100]
    --eps=<eps>                       epsilon [default: 1]
    --Omega=<Omega>                   Omega [default: 2.5]
    --T=<T>                           Run time [default: 30]
    --resl=<resl>                     Resolution [default: 96]
    --mesh=<mesh>                     Processor mesh for 3-D runs
    --restart_file=<restart_file>     Location of restart file
    --restartN=<restartN>             Restart number for paths [default: -1]
"""

import numpy as np
import time

from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import os
from tools import load_state_partial
import sys

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
T            = float(args['--T'])
resl         = int(args['--resl'])
restart_file = str(args['--restart_file'])
restartN     = int(args['--restartN'])


pathstr = "_Re{0:5.02e}_Rm{1:5.02e}_eps{2:5.02e}_Om{3:5.02e}_passive".format(Reynolds, Rm, eps, Ω)
pathstr = 'restart' + str(restartN) + pathstr

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

problem = de.IVP(domain, variables=['u','v','w','Ax','Ay','Az','p','phi','Alx','Aly','Alz','phi_l'])

problem.parameters['ReInv'] = 1./Reynolds
problem.parameters['RmInv'] = 1./Rm
problem.parameters['eps'] = eps
problem.parameters['Ω'] = Ω

problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A)) + dz(dz(A))"
problem.substitutions['ad(A)'] = "u*dx(A) + v*dy(A) + w*dz(A)"
problem.substitutions['Fx'] = "eps*Ω*cos(Ω*t)*(cos(z + eps*sin(Ω*t)) - sin(y + eps*sin(Ω*t))) + ReInv*(sin(z + eps*sin(Ω*t)) + cos(y + eps*sin(Ω*t)))"
problem.substitutions['Fy'] = "eps*Ω*cos(Ω*t)*(cos(x + eps*sin(Ω*t)) - sin(z + eps*sin(Ω*t))) + ReInv*(sin(x + eps*sin(Ω*t)) + cos(z + eps*sin(Ω*t)))"
problem.substitutions['Fz'] = "eps*Ω*cos(Ω*t)*(cos(y + eps*sin(Ω*t)) - sin(x + eps*sin(Ω*t))) + ReInv*(sin(y + eps*sin(Ω*t)) + cos(x + eps*sin(Ω*t)))"


problem.substitutions['Bx'] = " dy(Az) - dz(Ay)"
problem.substitutions['By'] = "-dx(Az) + dz(Ax)"
problem.substitutions['Bz'] = " dx(Ay) - dy(Ax)"

problem.substitutions['Jx'] = "dy(Bz) - dz(By)"
problem.substitutions['Jy'] = "dz(Bx) - dx(Bz)"
problem.substitutions['Jz'] = "dx(By) - dy(Bx)"

problem.substitutions['Lorentz_x'] = "Jy*Bz - Jz*By"
problem.substitutions['Lorentz_y'] = "Jz*Bx - Jx*Bz"
problem.substitutions['Lorentz_z'] = "Jx*By - Jy*Bx"

problem.substitutions['Blx'] = " dy(Alz) - dz(Aly)"
problem.substitutions['Bly'] = "-dx(Alz) + dz(Alx)"
problem.substitutions['Blz'] = " dx(Aly) - dy(Alx)"

# Hydrodynamic
problem.add_equation("dt(u) - ReInv*Lap(u) + dx(p) =  - ad(u) + Fx + Lorentz_x")
problem.add_equation("dt(v) - ReInv*Lap(v) + dy(p) =  - ad(v) + Fy + Lorentz_y")
problem.add_equation("dt(w) - ReInv*Lap(w) + dz(p) =  - ad(w) + Fz + Lorentz_z")
problem.add_equation("dx(u) + dy(v) + dz(w) = 0",condition="(nx != 0) or (ny != 0) or (nz !=0 )")
problem.add_equation("p = 0",condition="(nx == 0) and (ny == 0) and (nz == 0)")

# Magnetic
problem.add_equation("dt(Ax) - RmInv*Lap(Ax) + dx(phi) = Bz*v - By*w")
problem.add_equation("dt(Ay) - RmInv*Lap(Ay) + dy(phi) = Bx*w - Bz*u")
problem.add_equation("dt(Az) - RmInv*Lap(Az) + dz(phi) = By*u - Bx*v")
problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0",condition="(nx != 0) or (ny != 0) or (nz !=0 )")
problem.add_equation("phi = 0",condition="(nx == 0) and (ny == 0) and (nz == 0)")

# Passive
problem.add_equation("dt(Alx) - RmInv*Lap(Alx) + dx(phi_l) = Blz*v - Bly*w")
problem.add_equation("dt(Aly) - RmInv*Lap(Aly) + dy(phi_l) = Blx*w - Blz*u")
problem.add_equation("dt(Alz) - RmInv*Lap(Alz) + dz(phi_l) = Bly*u - Blx*v")
problem.add_equation("dx(Alx) + dy(Aly) + dz(Alz) = 0",condition="(nx != 0) or (ny != 0) or (nz !=0 )")
problem.add_equation("phi_l = 0",condition="(nx == 0) and (ny == 0) and (nz == 0)")

ts = de.timesteppers.RK443
solver =  problem.build_solver(ts)

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

u = solver.state['u']
v = solver.state['v']
w = solver.state['w']

try:
    write, last_dt = load_state_partial(solver,restart_file, restartN)
except:
    logger.info('Wrong file name. Make sure you run a non-linear simmulation to get restart states first')
    raise
T += solver.sim_time

solver.stop_sim_time = T
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf
solver.stop_iteration = np.inf
solver.stop_iteration = np.inf
initial_dt = last_dt

logger.info('Seeding passive field with a small random magnetic')


Alx = solver.state['Alx']
Aly = solver.state['Aly']
Alz = solver.state['Alz']
# Seed the magnetic field with a small random field
Alx.set_scales(1)
Aly.set_scales(1)
Alz.set_scales(1)

Alx['g'] = 1e-5*np.random.rand(*Alx['g'].shape)
Aly['g'] = 1e-5*np.random.rand(*Aly['g'].shape)
Alz['g'] = 1e-5*np.random.rand(*Alz['g'].shape)

# Trick to smooth the initial condition (get rid of 50% of modes
Alx.set_scales(0.5, keep_data=True)
Alx['c']
Alx['g']
Alx.set_scales(domain.dealias, keep_data=True)

Aly.set_scales(0.5, keep_data=True)
Aly['c']
Aly['g']
Aly.set_scales(domain.dealias, keep_data=True)

Alz.set_scales(0.5, keep_data=True)
Alz['c']
Alz['g']
Alz.set_scales(domain.dealias, keep_data=True)

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

analysis = solver.evaluator.add_file_handler(os.path.join('data/',pathstr,'analysis_passive'), sim_dt=0.1)
solver.evaluator.vars['Lx'] = Lx
solver.evaluator.vars['Ly'] = Ly
solver.evaluator.vars['Lz'] = Lz
analysis.add_task("integ(integ(integ(u**2 + v**2 + w**2,'x'),'y'),'z')/(Lx*Ly*Lz)", name='KE')
analysis.add_task("integ(integ(integ(Bx**2 + By**2 + Bz**2,'x'),'y'),'z')/(Lx*Ly*Lz)", name='ME')
analysis.add_task("integ(integ(integ(Blx**2 + Bly**2 + Blz**2,'x'),'y'),'z')/(Lx*Ly*Lz)", name='ME_passive')

snapshots = solver.evaluator.add_file_handler(os.path.join('dataSnaps/',pathstr,'snapshots_passive'), max_writes=50,sim_dt=10)
snapshots.add_task("Blx")
snapshots.add_task("Bly")
snapshots.add_task("Blz")

cfl = flow_tools.CFL(solver,initial_dt,safety=0.8)
cfl.add_velocities(('u','v','w'))

logger.info('Starting loop')
start_time = time.time()
while solver.ok:
    dt = cfl.compute_dt()
    solver.step(dt)

    if (solver.iteration-1) % 1000 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

end_time = time.time()

# Print statistics
logger.info('Run time: %f' %(end_time-start_time))
logger.info('Iterations: %i' %solver.iteration)
