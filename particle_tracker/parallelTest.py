"""
Run interpolation tests

Usage:
    parallelTest.py [--mesh=<mesh> --N=<N>]

Options:
    --mesh=<mesh>              Processor mesh for 3-D runs
    --N=<N>                    Number of particles [default: 1024]
"""
import particles
import dedalus.public as de
import numpy as np
from dedalus.core import field
from mpi4py import MPI
import time
from docopt import docopt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

args = docopt(__doc__)
N = 1024
if(rank==0):
    print("All tests running with {} particles".format(N))
mesh = args['--N']
mesh = args['--mesh']
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
    if(rank==0):
        print("3D tests with processor mesh={}".format(mesh))
else:
    log2 = np.log2(size)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    if(rank==0):
        print("3D tests with processor mesh={}".format(mesh))

if(rank==0):
    print()
    print('##############')
    print('## 3D Tests ##')
    print('##############')
    print()

# Test 1
Lx, Ly, Lz = (2*np.pi, 2*np.pi, 2*np.pi)
nx, ny, nz = (64,64,64)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64,mesh=mesh)

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

f = field.Field(domain, name='f')
f['g'] = np.random.rand(*f['g'].shape)

p = particles.particles(N,domain)

# Interpolate f at the particle positions using dedalus
dTime = time.time()
dedalusInterp = []
for pos in p.positions:
    dedalusInterp.append(de.operators.interpolate(f, x=pos[0], y=pos[1],z=pos[2]).evaluate()['g'][0,0,0])
dTime = time.time() - dTime

nTime = time.time()
newInterp = p.interpolate_3D(f,p.positions[:,0],p.positions[:,1],p.positions[:,2])
nTime = time.time() - nTime
if(rank==0):
    print('Fourier-Fourier-Chebyshev mean error: ',np.mean(np.abs(dedalusInterp-newInterp)), ' Speed up: ', dTime/nTime)

# Test 2
Lx, Ly, Lz = (2*np.pi, 2*np.pi, 2*np.pi)
nx, ny, nz = (64,64,64)

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Fourier('z', nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64,mesh=mesh)

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

f = field.Field(domain, name='f')
f['g'] = np.random.rand(*f['g'].shape)

p = particles.particles(N,domain)

# Interpolate f at the particle positions using dedalus
dTime = time.time()
dedalusInterp = []
for pos in p.positions:
    dedalusInterp.append(de.operators.interpolate(f, x=pos[0], y=pos[1],z=pos[2]).evaluate()['g'][0,0,0])
dTime = time.time() - dTime

nTime = time.time()
newInterp = p.interpolate_3D(f,p.positions[:,0],p.positions[:,1],p.positions[:,2])
nTime = time.time() - nTime
if(rank==0):
    print('Fourier-Fourier-Fourier   mean error: ',np.mean(np.abs(dedalusInterp-newInterp)), ' Speed up: ', dTime/nTime)

if(rank==0):
    print()
    print('##############')
    print('## 2D Tests ##')
    print('##############')
    print()

# Test 3
# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Chebyshev('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

x = domain.grid(0)
y = domain.grid(1)

f = field.Field(domain, name='f')
f['g'] = np.random.rand(*f['g'].shape)

p = particles.particles(N,domain)

dTime = time.time()
# Interpolate f at the particle positions using dedalus
dedalusInterp = []
for pos in p.positions:
    dedalusInterp.append(de.operators.interpolate(f, x=pos[0], y=pos[1]).evaluate()['g'][0,0])
dTime = time.time() - dTime

nTime = time.time()
newInterp = p.interpolate_2D(f,p.positions[:,0],p.positions[:,1])
nTime = time.time() - nTime

if(rank==0):
    print('Fourier-Chebyshev mean error: ',np.mean(np.abs(dedalusInterp-newInterp)), ' Speed up: ', dTime/nTime)

# Test 4
# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', ny, interval=(0, Ly), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

x = domain.grid(0)
y = domain.grid(1)

f = field.Field(domain, name='f')
f['g'] = np.random.rand(*f['g'].shape)

p = particles.particles(N,domain)

dTime = time.time()
# Interpolate f at the particle positions using dedalus
dedalusInterp = []
for pos in p.positions:
    dedalusInterp.append(de.operators.interpolate(f, x=pos[0], y=pos[1]).evaluate()['g'][0,0])
dTime = time.time() - dTime

nTime = time.time()
newInterp = p.interpolate_2D(f,p.positions[:,0],p.positions[:,1])
nTime = time.time() - nTime

if(rank==0):
    print('Fourier-Fourier   mean error: ',np.mean(np.abs(dedalusInterp-newInterp)), ' Speed up: ', dTime/nTime)
