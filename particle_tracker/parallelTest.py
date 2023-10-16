"""
Run interpolation tests

Usage:
    parallelTest.py [--mesh=<mesh> --N=<N>]

Options:
    --mesh=<mesh>              Processor mesh for 3-D runs
    --N=<N>                    Number of particles [default: 128]
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

N    = int(args['--N'])
mesh = args['--mesh']

if(rank==0):
    print("All tests running with {} particles".format(N))
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

Lx, Ly, Lz = (2*np.pi, 2*np.pi, 2*np.pi)
offx, offy, offz = (-1,2,-3)
nx, ny, nz = (64,64,64)

# Create list of all tests

x_basis = ['Fourier','Sin','Cos']
y_basis = ['Fourier','Sin','Cos']
z_basis = ['Fourier','Sin','Cos','Chebyshev']

all_tests = params = [(x_b,y_b,z_b) for x_b in x_basis for y_b in y_basis for z_b in z_basis]

for test in all_tests:
    # Create bases and domain
    if(test[0]=='Fourier'):
        x_basis = de.Fourier('x', nx, interval=(offx, offx+Lx), dealias=3/2)
    elif(test[0]=='Sin' or test[0]=='Cos'):
        x_basis = de.SinCos('x', nx, interval=(offx, offx+Lx), dealias=3/2)
    if(test[1]=='Fourier'):
        y_basis = de.Fourier('y', ny, interval=(offy, offy+Ly), dealias=3/2)
    elif(test[1]=='Sin' or test[1]=='Cos'):
        y_basis = de.SinCos('y', ny, interval=(offy, offy+Ly), dealias=3/2)
    if(test[2]=='Fourier'):
        z_basis = de.Fourier('z', nz, interval=(offz, offz+Lz), dealias=3/2)
    elif(test[2]=='Sin' or test[2]=='Cos'):
        z_basis = de.SinCos('z', nz, interval=(offz, offz+Lz), dealias=3/2)
    elif(test[2]=='Chebyshev'):
        z_basis = de.Chebyshev('z', nz, interval=(offz, offz+Lz), dealias=3/2)

    domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64,mesh=mesh)
    #
    x = domain.grid(0)
    y = domain.grid(1)
    z = domain.grid(2)
    #
    f = field.Field(domain, name='f')
    if(test[0]=='Sin'):
        f.meta['x']['parity'] = -1
    elif(test[0]=='Cos'):
        f.meta['x']['parity'] = 1
    if(test[1]=='Sin'):
        f.meta['y']['parity'] = -1
    elif(test[1]=='Cos'):
        f.meta['y']['parity'] = 1
    if(test[2]=='Sin'):
        f.meta['z']['parity'] = -1
    elif(test[2]=='Cos'):
        f.meta['z']['parity'] = 1

    f['g'] = np.random.rand(*f['g'].shape)

    p = particles.particles(N,domain)

    # Interpolate f at the particle positions using dedalus
    dTime = time.time()
    dedalusInterp = []
    for pos in p.positions:
        dedalusInterp.append(de.operators.interpolate(f, x=pos[0], y=pos[1],z=pos[2]).evaluate()['g'][0,0,0])
    dTime = time.time() - dTime

    nTime = time.time()
    newInterp = p.interpolate(f,(p.positions[:,0],p.positions[:,1],p.positions[:,2]))
    nTime = time.time() - nTime
    testStr = test[0]+'-'+test[1]+'-'+test[2]+':'
    if(rank==0):
        print('{0:30s} mean error: {1:7.4g} Speed up: {2:5.4g}'.format(testStr,np.mean(np.abs(dedalusInterp-newInterp)),dTime/nTime))

if(rank==0):
    print()
    print('##############')
    print('## 2D Tests ##')
    print('##############')
    print()

Lx, Ly     = (2*np.pi, 2*np.pi)
offx, offy = (-1,2)
nx, ny     = (64,64)

# Create list of all tests

x_basis = ['Fourier','Sin','Cos']
y_basis = ['Fourier','Sin','Cos','Chebyshev']

all_tests = params = [(x_b,y_b) for x_b in x_basis for y_b in y_basis]

for test in all_tests:
    # Create bases and domain
    if(test[0]=='Fourier'):
        x_basis = de.Fourier('x', nx, interval=(offx, offx+Lx), dealias=3/2)
    elif(test[0]=='Sin' or test[0]=='Cos'):
        x_basis = de.SinCos('x', nx, interval=(offx, offx+Lx), dealias=3/2)
    if(test[1]=='Fourier'):
        y_basis = de.Fourier('y', ny, interval=(offy, offy+Ly), dealias=3/2)
    elif(test[1]=='Sin' or test[1]=='Cos'):
        y_basis = de.SinCos('y', ny, interval=(offy, offy+Ly), dealias=3/2)
    elif(test[1]=='Chebyshev'):
        y_basis = de.Chebyshev('y', ny, interval=(offy, offy+Ly), dealias=3/2)

    domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)
    #
    x = domain.grid(0)
    y = domain.grid(1)
    #
    f = field.Field(domain, name='f')
    if(test[0]=='Sin'):
        f.meta['x']['parity'] = -1
    elif(test[0]=='Cos'):
        f.meta['x']['parity'] = 1
    if(test[1]=='Sin'):
        f.meta['y']['parity'] = -1
    elif(test[1]=='Cos'):
        f.meta['y']['parity'] = 1

    f['g'] = np.random.rand(*f['g'].shape)

    p = particles.particles(N,domain)

    # Interpolate f at the particle positions using dedalus
    dTime = time.time()
    dedalusInterp = []
    for pos in p.positions:
        dedalusInterp.append(de.operators.interpolate(f, x=pos[0], y=pos[1]).evaluate()['g'][0,0])
    dTime = time.time() - dTime

    nTime = time.time()
    newInterp = p.interpolate(f,(p.positions[:,0],p.positions[:,1]))
    nTime = time.time() - nTime
    testStr = test[0]+'-'+test[1]+':'
    if(rank==0):
        print('{0:30s} mean error: {1:7.4g} Speed up: {2:5.4g}'.format(testStr,np.mean(np.abs(dedalusInterp-newInterp)),dTime/nTime))