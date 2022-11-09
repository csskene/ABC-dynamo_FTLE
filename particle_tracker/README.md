## Particle tracker for Dedalus v2

A module that adds on to Dedalus for enabling particle tracking. Currently works in 2D and 3D for all combinations of Fourier, Sin, Cos, and Chebyshev bases.

The particle tracking code is contained in the *particles.py* python module. It is mainly based on a replacement of the Dedalus implementation of interpolation, to allow for more efficient interpolation at many positions simultaneously.

Together with the particle tracking module, there are tests and examples.

1. *parallelTest.py* tests the interpolation operator in *particles.py* against that contained in Dedalus v2. A particular mesh for the 3D tests can be specified with the argument *--mesh=n,m*. The number of particles for the test can be changed with the argument *--N=num_particles*. As well as the interpolation errors, the script also outputs the speedup gained over the Dedalus interpolation.

2. *RayleighBenard2D.py* tracks particle positions for the Dedalus 2D Rayleigh-Benard example.

3. *plot2D.ipynb* plots the particle positions from the 2D Rayleigh-Benard example.
