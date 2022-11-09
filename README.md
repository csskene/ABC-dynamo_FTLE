# ABC-dynamo
Companion code for the paper 'TBC'

Code structure
1. FTLE - Julia code to calculate the FTLEs for given flow profiles.
2. Floquet - Dedalus v3 code to calculate the Floquet exponents and mu
3. particles - Module for Dedalus v2 that allows for quick interpolation of velocities at particle positions and particle tracking.
4.  Lyapunov exponents - Dedalus v2 code to solve for time-dependent ABC flows. Also allows for the FTLE exponents to be calculated using the particle tracking module.

## Citation
If you find this code useful please cite the paper

...

and the code

...

## Acknowledgements

This work was undertaken on ARC4, part of the High Performance Computing facilities at the University of Leeds, UK. We acknowledge partial support from a grant from the Simons Foundation (Grant No. 662962, GF). We would also like to acknowledge support of funding from the European Union Horizon 2020 research and innovation programme (grant agreement no. D5S-DLV-786780).
