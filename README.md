[![DOI](https://zenodo.org/badge/510360742.svg)](https://zenodo.org/badge/latestdoi/510360742)

# Floquet stability and Lagrangian statistics of a nonlinear time-dependent ABC dynamo (companion code)
This repository contains companion code for the paper 'Floquet stability and Lagrangian statistics of a non-linear time-dependent ABC dynamo' by C. S. Skene and S. M. Tobias (Phys. Rev. Fluids in press 2023).

Code structure
1. FTLE_kinematic - Julia code to calculate the FTLEs for given flow profiles.
2. Floquet - Dedalus v3 code to calculate the Floquet exponents.
3. particle_tracker - Module for Dedalus v2 that allows for quick interpolation of velocities at particle positions and particle tracking.
4. FTLE_non_linear - Dedalus v2 code to solve for time-dependent ABC flows. Also allows for the FTLE exponents to be calculated using the particle tracking module.

# Citing the code
If you find this repository useful for your research please cite the following paper
```
@article{skene_tobias_2023, 
  title={Floquet stability and Lagrangian statistics of a non-linear time-dependent {ABC} dynamo}, 
  journal={Phys. Rev. Fluids}, 
  author={Skene, Calum S. and Tobias, Steven M.}, 
  year={2023},
  notes={in press}
  }
```
## Acknowledgements

This work was undertaken on ARC4, part of the High Performance Computing facilities at the University of Leeds, UK. We acknowledge partial support from a grant from the Simons Foundation (Grant No. 662962, GF). We would also like to acknowledge support of funding from the European Union Horizon 2020 research and innovation programme (grant agreement no. D5S-DLV-786780). Additionally, we would like to thank the Isaac Newton Institute for Mathematical Sciences, Cambridge, for support and hospitality during the programme DYT2 where work on this paper was undertaken. This work was supported by EPSRC grant no EP/R014604/1.
