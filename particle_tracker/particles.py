import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class particles:
    def __init__(self,N,domain):
        # N - number of particles
        # domain - domain object from the simulation
        self.N               = N
        self.dim             = domain.dim
        self.basis_objects   = [domain.get_basis_object(i) for i in range(self.dim)]
        self.coordBoundaries = [basis.interval for basis in self.basis_objects]
        self.coordLength     = [x[1] - x[0] for x in self.coordBoundaries]
        # Initialise
        self.initialisePositions()
        self.fluids_vel = np.zeros((self.N,self.dim))

        self.J = np.zeros((self.N,self.dim,self.dim))
        self.initialiseStress()
        self.S = np.zeros((self.N,self.dim,self.dim))

        meshShape = domain.distributor.mesh.shape[0]
        # Be fancy and make new comms
        if(size>1):
            if(meshShape==2):
                self.row_comm = domain.dist.comm_cart.Sub([0,1])
                self.col_comm = domain.dist.comm_cart.Sub([1,0])
            elif(meshShape==1):
                self.row_comm = domain.dist.comm_cart.Sub([0])
                self.col_comm = domain.dist.comm_cart.Sub([1])
        else:
            self.row_comm = domain.dist.comm_cart.Sub([])
            self.col_comm = domain.dist.comm_cart.Sub([])

    def interpolate_3D(self,F, xp, yp, zp):
        # Based on code in the Dedalus users group
        domain = F.domain

        C = F['c'].copy()

        xc = np.squeeze(domain.elements(0))
        yc = np.squeeze(domain.elements(1))
        zc = np.squeeze(domain.elements(2))

        xi = np.array([np.exp(1j*xc*xs) for xs in xp])
        yi = np.array([np.exp(1j*yc*ys) for ys in yp])
        if(type(self.basis_objects[-1]).__name__=='Fourier'):
            zi = np.array([np.exp(1j*zc*zs) for zs in zp])
        elif(type(self.basis_objects[-1]).__name__=='Chebyshev'):
            Lz = self.coordLength[2]
            left = self.coordBoundaries[2][0]
            zi = np.array([np.cos(zc*np.arccos(2*(zs-left)/Lz-1)) for zs in zp])

        if(xc[0]==0):
            C[0,:,:] *= 0.5
        D = np.einsum('ijk,lk,lj,li->li',C,zi,yi,xi,optimize=True)
        D = self.row_comm.allreduce(D)
        D = np.einsum('li->l',D,optimize=True)
        D = self.col_comm.allreduce(D)
        I = 2*np.real(D)

        return I

    def interpolate_2D(self,F, xp, yp):
        # Based on code in the Dedalus users group
        domain = F.domain

        C = F['c'].copy()

        xc = np.squeeze(domain.elements(0))
        yc = np.squeeze(domain.elements(1))

        xi = np.array([np.exp(1j*xc*xs) for xs in xp])
        if(type(self.basis_objects[-1]).__name__=='Fourier'):
            yi = np.array([np.exp(1j*yc*ys) for ys in yp])
        elif(type(self.basis_objects[-1]).__name__=='Chebyshev'):
            Lz = self.coordLength[-1]
            left = self.coordBoundaries[-1][0]
            yi = np.array([np.cos(yc*np.arccos(2*(ys-left)/Lz-1)) for ys in yp])

        if(xc[0]==0):
            C[0,:] *= 0.5
        D = np.einsum('ij,lj,li->li',C,yi,xi,optimize=True)
        D = np.einsum('li->l',D,optimize=True)
        D = comm.allreduce(D)
        I = 2*np.real(D)

        return I

    def initialisePositions(self):
        # Initialise using random distributed globally
        if(rank==0):
            rVec = np.random.random((self.dim,self.N))
        else:
            rVec = np.zeros((self.N,))
        rVec = comm.bcast(rVec,root=0)

        self.positions = np.array([self.coordBoundaries[i][0] + self.coordLength[i]*rVec[i] for i in range(self.dim)]).T


    def getFluidVel(self,velocities):

        assert(len(velocities)==self.dim)
        for coord in range(self.dim):
            if(self.dim==3):
                self.fluids_vel[:,coord] = self.interpolate_3D(velocities[coord], self.positions[:,0], self.positions[:,1],self.positions[:,2])
            elif(self.dim==2):
                self.fluids_vel[:,coord] = self.interpolate_2D(velocities[coord], self.positions[:,0], self.positions[:,1])

    def step(self,dt,velocities):
        self.getFluidVel(velocities)

        # Move particles
        self.positions += dt*self.fluids_vel
        # Apply BCs on the particle positions
        for coord in range(self.dim):
            if(type(self.basis_objects[coord]).__name__=='Fourier'):
                # Periodic boundary conditions
                self.positions[:,coord] = self.coordBoundaries[coord][0]+np.mod(self.positions[:,coord]-self.coordBoundaries[coord][0],self.coordLength[coord])
            if(type(self.basis_objects[coord]).__name__=='Chebyshev'):
                # Non-periodic boundary conditions
                self.positions[:,coord] = np.clip(self.positions[:,coord], self.coordBoundaries[coord][0], self.coordBoundaries[coord][1])

    def initialiseStress(self):

        for particle in range(self.N):
            self.J[particle,0,0] = 1./np.sqrt(2)
            self.J[particle,1,1] = 1./np.sqrt(2)

    def getFluidStress(self,velocities):
        # Check
        assert(len(velocities)==self.dim)

        for coordi in range(self.dim):
            for coordj in range(self.dim):
                diff_op = self.basis_objects[coordj].Differentiate(velocities[coordi])
                self.S[:,coordi,coordj] = self.interpolate_3D(diff_op, self.positions[:,0], self.positions[:,1],self.positions[:,2])

    def stepStress(self,dt,velocities):
        self.getFluidStress(velocities)
        # Move stress
        for particle in range(self.N):
             self.J[particle,:,:] += dt*self.S[particle,:,:]@self.J[particle,:,:]
