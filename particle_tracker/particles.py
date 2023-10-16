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

        # Work out if there are Fourier directions, and which is first
        num_Fourier_directions = 0
        first_Fourier_direction = 4
        for coord in range(self.dim):
            if(type(self.basis_objects[coord]).__name__=='Fourier'):
                num_Fourier_directions += 1
                if(first_Fourier_direction==4):
                    first_Fourier_direction = coord
        self.num_Fourier_directions = num_Fourier_directions
        self.first_Fourier_direction = first_Fourier_direction

    def interpolate(self,F, locations):

        assert(len(locations)==self.dim)
        # Based on code in the Dedalus users group
        domain = F.domain
        C = F['c'].copy()

        prod_list = []
        for coord in range(self.dim):
            coord_elements = np.squeeze(domain.elements(coord))
            if(type(self.basis_objects[coord]).__name__=='Fourier'):
                left = self.coordBoundaries[coord][0]
                zi = np.array([np.exp(1j*coord_elements*(zs-left)) for zs in locations[coord]])
            elif(type(self.basis_objects[coord]).__name__=='Chebyshev'):
                Lz = self.coordLength[coord]
                left = self.coordBoundaries[coord][0]
                zi = np.array([np.cos(coord_elements*np.arccos(2*(zs-left)/Lz-1)) for zs in locations[coord]])
            elif(type(self.basis_objects[coord]).__name__=='SinCos'):
                left = self.coordBoundaries[coord][0]
                parity = F.meta[coord]['parity']
                if(parity==1):
                    zi = np.array([np.cos(coord_elements*(zs-left)) for zs in locations[coord]])
                else:
                    zi = np.array([np.sin(coord_elements*(zs-left)) for zs in locations[coord]])
            prod_list.append(zi)

        if(self.dim==3):
            if(self.num_Fourier_directions>0):
                if(self.first_Fourier_direction==0 and np.squeeze(domain.elements(0))[0]==0):
                    C[0,:,:] *= 0.5
                if(self.first_Fourier_direction==1 and np.squeeze(domain.elements(1))[0]==0):
                    C[:,0,:] *= 0.5
                if(self.first_Fourier_direction==2 and np.squeeze(domain.elements(2))[0]==0):
                    C[:,:,0] *= 0.5
            D = np.einsum('ijk,lk,lj,li->li',C,prod_list[2],prod_list[1],prod_list[0],optimize=True)
            D = self.row_comm.allreduce(D)
            D = np.einsum('li->l',D,optimize=True)
            D = self.col_comm.allreduce(D)
            if(self.num_Fourier_directions>0):
                I = 2*np.real(D)
            else:
                I = np.real(D)
        elif(self.dim==2):
            if(self.num_Fourier_directions>0):
                if(self.first_Fourier_direction==0 and np.squeeze(domain.elements(0))[0]==0):
                    C[0,:] *= 0.5
                if(self.first_Fourier_direction==1 and np.squeeze(domain.elements(1))[0]==0):
                    C[:,0] *= 0.5
            D = np.einsum('ij,lj,li->l',C,prod_list[1],prod_list[0],optimize=True)
            D = comm.allreduce(D)
            if(self.num_Fourier_directions>0):
                I = 2*np.real(D)
            else:
                I = np.real(D)

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
                self.fluids_vel[:,coord] = self.interpolate(velocities[coord], (self.positions[:,0], self.positions[:,1],self.positions[:,2]))
            elif(self.dim==2):
                self.fluids_vel[:,coord] = self.interpolate(velocities[coord], (self.positions[:,0], self.positions[:,1]))

    def step(self,dt,velocities):
        self.getFluidVel(velocities)

        # Move particles
        self.positions += dt*self.fluids_vel
        # Apply BCs on the particle positions
        for coord in range(self.dim):
            if(type(self.basis_objects[coord]).__name__=='Fourier' or type(self.basis_objects[coord]).__name__=='SinCos'):
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
                self.S[:,coordi,coordj] = self.interpolate(diff_op, self.positions[:,0], self.positions[:,1],self.positions[:,2])

    def stepStress(self,dt,velocities):
        self.getFluidStress(velocities)
        # Move stress
        for particle in range(self.N):
             self.J[particle,:,:] += dt*self.S[particle,:,:]@self.J[particle,:,:]
