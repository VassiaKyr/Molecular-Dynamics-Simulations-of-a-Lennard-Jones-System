import os
import numpy as np
import gzip
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann   #boltzman constant
import seaborn as sns
# from numba import jit 

########### atomwrite ###############################################
def MinImage(Pos, L):
    """Returns a new Pos array with minimum-imaged positions."""
    return Pos - L * np.round_(Pos / L)


def PdbStr(Pos, L = None, AtomNames = ["C"], ModelNum = 1):
    """Gets a Pdb string.
Input:
    Pos: (N,3) numpy array of atomic positions
    L: scalar or vector of box lengths (or None to skip min-imaging)
    AtomNames: list of atom names; duplicated as necessary to length N
"""
    N = len(Pos)
    #make a new AtomNames the same length as the position array
    #by repeating it over and over
    AtomNames = AtomNames * (int(N / len(AtomNames)) + 1)
    AtomNames = AtomNames[:N]        
    #minimum image the positions
    if not L is None:
        Pos = MinImage(Pos, L)
    #make the pdb header
    s = "MODEL     %-4i\n" % ModelNum
    #add the coordinates to the pdb string
    for (i, (x,y,z)) in enumerate(Pos):
        an = AtomNames[i].strip()
        s += "HETATM%5i %4s %3s  %4i    %8.3f%8.3f%8.3f                     %2s  \n" % (
             i+1, an.ljust(4)[:4], "SYS", i+1, x, y, z, an.rjust(2)[:2])
    #make the footer
    s += "TER\nENDMDL\n"
    return s


def WritePdb(Filename, Pos, L = None, AtomNames = ["C"], First = False):
    """Writes a position array to a pdb file.
Input:
    Filename: string filename of file to write
    Pos: (N,3) numpy array of atomic positions
    L: scalar or vector of box lengths (or None to skip min-imaging)
    AtomNames: list of atom names; duplicated as necessary to length N
    First: True will overwrite or start a new Pdb file; False will append
"""
    #check to see if the file exists;
    #if so, and we are appending, get the model number
    ModelNum = 1
    if not First and os.path.isfile(Filename):
        for line in file(Filename, "r"):
            if line.startswith("MODEL"):
                ModelNum = int(line[10:].strip())
        ModelNum += 1
    #get the data
    s = PdbStr(Pos, L, AtomNames, ModelNum)
    #write to the file
    if First:
        file(Filename, "w").write(s)
    else:
        file(Filename, "a").write(s)


class pdbfile:
    def __init__(self, FileName, L = None, AtomNames = ["C"],Compressed = False):
        """Creates a new class for writing to a pdb file.
Input:
    Filename: string filename of file to write
    L: scalar or vector of box lengths (or None to skip min-imaging)
    AtomNames: list of atom names; duplicated as necessary to length N
    Compressed: True will write to a gzipped file (default True)
"""
        self.L = L
        self.AtomNames = AtomNames
        self.ModelNum = 1
        if Compressed:
            self.FileName = FileName + ".gz"
            self.fobj = gzip.GzipFile(self.FileName, "w")
        else:
            self.FileName = FileName
            self.fobj = open(self.FileName, "w")

    def write(self, Pos):
        """Writes positions to a Pdb file.
Input:
    Pos: (N,3) numpy array of atomic positions
"""
        s = PdbStr(Pos, self.L, self.AtomNames, self.ModelNum)
        self.fobj.write(s)
        self.ModelNum += 1

    def close(self):
        """Closes Pdb file object."""
        self.fobj.close()
        
###############################################################################################
        
def InitPositions(N, L):
    """Creates an array of initial position for each atom, that is placed on a cubic lattice.
Input:
    N: number of atoms
    L: box length
Output:
    Pos: (N,3) array of positions
"""
    # array of positions
    Pos = np.zeros((N,3), float)
    
    # compute integer grid # of locations for cubic lattice 
    NLat = int(N**(1./3.) + 1.)
    
    # make an array of lattice sites
    r = L * (np.arange(NLat, dtype=float)/NLat - 0.5)
    
    # loop through x,y,z positions in lattice until done 
    # for every atom in the system
    i = 0
    for x in r:
        for y in r:
            for z in r:
                Pos[i] = np.array([x,y,z], float)
                i += 1
                # if done placing atoms, return
                if i >= N:
                    return Pos
                
    return Pos


def RescaleVelocities(Vel, T):
    """Rescales velocities in the system to the target temperature.
Input:
    Vel: (N,3) array of atomic velocities
    T: target temperature
Output:
    Vel: same as above
"""
    #recenter to zero net momentum (assuming all masses same)
    Vel = Vel - Vel.mean(axis=0)
    #find the total kinetic energy
    KE = 0.5 * np.sum(Vel * Vel)
    #find velocity scale factor from ratios of kinetic energy
    VScale = np.sqrt(1.5 * len(Vel) * T / KE)
    Vel = Vel * VScale
    return Vel  


def InitVelocities(N, T, m):
    """Returns an initial random velocity set.
Input:
    N: number of atoms
    T: target temperature
Output:
    Vel: (N,3) array of atomic velocities
"""
    stdev = np.sqrt(1.0*T / m)
    Vel = np.random.normal(loc=0.0, scale=stdev, size=(N, 3))
    Vel = RescaleVelocities(Vel, T)
    return Vel


# def InitAccel(Pos, N, L, r):
#     """Returns the initial acceleration array.
# Input:
#     Pos: (N,3) array of atomic positions
#     L: simulation box length
# Output:
#     Accel: (N,3) array of acceleration vectors
# """
#     Accel = np.zeros_like(Pos)
#     #get the acceleration from the forces
#     PEnergy = u_LJ(Pos, sigma=1.0, epsilon=1.0)
#     Accel = calc_forces(N, L, r, epsilon=1.0, sigma=1.0, rc=2.5)
#     return Accel


def min_image_conv(L, r1, r2):
    """Impose minimum-image convention."""
    dr = np.copy(r2-r1)
    for k in range(3):
        if dr[k] > 0.5*L:
            dr[k] -= L
        elif dr[k] < -0.5*L:
            dr[k] += L
    return dr


def apply_PBC(N, L, r):
    """Apply periodic boundary conditions.
Input:
    N: number of atoms
    L: box length
    r: (N,3) array of positions
Output:
    r: (N,3) array of positions
"""
    for i in range(N):
        for j in range(3):
            R = r[i][j]
            if R > L:
                r[i][j] -= L
            elif R < 0:
                r[i][j] += L
    return r


def u_LJ(r, sigma=1.0, epsilon=1.0):
    """Calculation of Lenard-Jones potential energy.
Input:
    r: (N,3) array of positions
    sigma: standard deviation
    epsilon: system constant
Output:
    ulj: potential energy
"""
    d6 = (sigma/r)**6
    ulj = 4.0*epsilon*(d6**2 - d6)
    return ulj


def Hamiltonian(N, L, r, v, m=1.0, sigma=1.0, epsilon=1.0, rc=2.5):
    """Calculation of Hamiltonian (total energy).
Input:
    N: number of atoms
    L: box length
    r: (N,3) array of positions
    v: (N,3) array of velocities
    m: particle mass
    sigma: standard deviation
    epsilon: system constant
    rc: cut-off radius
Output:
    H, KE, U: hamiltonian, kinetic and potential energy
"""
    KE = 0.5*np.sum(v**2)
    U = 0.0
    dr = np.zeros(3)
    for i in range(0,N-1):
        for j in range(i+1,N):
            # enforce minimum image convention
            dr = min_image_conv(L, r[i], r[j])
            d = np.linalg.norm(dr, ord=2)
            if d <= rc:
                U += u_LJ(d, sigma, epsilon)
                
    H = KE + U
    return H, KE, U


def calc_forces(N, L, r, sigma=1.0, epsilon=1.0, rc=2.5):
    """Calculation of forces.
Input:
    N: number of atoms
    L: box length
    r: (N,3) array of positions
    sigma: standard deviation
    epsilon: system constant
    rc: cut-off radius
Output:
    Forces: (N,3) array of forces
"""
    N = r.shape[0]
    Forces = np.zeros((N,3))
    dr = np.zeros(3)
    
    for i in range(N-1):
        for j in range (i+1,N):
            # minimum image convention reinforcement
            dr = min_image_conv(L,r[i],r[j])
            d = np.linalg.norm(dr, ord=2)
            
            # calculate forces, enforce cut-off radius
            if d<=rc:
                d6 = (sigma/d)**6
                # calculate force vectors length
                F = (24.0*epsilon/d)*(-2.0*(d6**2) + d6)
                fv = F*dr/d
                Forces[i] += fv
                Forces[j] -= fv
    return Forces


def particle_distances(N, r):
    """Calculation of distances between all particles.
Input:
    N: number of atoms
    r: (N,3) array of positions
Output:
    Distances: (0.5*(N-1)*N,1) vector of euclidean distances
"""
    Distances = np.zeros(int(0.5*(N-1)*N))
    k = 0
    for i in range(N-1):
        for j in range(i+1,N):
            Distances[k] = np.linalg.norm(r[i]-r[j], ord=2)
            k += 1
            
    return Distances


# plot pdf of distances
def plot_dists(distances):
    plt.figure(figsize=(10,7))
    sns.histplot(distances, bins=100, edgecolor='k', kde=True)
    plt.title('Distribution of particle distances', fontsize=15, pad=15)
    plt.xlabel('Distances', fontsize=15, labelpad=15)
    plt.ylabel('Counts', fontsize=15, labelpad=15)
    plt.savefig('pdf_dists.png', bbox_inches='tight')
    
# plot hamiltonian
def plot_ham(Ham):
    plt.plot(Ham)
    plt.title('Hamiltonian', fontsize=15, pad=15)
    plt.xlabel('Time steps', fontsize=15, labelpad=15)
    plt.ylabel('Energy', fontsize=15, labelpad=15)
    plt.ylim(-300,-100)
    plt.savefig('hamiltonian.png', bbox_inches='tight')

# plot total momentum
def plot_tmomentum(TotalP):
    plt.plot(TotalP)
    plt.title('Total Momentum', fontsize=15, pad=15)
    plt.xlabel('Time steps', fontsize=15, labelpad=15)
    plt.ylim(-1,1)
    plt.savefig('tmomentum.png', bbox_inches='tight')
#     plt.ylabel('Y-axis ')



###########################################################################


# run test 
def run_test(N=100, rho=0.5, T=0.5, dt=1.e-4,\
                nsteps=10000, ksteps=100, psteps=10000,\
                rc=2.5, epsilon=1.0, sigma=1.0, m=1.0,\
                PlotHam=True, PlotTMomentum=True, SaveTrajectory=True,\
                struct='Sc'):
    
    # arrays for saving values every ksteps
    Ham = np.zeros(int(nsteps/ksteps))
    KE = np.zeros(int(nsteps/ksteps))
    U = np.zeros(int(nsteps/ksteps))
    TotalP = np.zeros((int(nsteps/ksteps),3))
    
    k = ksteps
    dt2 = (dt**2)
    t = 0
    
    #calculate box volume and length
    V = float(N)/rho
    L = V**(1/3)
    print('volume = ',V,' length = ',L)
    
    #set the random number seed; useful for debugging
    np.random.seed = 342324
    
    #initial positions, velocities
    r = InitPositions(N, L)
    v = InitVelocities(N, T, m)
    print(r.shape, v.shape)
            
    # write pdb file with positions
    if SaveTrajectory:
        Pdb = pdbfile("md.pdb", L)
        #write initial positions
        Pdb.write(r)
        
    rm1 = np.copy(r)
    rm2 = np.copy(r)
    
    a = calc_forces(N, L, r, rc=rc)
    
    for i in range(N):
        rm1[i] = rm2[i] + dt*v[i] + 0.5*dt2*a[i]
        
    a = calc_forces(N, L, rm1, rc=rc)
    
    rm1 = apply_PBC(N, L, rm1)

    Ham[0], KE[0], U[0] = Hamiltonian(N, L, rm1, v, rc=rc)
    print('Time step =',0, 'Hamiltonian = ',Ham[0])
    
    # write hamiltonian in a file
    hamiltonian_file = open("Hamiltonian.txt", "w")
    hamiltonian_file.write(str(Ham[0]) + '\n')
    
    # write total momentum in a file
    momentum_file = open("TotalMomentum.txt", "w")
    initmomentum = np.sum(v*m, axis=0)
    momentum_file.write(str(initmomentum) + '\n')
    
#     counts = np.zeros(pair_dist_bins)
    
    # integration
    time_counts = 0
    
    # for averaging the energies every ksteps
    ham = 0
    kinen = 0
    poten = 0
        
    for t in range(1, nsteps):
        for i in range(N):
            #translate atoms
            r[i] = -rm2[i] + 2.0*rm1[i] + dt2*a[i]
            # update velocities 
            v[i] = (r[i]-rm2[i])/(2.0*dt)
        
        #rescale velocities
        v = RescaleVelocities(v, T)
        
        #copy before BC cause we use them in differences 
        rm2 = np.copy(rm1)
        rm1 = np.copy(r)
            
        # calculate forces and particle acceleration
        a = calc_forces(N, L, rm1, rc=rc)
        
        #apply BC
        rm1 = apply_PBC(N, L, rm1)
        
        
#         #check if we need to output the positions 
#         if SaveTrajectory:
#             Pdb.write(rm1)
            
        # calculate energies
        ham1, kinen1, poten1 = Hamiltonian(N, L, rm1, v, rc=rc)
        ham += ham1
        kinen += kinen1
        poten += poten1
                
        #calculate particular values every ksteps
        if t==k:
            # find average values every ksteps
            j = int(k/ksteps)
            Ham[j] = ham/ksteps
            KE[j] = kinen/ksteps
            U[j] = poten/ksteps
            
            # calculation of total momentum P every k steps
            TotalP[j] = np.sum(v*m)
            
#             print(f'avgham={Ham[j]}, p={TotalP[j]}')
            
            # write hamiltonian in a file
            hamiltonian_file = open("Hamiltonian.txt", "a")
            hamiltonian_file.write(str(Ham[j]) + '\n')
            
            # write total momentum in a file
            momentum_file = open("TotalMomentum.txt", "a")
            momentum_file.write(str(TotalP[j]) + '\n')
            
            #check if we need to write the positions every ksteps
            if SaveTrajectory:
                Pdb.write(rm1)
            
            k += 100
            ham = 0
            kinen = 0
            poten = 0
    
    # calculate distances between all particles at the last time step and plot pdf
    distances = particle_distances(N, rm1)
    plot_dists(distances)
    
    # close hamiltonian file
    hamiltonian_file.close()
    
    # close momentum file
    momentum_file.close()
    
    #check if we need to close pdb file
    if SaveTrajectory:
        Pdb.close()
    
    # plot hamiltonian and total momentum
    if PlotHam and PlotTMomentum:
        plt.figure(figsize=(15,3))
        plt.subplot(1, 2, 1) # row 1, col 2 index 1
        plot_ham(Ham)
        
        plt.subplot(1, 2, 2) # row 1, col 2 index 2
        plot_tmomentum(TotalP)
        
        
if __name__=='__main__':
    run_test()