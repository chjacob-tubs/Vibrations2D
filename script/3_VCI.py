#!/usr/bin/env python3
import VibTools as LocVib
import Vibrations as vib
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot
import pickle
import os

# Functions below are used to localize modes in subsets

def localize_subset(modes,subset):
    # method that takes normal modes
    # and a range of modes, returns them
    # localized + the cmat
    tmpmodes = modes.get_subset(subset)
    tmploc = LocVib.LocVib(tmpmodes, 'PM')
    tmploc.localize()
    tmploc.sort_by_residue()
    tmploc.adjust_signs()
    tmpcmat = tmploc.get_couplingmat(hessian=True) # if hessian=False, returns [cm-1] otherwise returns [hartree]

    return tmploc.locmodes.modes_mw, tmploc.locmodes.freqs, tmpcmat

def localize_subsets(modes,subsets):
    # method that takes normal modes and list of lists (beginin and end)
    # of subsets and make one set of modes localized in subsets

    # first get number of modes in total
    total = 0
    modes_mw = np.zeros((0, 3*modes.natoms))
    freqs = np.zeros((0,))

    for subset in subsets:
        n = len(subset)
        total += n


    print('Modes localized: %i, modes in total: %i' %(total, modes.nmodes))

    if total > modes.nmodes:
        raise Exception('Number of modes in the subsets is larger than the total number of modes')
    else:
        cmat = np.zeros((total, total))
        actpos = 0 #actual position in the cmat matrix
        for subset in subsets:
            tmp = localize_subset(modes, subset)
            modes_mw = np.concatenate((modes_mw, tmp[0]), axis = 0)
            freqs = np.concatenate((freqs, tmp[1]), axis = 0)
            cmat[actpos:actpos + tmp[2].shape[0],actpos:actpos + tmp[2].shape[0]] = tmp[2]
            actpos = actpos + tmp[2].shape[0] 
        localmodes = LocVib.VibModes(total, modes.mol)
        localmodes.set_modes_mw(modes_mw)
        localmodes.set_freqs(freqs)

        return localmodes, cmat


# The vibrations script begins here

# Read in normal modes from SNF results
# using LocVib (LocVib package)
path = os.getcwd()

outname = os.path.join(path,'snf.out') 
restartname = os.path.join(path,'restart') 
coordfile = os.path.join(path,'coord') 

res = LocVib.SNFResults(outname=outname,restartname=restartname,
                     coordfile=coordfile)
res.read()



# Now localize modes in separate subsets

subsets = np.load(os.path.join(path,'subsets.npy'))
localmodes,cmat = localize_subsets(res.modes,subsets)

# Use normal modes:
#normmodes = res.modes


# Define the grid

ngrid = 16
amp = 14
grid = vib.Grid(res.mol,localmodes)
#grid = vib.Grid(res.mol,normmodes)
grid.generate_grids(ngrid,amp)

# Read in anharmonic 1-mode potentials

v1 = vib.Potential(grid, order=1)
v1.read_np(os.path.join(path,'V1_g16.npy'))

# Read in anharmonic 1-mode dipole moments

dm1 = vib.Dipole(grid)
dm1.read_np(os.path.join(path,'Dm1_g16.npy'))

# Read in anharmonic 2-mode potentials

v2h = vib.Potential(grid,order=2) 
v2h.generate_harmonic(cmat=cmat)

dVSCF = vib.VSCF2D(v1,v2h)
dVSCF.solve()

# Now run VCI calculations using the VSCF wavefunction

VCI = vib.VCI(dVSCF.get_groundstate_wfn(), v1,v2h)
VCI.generate_states(1) # singles only # for further calculations with test case water use triples (3) ######## 
VCI.solve()
VCI.print_results(which=8, maxfreq=8000)

VCI.calculate_IR(dm1)
ints = VCI.intensities[:]
freqs = VCI.energiesrcm[:] - VCI.energiesrcm[0]

for i in range(len(ints)):
    print(freqs[i],ints[i])

np.savetxt('VCI_freq.txt',freqs)
np.savetxt('VCI_ints.txt',ints)