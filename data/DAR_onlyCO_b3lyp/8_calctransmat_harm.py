import VibTools as LocVib
import Vibrations as vib
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot
import pickle
import os

print vib.Misc.fancy_box('Example 3:')
print 'Localization of modes of water, harmonic L-VCI-S'
print 'provides initial normal modes\' frequencies and intensities'
print 
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
    tmpcmat = tmploc.get_couplingmat(hessian=True)

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


    print 'Modes localized: %i, modes in total: %i' %(total, modes.nmodes)

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

#path = os.path.join(os.getcwd(),'potentials')
path = '/home/julia/testsystems/DAR/def2svp_b3lyp_V1V2_CO'

res = LocVib.SNFResults(outname=os.path.join(path,'snf.out'),
                        restartname=os.path.join(path,'restart'),
                        coordfile=os.path.join(path,'coord'))
res.read()

# Now localize modes in separate subsets

subsets = np.load(os.path.join(path,'subsets.npy'))
#subsets = [[23,24,25]]
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

#v1 = vib.Potential(grid, order=1)
#v1.read_np(os.path.join(path,'V1_g16.npy'))
v1h = vib.Potential(grid,order=1) # i don't need harmonic calculations
v1h.generate_harmonic(cmat=cmat)

# Read in anharmonic 1-mode dipole moments

dm1 = vib.Dipole(grid)
dm1.read_np(os.path.join(path,'Dm1_g16.npy'))
dm1h = vib.Dipole(grid)
dm1h.generate_harmonic(res)

# Read in anharmonic 2-mode potentials

#v2 = vib.Potential(grid, order=2)
#v2.read_np(os.path.join(path,'V2_g16.npy'))
v2h = vib.Potential(grid,order=2) # i don't need harmonic calculations
v2h.generate_harmonic(cmat=cmat)

# Read in anharmonic 2-mode dipole moments

#dm2 = vib.Dipole(grid, order=2)
#dm2.read_np(os.path.join(path,'Dm2_g16.npy'))

# Run VSCF calculations for these potentials
# Here we solve only for the vibrational ground state

dVSCF = vib.VSCF2D(v1h,v2h)
#dVSCF = vib.VSCFDiag(v1)
dVSCF.solve()

# Now run VCI calculations using the VSCF wavefunction

VCI = vib.VCI(dVSCF.get_groundstate_wfn(), v1h,v2h)
VCI.generate_states(2) # singles only # for further calculations with test case water use triples (3)
VCI.solve()
VCI.print_results(which=8, maxfreq=8000)

# Calculate Transition Matrix
#transm0 = VCI.calculate_transition_moments([dm1],[dm2])
transm, inten, freqs = VCI.calculate_transition_matrix([dm1h])

#np.savetxt('intenmatharm.txt',inten)
#np.savetxt('freqsmatharm.txt',freqs)

np.save('VCI_intensities_harm.npy',inten)
np.save('VCI_frequencies_harm.npy',freqs)
np.save('VCI_dipolemoments_harm.npy',transm)



print 
print 
print vib.Misc.fancy_box('http://www.christophjacob.eu')