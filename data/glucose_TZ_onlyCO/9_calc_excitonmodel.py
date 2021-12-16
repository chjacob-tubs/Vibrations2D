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
    tmpcmat = tmploc.get_couplingmat(hessian=False) #if hessian=False, returns [cm-1] otherwise returns [hartree]

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
path = '/home/julia/testsystems/glucose/def2tzvp_b3lyp_V1V2_CO'

res = LocVib.SNFResults(outname=os.path.join(path,'snf.out'),
                        restartname=os.path.join(path,'restart'),
                        coordfile=os.path.join(path,'coord'))
res.read()

# Now localize modes in separate subsets

subsets = np.load(os.path.join(path,'subsets.npy'))
#subsets = [[28,29]]
localmodes,cmat = localize_subsets(res.modes,subsets)

print "\n cmat \n",cmat

dipoles = res.get_tensor_deriv_nm('dipole', modes=localmodes)
#print "\n dipoles local modes: \n",dipoles
dipoles_norm = res.get_tensor_deriv_nm('dipole', modes=res.modes)
#print "\n dipoles normal modes: \n 28: ",dipoles_norm[28],'\n 29: ',dipoles_norm[29],'\n \n',dipoles_norm


np.save('Exciton_cmat_lm.npy', cmat)
np.save('Exciton_dipolemoments_lm.npy', dipoles)
np.save('Exciton_dipolemoments_nm.npy', dipoles_norm)