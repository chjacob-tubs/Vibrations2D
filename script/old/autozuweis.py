import math
import VibTools
import Vibrations as vib
import numpy as np
import numpy.linalg

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
    tmploc = VibTools.LocVib(tmpmodes, 'PM')
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
        n = subset[1] - subset[0]
        total += n


    print 'Modes localized: %i, modes in total: %i' %(total, modes.nmodes)

    if total > modes.nmodes:
        raise Exception('Number of modes in the subsets is larger than the total number of modes')
    else:
        cmat = np.zeros((total, total))
        actpos = 0 #actual position in the cmat matrix
        for subset in subsets:
            tmp = localize_subset(modes, range(subset[0], subset[1]))
            modes_mw = np.concatenate((modes_mw, tmp[0]), axis = 0)
            freqs = np.concatenate((freqs, tmp[1]), axis = 0)
            cmat[actpos:actpos + tmp[2].shape[0],actpos:actpos + tmp[2].shape[0]] = tmp[2]
            actpos = actpos + tmp[2].shape[0] 
        localmodes = VibTools.VibModes(total, modes.mol)
        localmodes.set_modes_mw(modes_mw)
        localmodes.set_freqs(freqs)

        return localmodes, cmat

def print_locmode_overview(lv):
    print '%10s %13s %10s' %('Normal','Localized','L-VCI-S')
    print '%2s %10s %10s %10s' %('No','Freq.','Freq.','Freq.')
    print '-'*36

    vci_freqs = np.linalg.eigh(lv.get_vcismat())[0]

    diffs = [vci_freqs[i] - lv.startmodes.freqs[i] for i in range(lv.nmodes)]
    for i in range(lv.nmodes):
        print '%2i %10.1f %10.1f %10.1f %10.3f' %(i, lv.startmodes.freqs[i], lv.locmodes.freqs[i], 
                                                       vci_freqs[i], diffs[i])

    print
    print "Max: %10.3f " % np.max(np.abs(diffs))
    print "MAD: %10.3f " % (np.sum(np.abs(diffs))/lv.nmodes)
    print
    print "final p: %6.3f" % lv2.calc_p(lv2.locmodes.modes_mw)

def print_subset_summary(lv):
    if lv.subsets is not None :
        print "Subsets: ", lv.subsets # shift subset indices by one

    print "final p: %6.3f        " % lv.calc_p(lv.locmodes.modes_mw),
    
    vci_freqs = np.linalg.eigh(lv.get_vcismat())[0]
    diffs = [vci_freqs[i] - lv.startmodes.freqs[i] for i in range(lv.nmodes)]

    print "Max: %6.3f        " % np.max(np.abs(diffs)),
    print "MAD: %6.3f        " % (np.sum(np.abs(diffs))/lv.nmodes)
    print


# The vibrations script begins here

# Read in normal modes from SNF results
# using VibTools (LocVib package)

res = VibTools.SNFResults()
res.read()

mol = VibTools.VibToolsMolecule()
mol.read_from_coord(filename='coord')

print "Normal modes: "
lv = VibTools.LocVib(res.modes, 'PM')
print_subset_summary(lv)

for maxerr in [2.0, 1.0, 0.1] :
    print "Maxerr = %6.1f: " % maxerr

    lv = VibTools.LocVib(res.modes, 'PM')
    lv.localize_automatic_subsets(maxerr)
    print_subset_summary(lv)

