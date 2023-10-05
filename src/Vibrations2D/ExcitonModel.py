import numpy as np
from numpy import linalg as LA

from Vibrations2D import Calc2dir_base

# EXCITON MODEL FUNCTIONS
"""Exciton model functions."""

class excitonmodel(Calc2dir_base):
    '''
    :param cmat: Coupling matrix,
        which is in the shape of the one-exciton hamiltonian
    :type cmat: List of lists of floats

    :param dipoles: Matrix of dipole moments
    :type dipoles: List of lists of lists
    '''

    def __init__(self, cmat: np.ndarray, dipoles: np.ndarray):
        '''
        :param cmat: Coupling matrix,
            which is in the shape of the one-exciton hamiltonian
        :type cmat: List of lists of floats

        :param dipoles: Matrix of dipole moments
        :type dipoles: List of lists of lists
        '''
        self.cmat = cmat
        self.dipoles = dipoles
        self.noscill = len(cmat)

    def generate_sorted_states(self) -> list:
        '''
        We need the first and second excitations
        for a system with n_oscill Oscillators.
        n_oscill is the length of the tuples for the states.
        The digits are [0,1,2], refering to the ground,
        first and second excited states.

        Example:

        n_oscill = 2

        We have the ground state (0,0),
        two first excited states (0,1) and (1,0) and
        three second excited states (0,2), (2,0) and (1,1).

        n_oscill = 3

        (0,0,0),
        (0,0,1), (0,1,0), (1,0,0),
        (0,0,2), (0,2,0), (2,0,0), (0,1,1), (1,0,1), (1,1,0)

        :return: a sorted list of the possible states
        :rtype: list of tuples of ints
        '''
        states = []

        def generate_states(*args, repeat=1):
            '''
            Generates combinations.
            Roughly the same as Itertools.product.

            Example:
            product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
            '''
            pools = [tuple(pool) for pool in args] * repeat
            result = [[]]
            for pool in pools:
                result = [x+[y] for x in result for y in pool]
            for prod in result:
                yield tuple(prod)

        # Get all combinations of 0,1,2 for a tuple with length n,
        # while the sum of the tuple is less than 3
        init_states = [i for i in list(generate_states([0, 1, 2],
                                       repeat=self.noscill)) if sum(i) < 3]

        # Resort the states from left to right
        # E.g.: [(0,0),(1,0),(0,1)] --> [(0,0),(0,1),(1,0)]
        init_states = sorted(init_states, key=lambda x: (x[::-1]))

        # Add ground state to the states list:
        for i in init_states:
            if sum(i) == 0:
                states.append(i)
        # Add states with sum = 1:
        for i in init_states:
            if sum(i) == 1:
                states.append(i)
        # Add states with just one two (2):
        for i in init_states:
            if any(j == 2 for j in i):
                states.append(i)
        # Extract combination states:
        for i in init_states:
            if any(j == 1 for j in i) and sum(i) == 2:
                states.append(i)

        return states

    def eval_hamiltonian(self, states: list) -> np.ndarray:
        '''
        Evaluates the Hamiltonian used for the Exciton Model
        from the local mode coupling matrix.
        This does not yet include the anharmonicity.

        :param states: Sorted list of possible states
        :type states: List of tuples of ints

        :return: Exciton Model Hamiltonian
        :rtype: List of lists of floats
        '''
        lenstates = len(states)  # the number of states
        hamiltonian = np.zeros((lenstates, lenstates))

        # loop over all the states in order to evaluate the matrix elements:
        # < state | H | state >
        for i, rstate in enumerate(states):

            values = []

            for a in range(len(self.cmat)):
                for b in range(len(self.cmat)):

                    if rstate[b] != 0:

                        creation = b
                        annihilation = a
                        newstate = list(rstate)

                        # first apply annihilation operator:
                        # b | n > = sqrt(n) | n-1 >
                        fac = np.sqrt(newstate[creation])  # sqrt(n)
                        newstate[creation] = newstate[creation]-1  # | n-1 >

                        # then apply creation operator:
                        # b^dagger | n > = sqrt(n+1) | n+1 >
                        fac *= np.sqrt(newstate[annihilation]+1)  # sqrt(n+1)

                        # | n+1 >
                        newstate[annihilation] = newstate[annihilation]+1

                        # print([coup_lm[a][b],fac,newstate])
                        values.append([self.cmat[a][b], fac, newstate])

            # loop over the left states: < state |
            for j, lstate in enumerate(states):
                leftstate = list(lstate)

                for val in values:
                    op = val[0]
                    prefactor = val[1]
                    rightstate = val[2]

                    if rightstate == leftstate:
                        hamiltonian[i][j] += prefactor * op

        return hamiltonian

    def eval_dipolmatrix(self, states: list) -> np.ndarray:
        '''
        Evaluates the Transition Dipole Moment Matrix used
        for the Exciton Model
        from the local mode transition dipole moments.

        :param states: Sorted list of possible states
        :type states: List of tuples of ints

        :return: Exciton Model Hamiltonian
        :rtype: List of lists of floats
        '''
        lenstates = len(states)  # the number of states
        dipmatrix = [[[0, 0, 0] for i in range(lenstates)]
                     for i in range(lenstates)]

        for i, rstate in enumerate(states):
            for j, lstate in enumerate(states):

                if (i > j
                        and abs(sum(lstate)-sum(rstate)) < 2
                        and sum(lstate)+sum(rstate) < 4
                        and not sum(lstate) == sum(rstate) == 1):

                    for k, val in enumerate(self.dipoles):

                        rlist = list(rstate)
                        llist = list(lstate)
                        rindex = rlist.pop(k)
                        lindex = llist.pop(k)

                        # print(i,j,k,lstate,operator,rstate,'|',llist,rlist)

                        if rlist == llist:

                            if lindex > 0:
                                prefac = np.sqrt(lindex)
                                # print('l',prefac)
                            if rindex > 1:
                                prefac = prefac*np.sqrt(rindex)
                                # print('r',prefac)
                            else:
                                prefac = 1

                            # print(prefac,val,prefac*val)
                            dipmatrix[i][j] = list(
                                map(lambda x: prefac*x, val)
                                )
                            dipmatrix[j][i] = list(
                                map(lambda x: prefac*x, val)
                                )

        return dipmatrix

    def add_anharmonicity(self, hamiltonian: np.ndarray,
                          anharm: float) -> np.ndarray:
        '''
        Adds the anharmonicitiy shift Delta to the second excited matrix
        elements, exept for the combination bands.
        For a two oscillator system, the diagonal elements are:
        (0), (hbar omega1), (hbar omega2),
        (2 hbar omega1 - Delta),
        (2 hbar omega2 - Delta),
        (hbar omega1 + hbar omega2).

        :param hamiltonian: Hamiltonian Matrix
        :type hamiltonian: List of lists of floats

        :return: Hamiltonian including anharmoncity
        :rtype: List of lists of floats
        '''
        for i in range(self.noscill+1, 2*self.noscill+1):

            hamiltonian[i][i] -= anharm

        return hamiltonian

    def add_all_anharmonicity(self, hamiltonian: np.ndarray,
                              anharm: float) -> np.ndarray:
        '''
        Adds the anharmonicitiy shift Delta to the all diagonal elements in
        addition to the second excitation elements.
        For a two oscillator system, the diagonal elements are:
        (0), (hbar omega1 - Delta), (hbar omega2 - Delta),
        (2 hbar omega1 - 3 Delta),
        (2 hbar omega2 - 3 Delta),
        (hbar omega1 + hbar omega2 - 2 Delta).


        :param hamiltonian: Hamiltonian Matrix
        :type hamiltonian: List of lists of floats

        :return: Hamiltonian including anharmoncity (Delta = anharm)
        :rtype: List of lists of floats
        '''
        # loop leaves out the first state, which is 0.
        for i in range(1, len(hamiltonian)):
            # first excitations: H[i][i] hbar*omega_i - Delta
            if i < self.noscill+1:
                hamiltonian[i][i] -= anharm

            # second excitations: H[i][i] hbar*omega_i - 3 * Delta
            elif i > self.noscill and i < 2*self.noscill+1:
                hamiltonian[i][i] -= 3 * anharm

            # combination excitations: H[i][i] hbar*omega_i - 2 * Delta
            elif i > 2*self.noscill:
                hamiltonian[i][i] -= 2 * anharm

        return hamiltonian

    def get_nm_freqs_dipolmat(self, anharm: float, shift='all') -> np.ndarray:
        '''
        Calculates the normal mode frequencies and transition dipole
        moment matrix.
        Can also calculate the normal mode hamiltonian.

        :param anharm: anharmonic shift added onto the exciton hamiltonian
        :type anharm: Integer or float

        :param shift: determines how the anharmonicities are added
        :type shift: String

        :return: Frequencies and transition dipole moment matrix
        :rtype: Two lists of lists of float
        '''
        states = self.generate_sorted_states()
        if shift == 'all':
            hamilt_lm = self.add_all_anharmonicity(
                self.eval_hamiltonian(states),
                anharm
                )
        if shift == 'exc2':
            hamilt_lm = self.add_anharmonicity(
                self.eval_hamiltonian(states),
                anharm
                )
        dipole_lm = self.eval_dipolmatrix(states)

        ew, ev = LA.eigh(hamilt_lm)

        # ##  The normal mode hamiltonian can be calculated, but is not needed
        # ## in order to calculate 2D IR spectra.
        # hamilt_nm = np.dot(np.dot(ev.T,hamilt_lm),ev)

        firstmult = np.tensordot(ev.T, dipole_lm, axes=1)
        dipole_nm = np.tensordot(ev, firstmult, axes=(0, 1))

        # simulating a symmetrical frequency matrix from the eigen values:
        freqs_lm = np.zeros((len(ew), len(ew)))
        for i in range(len(ew)):
            freqs_lm[i][0] = ew[i]
            freqs_lm[0][i] = ew[i]

        return freqs_lm, dipole_nm

    def get_nm_freqs_dipolmat_from_VSCF(self, VSCF_freqs: list) -> np.ndarray:
        '''
        Calculates the normal mode frequencies and transition dipole
        moment matrix without any need for anharmonic shift parameters,
        as it puts L-VSCF frequencies instead of harmonic frequencies into
        the calculation.

        :param VSCF_freqs: frequencies obtained from L-VSCF calculation
        :type shift: List of float

        :return: Frequencies and transition dipole moment matrix
        :rtype: Two lists of lists of float
        '''
        states = self.generate_sorted_states()

        hamilt_lm = self.eval_hamiltonian(states)
        dipole_lm = self.eval_dipolmatrix(states)

        if len(VSCF_freqs) != len(hamilt_lm):
            raise ValueError('The length of the given list '
                             'is not equal to the length of '
                             'the calculated hamiltonian matrix.')
        else:
            for i in range(len(hamilt_lm)):
                hamilt_lm[i][i] = VSCF_freqs[i]

        ew, ev = LA.eigh(hamilt_lm)

        firstmult = np.tensordot(ev.T, dipole_lm, axes=1)
        dipole_nm = np.tensordot(ev, firstmult, axes=(0, 1))

        # simulating a symmetrical frequency matrix from the eigen values:
        freqs_lm = np.zeros((len(ew), len(ew)))
        for i in range(len(ew)):
            freqs_lm[i][0] = ew[i]
            freqs_lm[0][i] = ew[i]

        return freqs_lm, dipole_nm
