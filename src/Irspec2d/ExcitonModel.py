import numpy as np
from numpy import linalg as LA

from Calc2dir import *

# EXCITON MODEL FUNCTIONS

class excitonmodel(Calc2dir_base):
    
    def __init__(self, cmat, dipoles, anharm):
        '''
        
        @param cmat: coupling matrix, which is in the shape of the one-exciton hamiltonian
        @type cmat: list of lists of floats
        '''
        self.cmat = cmat
        self.dipoles = dipoles
        self.anharm = anharm
        self.noscill = len(cmat)
        
    @staticmethod
    def multiply(A,B):
        '''
        For multiplying the (m,m,3)x(m,m) arrays.
        Works also for simple (m,n)x(n,l) multiplications.
        No further cases were tested.
        
        @param A: (m,m,3) shaped array
        @type A: array or list of lists of float
        
        @param B: (m,m) shaped array
        @type B: array or list of lists of float
        
        @return: product of A and B
        @rtype: list of lists of float

        '''
        a = int(np.asarray(A).shape[0])
        b = int(np.asarray(B).shape[1])

        C = [[0 for i in range(b)] for j in range(a)]

        for i, ii in enumerate(C):
            for j, jj in enumerate(ii):

                D = []
                for k in B:
                    # print(k[j])
                    D.append(k[j])

                matelement = []
                for k,kk in enumerate(A[i]):
                    prod = np.dot(kk,D[k])
                    # print('kk:',kk,'D[i]:',D[k],'prod:',prod)
                    matelement.append(prod)

                C[i][j] = sum(matelement)

        return C
    
    def generate_sorted_states(self):
        '''
        We need the first and second excitations for a system with n_oscill Oscillators.
        n_oscill is the length of the tuples for the states.
        The digits are [0,1,2], refering to the ground, first and second excited states.
        
        Example: 
        n_oscill = 2
        We have the ground state (0,0), two first excited states (0,1) and (1,0) and
        three second excited states (0,2), (2,0) and (1,1). 
        n_oscill = 2
        (0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,0,2), (0,2,0), (2,0,0), (0,1,1), (1,0,1), (1,1,0)
        
        @return: a sorted list of the possible states
        @rtype: list of tuples of ints
        
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

        # Get all combinations of 0,1,2 for a tuple with length n, while the sum of the tuple is less than 3
        init_states = [i for i in list(generate_states([0,1,2],repeat=self.noscill)) if sum(i) < 3]
        
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
    
    def eval_hamiltonian(self,states):
        '''
        Evaluates the Hamiltonian used for the Exciton Model
        from the local mode coupling matrix.
        This does not yet include the anharmonicity.
        
        @param states: Sorted list of possible states
        @type states: List of tuples of ints
        
        @return: Exciton Model Hamiltonian
        @rtype: List of lists of floats
        
        '''
        lenstates = len(states) # the number of states
        hamiltonian = [[0 for i in range(lenstates)] for i in range(lenstates)] 

        # loop over all the states in order to evaluate the matrix elements: < state | H | state >
        for i, rstate in enumerate(states) : 
            
            values = []
            # print(rstate)
            
            for a in range(len(self.cmat)):
                for b in range(len(self.cmat)):
                    
                    if rstate[b] != 0:

                        creation = b
                        annihilation = a
                        newstate = list(rstate)

                        # first apply annihilation operator b | n > = sqrt(n) | n-1 >
                        fac = np.sqrt(newstate[creation]) # sqrt(n)
                        newstate[creation] = newstate[creation]-1 # | n-1 >

                        # then apply creation operator b^dagger | n > = sqrt(n+1) | n+1 >
                        fac *= np.sqrt(newstate[annihilation]+1) # sqrt(n+1)
                        newstate[annihilation] = newstate[annihilation]+1 # | n+1 >

                        # print([coup_lm[a][b],fac,newstate])
                        values.append([self.cmat[a][b],fac,newstate])

            # loop over the left states < state |
            for j, lstate in enumerate(states) : 
                leftstate = list(lstate)

                for val in values :
                    op = val[0]
                    prefactor = val[1]
                    rightstate = val[2]

                    if rightstate == leftstate :
                        #print('found non-vanishing element: ', leftstate, rightstate, prefactor, op, i,j)

                        hamiltonian[i][j] += prefactor*op
                        
        return hamiltonian 
    
    def eval_dipolmatrix(self,states):
        '''
        Evaluates the Transition Dipole Moment Matrix used for the Exciton Model
        from the local mode transition dipole moments.
        
        @param states: Sorted list of possible states
        @type states: List of tuples of ints
        
        @return: Exciton Model Hamiltonian
        @rtype: List of lists of floats
        
        '''
        lenstates = len(states) # the number of states
        dipmatrix = [[[0,0,0] for i in range(lenstates)] for i in range(lenstates)] 

        for i, rstate in enumerate(states) :
            for j, lstate in enumerate(states) :

                if i>j and abs(sum(lstate)-sum(rstate))<2 and sum(lstate)+sum(rstate)<4 and not sum(lstate)==sum(rstate)==1: 

                    for k, val in enumerate(self.dipoles) : 

                        rlist = list(rstate)
                        llist = list(lstate)
                        rindex = rlist.pop(k)
                        lindex = llist.pop(k)

                        #print(i,j,k,lstate,operator,rstate,'|',llist,rlist)

                        if rlist == llist:

                            if lindex>0:
                                prefac = np.sqrt(lindex)
                                #print('l',prefac)
                            if rindex>1:
                                prefac = prefac*np.sqrt(rindex)
                                #print('r',prefac)
                            else:
                                prefac = 1

                            # print(prefac,val,prefac*val)
                            dipmatrix[i][j] = list(map(lambda x : prefac*x, val))
                            dipmatrix[j][i] = list(map(lambda x : prefac*x, val))
                            
        return dipmatrix
    
    def add_anharmonicity(self,hamiltonian):
        '''
        Adds the anharmonicitiy factor to the second excited matrix 
        elements, exept for the combination bands.
        
        @param hamiltonian: Hamiltonian Matrix
        @type hamiltonian: List of lists of floats
        
        @return: Hamiltonian including anharmoncity
        @rtype: List of lists of floats
        
        '''
        for i in range(self.noscill+1,2*self.noscill+1):
            
            hamiltonian[i][i] -= self.anharm
            
        return hamiltonian
    
    def get_nm_freqs_dipolmat(self):
        '''
        Calculates the normal mode frequencies and transition dipole
        moment matrix. 
        Can also calculate the normal mode hamiltonian.
        
        @return: Frequencies and transition dipole moment matrix
        @rtype: Two lists of lists of float
        
        '''
        states = self.generate_sorted_states()
        hamilt_lm = self.add_anharmonicity(self.eval_hamiltonian(states))
        dipole_lm = self.eval_dipolmatrix(states)
        
        ew, ev = LA.eigh(hamilt_lm)
        
        ### The normal mode hamiltonian can be calculated, but is not needed
        ### in order to calculate 2D IR spectra.
        # hamilt_nm = np.dot(np.dot(ev.T,hamilt_lm),ev)
        
        firstmult = self.multiply(ev.T,dipole_lm)
        dipole_nm = self.multiply(firstmult,ev)
        
        return [ew], dipole_nm
        