import numpy as np
from numpy import linalg as LA
import itertools
from Irspec2d import *

class BuildExcitonModel():
    '''
    Add text.
    
    '''
    
    def __init__(self, noscill):
        '''
        Add text. 
        
        '''
        
        self.noscill = noscill
        self.combinations = self.eval_combinations()
        self.nmodes = int(1 + 2*self.noscill + (self.noscill*(self.noscill-1)/2))
        
    def eval_combinations(self):
        '''
        Add Text.
        
        '''
        return int(int((self.noscill*(self.noscill-1))/2))
        
    def build_wavenumbers(self):
        '''
        Add Text.
        
        '''
        wavenumbers = [0]
        for i in range(1,self.noscill+1):
            wavenumbers.append('w'+str(i))
        return wavenumbers
    
    def build_dipmoments(self):
        '''
        Add Text.
        
        '''
        dipmoments = []
        for i in range(1,self.noscill+1):
            dipmoments.append('u'+str(i))
        return dipmoments
    
    def build_indices(self):
        '''
        Add Text.

        '''
        indices = [] # the indices of the operators in the combination terms of the hamilton function
        for j in range(1,self.noscill+1) :
            for k in range(1,self.noscill+1) : # all possible combinations of operators
                if j!=k and j<k :        # makes sure that every combination appears just once and no number appears twice
                    indices.append((j,k))
        return indices
    
    def build_hamiltonianfunction(self):
        '''
        Add Text.

        '''
        hamilt = [] # builds the actual hamilton function
        wavenumbers = self.build_wavenumbers()
        indices = self.build_indices()

        for i in range(self.noscill) :

            if i<self.noscill : # evaluates the first sum term
                
                hamilt.append(['h'+str(wavenumbers[i+1]),i+1,i+1]) # hbar omega_i b^dagger_i b_i

            if i>self.noscill-1 and i<self.noscill+self.combinations : # evaluates the second double sum term
                
                hamilt.append(['b'+str(indices[i-self.noscill][0])+str(indices[i-self.noscill][1]), \
                                 indices[i-self.nmodes][0],indices[i-self.nmodes][1]]) # beta_ij b^dagger_i b_j
                
                hamilt.append(['b'+str(indices[i-self.noscill][0])+str(indices[i-self.noscill][1]), \
                                 indices[i-self.noscill][1],indices[i-self.noscill][0]]) # beta_ij b^dagger_j b_i
        return hamilt
    
    def calc_states(self) : # evaluates all possible states
        '''
        Add Text.
        
        '''

        state_list_sort = []
        state_list_sorted = []
        combination_list = []

        state_list = list(itertools.product([0,1,2],repeat=self.noscill)) # gets all possible combinations of 0,1,2.

        for i in state_list : 
            if sum(i)<3 : # all combinations with a sum >2 are sorted out, because they would access higher states
                state_list_sort.append(i) 

        state_list_sort = sorted(state_list_sort, key=lambda x: (x[::-1])) # sorts the list of tuples from right to left 

        for i in state_list_sort : 
            if sum(i) == 0 : state_list_sorted.append(i) # adds the ground state
        for i in state_list_sort : 
            if sum(i) == 1 : state_list_sorted.append(i) # adds the states with just one (2)
        for i in state_list_sort : 
            if any(j == 2 for j in i) : state_list_sorted.append(i) # adds the states with just one (2)
        for i in state_list_sort : 
            if any(j==1 for j in i) and sum(i)==2 : # this extracts the combination states
                combination_list.append(i)
        for i in sorted(combination_list,reverse=True) : state_list_sorted.append(i) # adds the (sorted) combination states

        return state_list_sorted
    
    def eval_hamiltonmatrix(self):
        '''
        Add Text.
        
        '''
        
        hamiltonian = [[0 for i in range(self.nmodes)] for i in range(self.nmodes)] # this is the hamiltonian matrix
        states = self.calc_states()
        hamilt = self.build_hamiltonianfunction()

        for i, rstate in enumerate(states) : # loop over all the states in order to evaluate the matrix elements: < state | H | state >
            values = []

            for j, operator in enumerate(hamilt) : # loop over all elements in H, in order to evaluate: H | state >
                if rstate[operator[2]-1] != 0 :

                    creation = operator[2]-1
                    annihilation = operator[1]-1
                    newstate = list(rstate)

                    # first apply annihilation operator b | n > = sqrt(n) | n-1 >
                    fac = np.sqrt(newstate[creation]) # sqrt(n)
                    newstate[creation] = newstate[creation]-1 # | n-1 >

                    # then apply creation operator b^dagger | n > = sqrt(n+1) | n+1 >
                    fac *= np.sqrt(newstate[annihilation]+1) # sqrt(n+1)
                    newstate[annihilation] = newstate[annihilation]+1 # | n+1 >

                    #print(operator, rstate, '-->', fac, operator[0], newstate)
                    values.append([operator[0],fac,newstate])

            for j, lstate in enumerate(states) : # loop over the left states < state |
                leftstate = list(lstate)
                matelement = []

                for val in values :
                    op = val[0]
                    prefactor = val[1]
                    rightstate = val[2]

                    if rightstate == leftstate :
                        #print('found non-vanishing element: ', leftstate, rightstate, prefactor, op, i,j)

                        matelement.append(prefactor)
                        matelement.append(op)

                        hamiltonian[i][j] = matelement
                        
        return hamiltonian
    
    
    def eval_dipolmatrix(self):
        '''
        Add Text.
        
        '''
        states = self.calc_states()
        dipmoments = self.build_dipmoments()
        dipmatrix = [[0 for i in range(self.nmodes)] for i in range(self.nmodes)] # this is the hamiltonian matrix

        for i, rstate in enumerate(states) :

            for j, lstate in enumerate(states) :

                #if i>j and abs(sum(lstate)-sum(rstate))<2 and not sum(lstate)==sum(rstate)==1: 
                if i>j and abs(sum(lstate)-sum(rstate))<2 and sum(lstate)+sum(rstate)<4 and not sum(lstate)==sum(rstate)==1: 

                    for k, operator in enumerate(dipmoments) : 

                        #print('***')

                        rlist = list(rstate)
                        llist = list(lstate)
                        rindex = rlist.pop(k)
                        lindex = llist.pop(k)

                        #print(i,j,k,lstate,operator,rstate,'|',llist,rlist)

                        if rlist == llist:

                            matelement = []

                            if lindex>0:
                                prefac = np.sqrt(lindex)
                                #print('l',prefac)
                            if rindex>1:
                                prefac = prefac*np.sqrt(rindex)
                                #print('r',prefac)
                            else:
                                prefac = 1

                            #print(lstate,operator,rstate,lindex,rindex)
                            #print(llist,rlist)

                            matelement.append(prefac)
                            matelement.append(operator)

                            dipmatrix[i][j] = matelement
                            dipmatrix[j][i] = matelement
        return dipmatrix