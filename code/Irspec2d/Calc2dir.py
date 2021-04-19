import numpy as np
from numpy import linalg as LA

class basics():
    '''
    This class includes all basic functions that are needed for the calculation of 2D IR spectra.
    
    '''
    
    def __init__(self, freqmat, transmat):
        '''
        Add text. 
        
        '''
        
        self.freqmat = freqmat
        self.transmat = self.read_transmat(transmat)
        self.nmodes = self.calc_nmodes()
        self.noscill = self.calc_num_oscill(self.nmodes)
        self.intfactor = 2.5066413842056297 # factor to calculate integral absorption coefficient having  [cm-1]  and  [Debye] ; see Vibrations/Misc.py code
        
        if self.check_symmetry(self.transmat) == False:
            print('Transition dipole moment matrix is not (skew-)symmetrical. Please check!')
        if self.check_symmetry(self.freqmat) == False:
            print('Frequency matrix is not (skew-)symmetrical. Please check!')
        
        
    def calc_nmodes(self):
        '''
        Returns the number of modes.
        
        '''
        if len(self.transmat) == len(self.freqmat):
            return int(len(self.freqmat))
        else:
            raise ValueError('The matrices containing the frequencies and the transition dipole moments do not have the same length.')
        
    def calc_nmodesexc(self):
        '''
        Returns the number of excited modes.
        n_modesexc = n_oscill + (n_oscill*(n_oscill-1))/2
        
        '''
        return int(self.noscill + (self.noscill*(self.noscill-1))/2)
    
    def calc_num_oscill(self,nmodes):
        '''
        Calculates the number of oscillators n_oscill based on a 
        given number of modes n_modes. This is based on the assumption 
        that there are 
           n_modes = 1 + 2n_oscill + (n_oscill*(n_oscill-1))/2 
        modes. There is one ground state (0) plus n first excited states 
        plus n second excited states plus (n_oscill*(n_oscill-1))/2 combination 
        states. 
        This leads to 
           n_oscill = 1/2 * (-3 + sqrt(8*n_modes - 1)).
        
        
        If there are 
           n_modes = 2n_oscill + (n_oscill*(n_oscill-1))/2
        modes, then 
           n_oscill = 1/2 * (-3 + sqrt(8*n_modes + 9)).

        '''
        noscill = (-3. + np.sqrt(8.*nmodes +1.) ) /2.
        if nmodes == 0: print('There are no modes, because nmodes=0.')
        if noscill-int(noscill)==0:
            return int(noscill)
        else:
            new_noscill = (-3. + np.sqrt(8.*nmodes +9.) ) /2.
            if new_noscill-int(new_noscill)==0:
                return int(new_noscill)
            else:
                raise ValueError("Number of Oscillators couldn't be evaluated.")
        
    
    def check_symmetry(self, a, tol=1e-5):
        '''
        Checks if a given matrix a is symmetric or skew-symmetric.
        Returns True/False.
        
        If the matrix is three-dimensional, as in case of the 
        transition dipole moment matrix, the function transposes 
        the matrix with respect to the axes. This leads to 
        leaving the single vectors as vectors. 

        '''
        if len(a.shape)==2:
            return np.all(np.abs(abs(a)-abs(a).T) < tol)
        if len(a.shape)==3:
            return np.all(np.abs(abs(a)-np.transpose(abs(a),(1,0,2))) < tol)
        else:
            raise ValueError('The shape of the given matrix is not implemented in the check_symmetry function.')
    
    def read_transmat(self,oldtransmat):
        '''
        The transition dipole moment matrix that is obtained by VIBRATIONS calculations
        has the shape (n,n,1,3). In order to use it in the following calculations it is
        reduced to the shape (n,n,3). 
        
        '''
        return np.reshape(oldtransmat,(len(oldtransmat),len(oldtransmat),3))
    
    def calc_trans2int(self):
        '''
        Calculates the intensity matrix from the given transition dipole moment matrix and the given frequeny matrix.
        
        '''
        intenmat = np.zeros_like(self.freqmat)
        for i in range(len(intenmat)):
            for j in range(len(intenmat)):
                intenmat[i][j]= (LA.norm(self.transmat[i][j]))**2 * self.intfactor * self.freqmat[i][j]
        return intenmat
    



class calc_2dirsimple(basics):
    '''
    This class is supposed to evaluate 2D IR spectra in a simple approach.
    
    '''
    
    def __init__(self, freqmat, transmat, verbose_all=False):
        
        super().__init__(freqmat, transmat)
        self.freqs = self.freqmat[0]
        self.verbose_all = verbose_all
        self.intmat = self.calc_trans2int()
    
    def calc_excitation(self,verbose=False):
        '''
        Takes the energy levels and the intensity matrix in order to find 
        the excited state absorption processes that occur in an 2D IR
        experiment. 

        '''
        exc_x = [] # excitation coords
        exc_y = [] 
        exc_i = [] # intensity

        for i in range(len(self.intmat)):
            if self.intmat[0][i] and i<=self.noscill:
                for j in range(len(self.intmat)):
                    if j>i:

                        y_coor = self.freqs[j]-self.freqs[i]
                        x_coor = self.freqs[i]-self.freqs[0]
                        exc_inten = self.intmat[i][j]

                        exc_y.append(y_coor)
                        exc_x.append(x_coor)
                        exc_i.append(exc_inten)

                        if verbose_all or verbose : print('Excitation from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',exc_inten)
        return (exc_x, exc_y, exc_i)
    
    def calc_stimulatedemission(self,verbose=False):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the stimulated emission processes that occur in an 2D IR experiment.
        In order to match the experiment the stimulated emission can only 
        happen in transition to the ground state energy level!

        '''
        emi_x = [] # excitation coords
        emi_y = [] 
        emi_i = [] # intensity

        for i in range(len(self.intmat)):
            for j in range(len(self.intmat)):
                if j==0 and i>j and i<=self.noscill:

                    y_coor = self.freqs[i]-self.freqs[j]
                    x_coor = self.freqs[i]-self.freqs[j]
                    emi_inten = self.intmat[j][i]

                    emi_y.append(y_coor)
                    emi_x.append(x_coor)
                    emi_i.append(emi_inten)

                    if verbose_all or verbose : print('Stimulated emission from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',emi_inten)
        return (emi_x, emi_y, emi_i)

    def calc_bleaching(self,verbose=False):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the bleaching processes that occur in an 2D IR experiment.

        '''

        ble_x = [] # excitation coords
        ble_y = [] 
        ble_i = [] # intensity

        for i in range(len(self.intmat)):
            if self.intmat[0][i] != 0 and i<=self.noscill:
                
                y_coor = self.freqs[i]-self.freqs[0]
                ble_inten = self.intmat[0][i]
                
                for j in range(len(self.intmat)):
                    if self.intmat[0][j] != 0 and j<=self.noscill:
                        x_coor = self.freqs[j]-self.freqs[0]
                        ble_x.append(x_coor)
                        ble_y.append(y_coor)
                        ble_i.append(ble_inten)
                        if verbose_all or verbose : print('Bleaching from energy level 0 to',i,'at (',x_coor,',',y_coor,') rcm and intensity: ',ble_inten)

        return (ble_x, ble_y, ble_i)
                      
    def calc_all_2d_process(self,verbose=False):
        '''
        Calculates all processes that can occur within a
        2D IR experiment from the energy levels and the
        intensity matrix. 

        '''
        return self.calc_excitation(verbose=verbose), self.calc_stimulatedemission(verbose=verbose), self.calc_bleaching(verbose=verbose)







class calc_2dirtimedomain(basics):
    '''
    This class is supposed to evaluate 2D IR spectra within a time domain approach.
    
    '''
    
    def __init__(self, freqmat, transmat, verbose_all=False, verbose=False, **params):
        '''
        Add text.
        
        '''
        super().__init__(freqmat, transmat)

        if 'n_t' in params : 
            self.n_t = params.get('n_t')
            if verbose_all or verbose : print('Set the number of time points n_t to',str(self.n_t)+'.')
        else : 
            self.n_t = 128
            if verbose_all or verbose : print('Set the number of time points n_t to',self.n_t,'(default value).')
        if 'n_zp' in params : 
            self.n_zp = params.get('n_zp')
            if verbose_all or verbose : print('Set the zeropadded length n_zp to',str(self.n_zp)+'.')
        else : 
            self.n_zp = 2*self.n_t
            if verbose_all or verbose : print('Set the zeropadded length n_zp to',self.n_zp,'(default value).')
        if 'dt' in params : 
            self.dt = params.get('dt')
            if verbose_all or verbose : print('Set the time step length dt to',self.dt,'ps.')
        else : 
            self.dt = 0.25
            if verbose_all or verbose : print('Set the time step length dt to',self.dt,'ps (default value).')
        if 'T2' in params : 
            self.T2 = params.get('T2')
            if verbose_all or verbose : print('Set the dephasing time T2 to',self.T2,'ps.')
        else : 
            self.T2 = 2
            if verbose_all or verbose : print('Set the zeropadded length T2 to',self.T2,'ps (default value).')
        if 't2' in params : 
            self.t2 = params.get('t2')
            if verbose_all or verbose : print('Set the population time t2 to',self.t2,'ps.')
        else : 
            self.t2 = 0
            if verbose_all or verbose : print('Set the population time t2 to',self.t2,'ps (default value).')
        if 'pol' in params :
            implementedpolarizations = ['ZZZZ','ZZXX']
            if params.get('pol') in implementedpolarizations:
                self.polarization = params.get('pol')
                if verbose_all or verbose : print('Set the polarization to',str(self.polarization)+'.')
            else:
                raise Exception('Could not set '+str(params.get('pol'))+' polarization condition. Please chose from '+str(implementedpolarizations)+'.')
        else : 
            self.polarization = 'ZZZZ'
            if verbose_all or verbose : print('Set the polarization to',self.polarization,'(default).')
        
        self.freqs = self.freqmat[0]
        self.verbose_all = verbose_all
        self.nmodesexc = self.calc_nmodesexc()
        self.unitconvfactor = 0.188 # unit conversion factor
        self.timepoints = np.arange(0,(self.n_t)*self.dt)
        
        if (self.nmodesexc+self.noscill+1)!=self.nmodes: 
            print('The number of modes is',self.nmodes,'not equal to the sum of the number of oscillators',self.noscill,' and the number of excited states',self.nmodesexc,' plus 1 (for the ground state).')
        
    def calc_secondexcitation(self):
        '''
        Extracts the matrix for the excited state transition dipole moments.

        '''
        exc_trans = []

        for i in range(1,len(self.transmat)):
            if i <= self.noscill:
                transcolumn = []
                for j in range(len(self.transmat)):
                    if j > self.noscill :
                        transcolumn.append(self.transmat[i][j])
                exc_trans.append(transcolumn)
        return np.asarray(exc_trans)
    
    def calc_omega_off(self,value):
        '''
        Returns the rounded average of a given set of values (frequencies).
        
        '''
        return round(sum(value)/len(value))
    
    def set_omega_off(self):
        '''
        Sets the omega_off value that is used to center the peaks, in order for better computation of the fourier transforms.
        
        '''
        return self.calc_omega_off(self.freqs[1:self.noscill+1])
    
    def set_omega_initial(self):
        '''
        There are n_oscill first excited states. 
        The first state in the frequency matrix is zero.
        Therefore, the states 1 to n_oscill are the initial frequency values.
        
        '''
        return self.freqs[1:self.noscill+1]
    
    def set_omega(self):
        '''
        In the frequency matrix, the first state is 0. 
        The states 1 to n_oscill are the first excited states.
        The states after state n_oscill are the second excited states.
        
        '''
        omega_init = self.set_omega_initital()
        omega2_init = self.freqs[self.noscill+1:]
        
        omega_off_value = self.set_omega_off()
        
        omega = [i-omega_off_value for i in omega_init]
        omega2 = [i-2*omega_off_value for i in omega2_init]
        
        if self.noscill != len(omega):
            print('The number of first excitation frequencies does not equal the number of oscillators.')
        if self.nmodesexc != len(omega2):
            print('The number of second excitation frequencies does not equal the number of excited states.')
        
        return omega, omega2
    
    def set_mu(self):
        '''
        This returns the seperated transition matrices for the first and second excited states.
        
        '''
        return self.transmat[0][1:self.noscill+1] , self.calc_secondexcitation()
    
    def calc_diagrams(self,verbose=False):
        '''
        This computes the diagrams R_1 to R_6.
        R_1, R_2 and R_3 are rephasing diagrams and R_4, R_5 and R_6 are non-rephasing diagrams.
        It also computes angles for different (ZZZZ, ZZXX) polarization functions.
        
        '''        
        R1 = np.zeros((self.n_t,self.n_t),dtype=np.complex_)
        R2 = np.zeros_like(R1,dtype=np.complex_)
        R3 = np.zeros_like(R1,dtype=np.complex_)
        R4 = np.zeros_like(R1,dtype=np.complex_)
        R5 = np.zeros_like(R1,dtype=np.complex_)
        R6 = np.zeros_like(R1,dtype=np.complex_)
        
        t = np.arange(0,self.n_t*self.dt,self.dt)
        mu, mu2 = self.set_mu()
        omega, omega2 = self.set_omega()

        for j in range(self.noscill):
            for i in range(self.noscill):

                if self.verbose_all or verbose : print('i:',i,'j:',j)

                mui = LA.norm(mu[i])
                muj = LA.norm(mu[j])
                cos1 = (mu[i][0]*np.conj(mu[j][0])+mu[i][1]*np.conj(mu[j][1])+mu[i][2]*np.conj(mu[j][2])) / (mui*muj)
                dipole = mui**2 * muj**2
                if self.polarization == 'ZZZZ':
                    angle = (1 + 2*cos1**2) /15
                elif self.polarization == 'ZZXX':
                    angle = (2 - cos1**2) /15
                else:
                    raise Exception('No polarization function found. This error should be cought in the init method. Please check.')
                factor = angle*dipole

                if self.verbose_all or verbose : print('mu_i:',mui,'mu_j:',muj)
                if self.verbose_all or verbose : print('cos1:',cos1,'angle:',angle,'dipole:',dipole)

                for jj,T3 in enumerate(t):
                    for ii,T1 in enumerate(t):
                        R1[ii][jj] -= factor * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                        R2[ii][jj] -= factor * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 - (T1+T3)/self.T2)
                        R4[ii][jj] -= factor * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                        R5[ii][jj] -= factor * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 - (T1+T3)/self.T2)

                for k in range(self.nmodesexc):

                    if self.verbose_all or verbose : print('k:',k)

                    muik = LA.norm(mu2[i][k])
                    mujk = LA.norm(mu2[j][k])
                    dipole2 = mui*muj*muik*mujk

                    if self.verbose_all or verbose : print('mu_ik:',muik,'mu_jk:',mujk)
                    if self.verbose_all or verbose : print('dipole2:',dipole2)

                    if muik != 0 and mujk != 0 : 
                        cos2 = ( mu2[i][k][0]*np.conj(mu2[j][k][0]) + mu2[i][k][1]*np.conj(mu2[j][k][1]) + mu2[i][k][2]*np.conj(mu2[j][k][2]) ) / ( mujk*muik )
                    else: 
                        cos2 = 0 

                    if muik != 0 : 
                        cos3 = ( mu[i][0]*np.conj(mu2[i][k][0]) + mu[i][1]*np.conj(mu2[i][k][1]) + mu[i][2]*np.conj(mu2[i][k][2]) ) / ( mui*muik )
                        cos6 = ( mu[j][0]*np.conj(mu2[i][k][0]) + mu[j][1]*np.conj(mu2[i][k][1]) + mu[j][2]*np.conj(mu2[i][k][2]) ) / ( muj*muik )
                    else: 
                        cos3 = 0 
                        cos6 = 0

                    if mujk != 0 : 
                        cos4 = ( mu[j][0]*np.conj(mu2[j][k][0]) + mu[j][1]*np.conj(mu2[j][k][1]) + mu[j][2]*np.conj(mu2[j][k][2]) ) / ( muj*mujk )
                        cos5 = ( mu[i][0]*np.conj(mu2[j][k][0]) + mu[i][1]*np.conj(mu2[j][k][1]) + mu[i][2]*np.conj(mu2[j][k][2]) ) / ( mui*mujk )
                    else: 
                        cos4 = 0 
                        cos5 = 0

                    if self.verbose_all or verbose : print('cos2:',cos2)
                    if self.verbose_all or verbose : print('cos3:',cos3)
                    if self.verbose_all or verbose : print('cos4:',cos4)
                    if self.verbose_all or verbose : print('cos5:',cos5)
                    if self.verbose_all or verbose : print('cos6:',cos6)

                    if self.polarization == 'ZZZZ':
                        angle2 = (cos1*cos2 + cos3*cos4 + cos5*cos6) / 15
                    elif self.polarization == 'ZZXX':
                        angle2 = (4*cos1*cos2 - cos3*cos4 - cos5*cos6) / 30
                    else:
                        raise Exception('No polarization function found. This error should be cought in the init method. Please check. (Access to higher states.)')
                    factor2 = dipole2*angle2

                    if self.verbose_all or verbose : print('angle2:',angle2)

                    for jj,T3 in enumerate(t):
                        for ii,T1 in enumerate(t):
                            R3[ii][jj] += factor2 * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*(omega2[k]-omega[j])*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                            R6[ii][jj] += factor2 * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*(omega2[k]-omega[i])*self.unitconvfactor*T3 - 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)

                if self.verbose_all or verbose : print()

        return R1,R2,R3,R4,R5,R6
    
    def calc_axes(self):
        '''
        Calculates the time axis into a frequency axis.
        
        '''
        ticks = []
        for i in range(0,self.n_zp):
            ticks.append( (i-(self.n_zp/2))*2*np.pi / (self.unitconvfactor*self.n_zp*self.dt) + self.set_omega_off() )
        return ticks