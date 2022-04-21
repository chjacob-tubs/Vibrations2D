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
        if len(self.transmat) == len(self.freqmat[0]):
            return int(len(self.freqmat[0]))
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
    
    
    def read_transmat(self,oldtransmat):
        '''
        The transition dipole moment matrix that is obtained by VIBRATIONS calculations
        has the shape (n,n,1,3). In order to use it in the following calculations it is
        reduced to the shape (n,n,3). 
        
        '''
        if len(np.asarray(oldtransmat).shape) == 4:
            return np.reshape(oldtransmat,(len(oldtransmat),len(oldtransmat),3))
        if len(np.asarray(oldtransmat).shape) == 3:
            return oldtransmat
        
    
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
    
    def calc_trans2int(self):
        '''
        Calculates the intensity matrix from the given transition dipole moment matrix and the given frequeny matrix.
        
        '''
        intenmat = np.zeros_like(self.freqmat)
        for i in range(len(intenmat)):
            for j in range(len(intenmat)):
                intenmat[i][j]= (LA.norm(self.transmat[i][j]))**2 * self.intfactor * self.freqmat[i][j]
        return intenmat
    
    def print_matrix(self,matrix):
        '''
        Add Text.
        
        '''
        
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
    



class calc_2dirsimple(basics):
    '''
    This class is supposed to evaluate 2D IR spectra in a simple approach.
    
    '''
    
    def __init__(self, freqmat, transmat, verbose_all=False):
        
        super().__init__(freqmat, transmat)
        self.freqs = self.freqmat[0]
        self.verbose_all = verbose_all
        self.intmat = self.calc_trans2int()
        
    def set_initial_freqs(self):
        '''
        There are n_oscill first excited states. 
        The first state in the frequency matrix is zero.
        Therefore, the states 1 to n_oscill are the initial frequency values.
        
        '''
        return self.freqmat[0][1:self.noscill+1]
    
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

                        if self.verbose_all or verbose : print('Excitation from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',exc_inten)
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

                    if self.verbose_all or verbose : print('Stimulated emission from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',emi_inten)
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
                        if self.verbose_all or verbose : print('Bleaching from energy level 0 to',i,'at (',x_coor,',',y_coor,') rcm and intensity: ',ble_inten)

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
            self.polarization = params.get('pol')
            if verbose_all or verbose : print('Set the polarization to',self.polarization,'.')
        else : 
            self.polarization = 'ZZZZ'
            if verbose_all or verbose : print('Set the polarization to',self.polarization,'(default).')
        
        if 'pol_list' in params :
            self.pol_list = params.get('pol_list')
            if verbose_all or verbose : print('Set the polarization angles to',self.pol_list,'.')
        else : 
            self.pol_list = self.set_pulse_angles(self.polarization)
            if verbose_all or verbose : print('Set the polarization angles to',self.pol_list,'(calculated default).')
        
        self.n_zp = 2*self.n_t
        self.freqs = self.freqmat[0]
        self.verbose_all = verbose_all
        self.nmodesexc = self.calc_nmodesexc()
        self.unitconvfactor = 0.188 # unit conversion factor
        self.timepoints = np.arange(0,(self.n_t)*self.dt)
        self.fak1, self.fak2, self.fak3 = self.calc_fourpoint_faktors(self.pol_list)
        
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
        omega_init = self.set_omega_initial()
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
    
    def set_pulse_angles(self,pol):
        '''
        Returns a list of different angles for different polarization conditions.
        E.g. for the <ZZZZ> polarization condition the list is [0,0,0,0].
        
        '''
        
        pol_list = [0,0,0,0]
        
        for i, val in enumerate(pol):
            if val == pol[0]:
                pol_list[i] = 0
            if val != pol[0]:
                pol_list[i] = 90
        
        return pol_list
    
    def calc_fourpoint_faktors(self,pol):
        '''
        Needs the list of angles of the polarization condition.
        Calculating parts of the four-point correlation function:
        row1 = 4 * cos theta_ab * cos theta_cd - cos theta_ac * cos theta_bd - cos theta_ad * cos theta_bc
        row2 = 4 * cos theta_ac * cos theta_bd - cos theta_ab * cos theta_cd - cos theta_ad * cos theta_bc
        row3 = 4 * cos theta_ad * cos theta_bc - cos theta_ab * cos theta_cd - cos theta_ac * cos theta_bd

        '''

        ab = np.deg2rad(pol[0]-pol[1])
        cd = np.deg2rad(pol[2]-pol[3])
        ac = np.deg2rad(pol[0]-pol[2])
        bd = np.deg2rad(pol[1]-pol[3])
        ad = np.deg2rad(pol[0]-pol[3])
        bc = np.deg2rad(pol[1]-pol[2])

        row1 = 4 * np.cos(ab) * np.cos(cd) - np.cos(ac) * np.cos(bd) - np.cos(ad) * np.cos(bc)
        row2 = 4 * np.cos(ac) * np.cos(bd) - np.cos(ab) * np.cos(cd) - np.cos(ad) * np.cos(bc)
        row3 = 4 * np.cos(ad) * np.cos(bc) - np.cos(ab) * np.cos(cd) - np.cos(ac) * np.cos(bd)

        return row1,row2,row3
    
    def calc_cos(self,vec1,vec2):
        '''
        calculates the cosine between two three-dimensional vectors

        '''

        mu1 = LA.norm(vec1)
        mu2 = LA.norm(vec2)

        if mu1 != 0 and mu2 !=0:
            cos12 = ( vec1[0]*np.conj(vec2[0])+vec1[1]*np.conj(vec2[1])+vec1[2]*np.conj(vec2[2]) ) / (mu1*mu2)
        else:
            cos12 = 0

        return cos12
    
    def calc_fourpointcorr(self,pathway,fak1,fak2,fak3,*mus):
        '''
        pathway : 'jjii', 'jiji', 'jiij', 'jikl'

        S = 1/30 * ( cos theta_alpha_beta  * cos theta_gamma_delta * fak1
                   - cos theta_alpha_gamma * cos theta_beta_delta  * fak2
                   - cos theta_alpha_delta * cos theta_beta_gamma  * fak3 )

        '''

        mu = [0,0,0,0]

        for i, val in enumerate(pathway):

            if val == 'j':
                mu[i] = mus[0]

            if val == 'i':
                mu[i] = mus[1]

            if val == 'k':
                mu[i] = mus[2]

            if val == 'l':
                mu[i] = mus[3]

        S1 = fak1 * self.calc_cos(mu[0],mu[1]) * self.calc_cos(mu[2],mu[3])
        S2 = fak2 * self.calc_cos(mu[0],mu[2]) * self.calc_cos(mu[1],mu[3])
        S3 = fak3 * self.calc_cos(mu[0],mu[3]) * self.calc_cos(mu[1],mu[2])

        S = (S1 + S2 + S3) / 30

        return S
    
    
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
                
                dipole = mui**2 * muj**2
                
                angle_jjii = self.calc_fourpointcorr('jjii',self.fak1,self.fak2,self.fak3,mu[i],mu[j])
                angle_jiji = self.calc_fourpointcorr('jiji',self.fak1,self.fak2,self.fak3,mu[i],mu[j])
                angle_jiij = self.calc_fourpointcorr('jiij',self.fak1,self.fak2,self.fak3,mu[i],mu[j])

                if self.verbose_all or verbose : print('mu_i:',mui,'mu_j:',muj)
                if self.verbose_all or verbose : print('cos1:',cos1,'angle:',angle,'dipole:',dipole)

                for jj,T3 in enumerate(t):
                    for ii,T1 in enumerate(t):
                        R1[ii][jj] -= angle_jiji*dipole * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                        R2[ii][jj] -= angle_jjii*dipole * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 - (T1+T3)/self.T2)
                        R4[ii][jj] -= angle_jiij*dipole * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                        R5[ii][jj] -= angle_jjii*dipole * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 - (T1+T3)/self.T2)

                for k in range(self.nmodesexc):

                    if self.verbose_all or verbose : print('k:',k)

                    muik = LA.norm(mu2[i][k])
                    mujk = LA.norm(mu2[j][k])
                    dipole2 = mui*muj*muik*mujk

                    if self.verbose_all or verbose : print('mu_ik:',muik,'mu_jk:',mujk)
                    if self.verbose_all or verbose : print('dipole2:',dipole2)


                    angle_jilk = self.calc_fourpointcorr('jilk',self.fak1,self.fak2,self.fak3,mu[i],mu[j],mu2[i][k],mu2[j][k])
                    angle_jikl = self.calc_fourpointcorr('jikl',self.fak1,self.fak2,self.fak3,mu[i],mu[j],mu2[i][k],mu2[j][k])

                    for jj,T3 in enumerate(t):
                        for ii,T1 in enumerate(t):
                            R3[ii][jj] += dipole2*angle_jilk * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*(omega2[k]-omega[j])*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                            R6[ii][jj] += dipole2*angle_jikl * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*(omega2[k]-omega[i])*self.unitconvfactor*T3 - 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)

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