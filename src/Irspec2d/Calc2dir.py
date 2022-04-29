import numpy as np
from numpy import linalg as LA

class Calc2dir_base():
    '''
    This is supposed to calculate all kinds of different 2D IR spectra.
    
    '''
    
    def __init__(self, freqmat, dipoles):
        '''
        Create settings for object to calculate 2D IR spectra 
        
        @param dipoles: Matrix of transition dipole moments
        @type dipoles: list of lists of numbers
        
        @param freqmat: Matrix of frequencies
        @type freqmat: list of lists of numbers
        
        '''
        
        self.freqmat = freqmat
        self.dipoles = self.read_dipolemat(dipoles)
        
        self.check_input()
        self.check_symmetry(self.freqmat)
        self.check_symmetry(self.dipoles)
        
        self.noscill = self.calc_num_oscill(self.calc_nmodes())
    
    
    def read_dipolemat(self,olddipole):
        '''
        The transition dipole moment matrix that is obtained by VIBRATIONS calculations
        has the shape (n,n,1,3). In order to use it in the following calculations it is
        reduced to the shape (n,n,3). 
        
        n = number of frequencies/transition dipole moments. 
        
        @param olddipole: Given transition dipole moment matrix
        @type olddipole: list of lists of numbers
        
        @return: Transition dipole moment matrix in reduced form
        @rtype: list of lists of numbers
        
        '''
        if len(np.asarray(olddipole).shape) == 4:
            dipolemat = np.reshape(olddipole,(len(olddipole),len(olddipole),3))
        if len(np.asarray(olddipole).shape) == 3:
            dipolemat = olddipole
            
        return dipolemat
        
    def check_input(self):
        '''
        Compares the frequency matrix (n,n) and the transition dipole moment matrix (n,n,3).
        Due to the transition dipole moments being vectors, the length of the first two elements
        are compared. 
        
        '''
        assert self.freqmat.shape[0] == self.dipoles.shape[0], 'First axis of frequency matrix and transition dipole moment matrix do not have the same length.'
        assert self.freqmat.shape[1] == self.dipoles.shape[1], 'Second axis of frequency matrix and transition dipole moment matrix do not have the same length.'
            
    def check_symmetry(self, a, tol=1e-5):
        '''
        Checks if a given matrix a is symmetric or skew-symmetric.
        Returns True/False.
        
        If the matrix is three-dimensional, as in case of the 
        transition dipole moment matrix, the function transposes 
        the matrix with respect to the axes. This leads to 
        leaving the single vectors as vectors. 
        
        @param a: two- or three-dimensional matrix
        @type a: list of lists of numbers

        '''
        if len(a.shape) == 2:
            val = np.all(np.abs(abs(a)-abs(a).T) < tol)
            
        elif len(a.shape) == 3:
            val = np.all(np.abs(abs(a)-np.transpose(abs(a),(1,0,2))) < tol)
            
        else:
            raise ValueError('The shape of the given matrix is not implemented in the check_symmetry function.')
        
        assert val == True, 'Given matrix is not (skew-)symmetrical. Please check!'
        
    def calc_nmodes(self):
        '''
        The number of modes equals the length of the frequency matrix in one direction.
        
        @return: number of modes
        @rtype: integer
        
        '''
        if len(self.dipoles) == len(self.freqmat[0]):
            n = int(len(self.freqmat[0]))
        else:
            raise ValueError('The matrices containing the frequencies and the transition dipole moments do not have the same length.')
            
        return n
    
    def calc_num_oscill(self,n):
        '''
        Calculates the number of oscillators n_oscill based on a 
        given number of modes n. This is based on the assumption 
        that there are 
           n_modes = 1 + 2n_oscill + (n_oscill*(n_oscill-1))/2 
        modes. There is one ground state (0) plus n first excited states 
        plus n second excited states plus (n_oscill*(n_oscill-1))/2 combination 
        states. 
        This leads to 
           n_oscill = 1/2 * (-3 + sqrt(8*n - 1)).
        
        If there are 
           n = 2n_oscill + (n_oscill*(n_oscill-1))/2
        modes, then 
           n_oscill = 1/2 * (-3 + sqrt(8*n + 9)).
           
        @param nmodes: number of modes
        @type nmodes: integer
        
        @return: number of oscillators
        @rtype: integer

        '''
        noscill = (-3. + np.sqrt(8.*n +1.) ) /2.
        
        assert n != 0, 'There are no modes, because nmodes=0.'
        
        if noscill-int(noscill)==0:
            val = int(noscill)
        else:
            new_noscill = (-3. + np.sqrt(8.*n +9.) ) /2.
            if new_noscill-int(new_noscill)==0:
                val = int(new_noscill)
            else:
                raise ValueError("Number of Oscillators couldn't be evaluated.")
                
        return val
    
    def calc_trans2int(self):
        '''
        Calculates the intensity matrix from the given transition dipole moment matrix 
        and the given frequeny matrix.
        
        @return: intensity matrix
        @rtype: numpy.ndarray
        
        '''
        intfactor = 2.5066413842056297 # factor to calculate integral absorption coefficient having  [cm-1]  and  [Debye] ; see Vibrations/Misc.py code
        intenmat = np.zeros_like(self.freqmat)
        
        for i in range(len(intenmat)):
            for j in range(len(intenmat)):
                intenmat[i][j]= (LA.norm(self.dipoles[i][j]))**2 * intfactor * self.freqmat[i][j]
                
        return intenmat
    
    @staticmethod
    def n2s(number):
        '''
        Takes a number with a decimal point and changes it to an underscore. 
        
        @param number: any number
        @type number: float
        
        @return: number without decimal point
        @rtype: string
        '''
        if str(number).find('.') != -1 : 
            val = str(number)[0:str(number).find('.')]+'_'+str(number)[str(number).find('.')+1:]
        else : 
            val = str(number)
            
        return val
    
    @staticmethod
    def set_line_spacing(maximum,number):
        '''
        Use this for matplotlib.pyplot contour plots in order to set the 
        number of lines. 
        Example: plt.contour(x,y,z,set_line_spacing(abs(z.max()),20))
        
        @param maximum: maximum value of the plotted array
        @type maximum: float
        
        @param number: number of plotted contour lines for the positive/negative values
        @type number: int
        
        @return: new values at which the lines are plotted
        @rtype: list

        '''
        firstvalue = maximum/number
        negspace = np.linspace(-maximum,-firstvalue,number)
        posspace = np.linspace(firstvalue,maximum,number)
        return np.concatenate((negspace,posspace))
    
    
# EXCITON MODEL FUNCTIONS

class excitonmodel(Calc2dir_base):
    
    def __init__(self, cmat, dipoles, anharm):
        '''
        
        @param cmat: coupling matrix, which is in the shape of the one-exciton hamiltonian
        @type cmat: list of lists of floats
        '''
        self.anharm = anharm
        
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


    # FREQUENCY DOMAIN FUNCTIONS

class freqdomain(Calc2dir_base):
    
    def __init__(self, freqmat, dipoles):
        
        super().__init__(freqmat, dipoles)
    
    def calc_excitation(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find 
        the excited state absorption processes that occur in an 2D IR
        experiment. 
        
        @param noscill: number of oscillators
        @type noscill: integer
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        exc_x = [] # excitation coords
        exc_y = [] 
        exc_i = [] # intensity

        for i in range(len(intmat)):
            if intmat[0][i] and i<=self.noscill:
                for j in range(len(intmat)):
                    if j>i:

                        y_coor = self.freqmat[0][j]-self.freqmat[0][i]
                        x_coor = self.freqmat[0][i]-self.freqmat[0][0]
                        exc_inten = intmat[i][j]

                        exc_y.append(y_coor)
                        exc_x.append(x_coor)
                        exc_i.append(exc_inten)

                        # print('Excitation from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',exc_inten)
                        
        return (exc_x, exc_y, exc_i)
    
    def calc_stimulatedemission(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the stimulated emission processes that occur in an 2D IR experiment.
        In order to match the experiment the stimulated emission can only 
        happen in transition to the ground state energy level!
        
        @param noscill: number of oscillators
        @type noscill: integer
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        emi_x = [] # stimulated emission coords
        emi_y = [] 
        emi_i = [] # intensity

        for i in range(len(intmat)):
            for j in range(len(intmat)):
                if j==0 and i>j and i<=self.noscill:

                    y_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    x_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    emi_inten = intmat[j][i]

                    emi_y.append(y_coor)
                    emi_x.append(x_coor)
                    emi_i.append(emi_inten)

                    # print('Stimulated emission from energy level',i,'to',j,'at (',x_coor,',',y_coor,') rcm and intensity: ',emi_inten)
        return (emi_x, emi_y, emi_i)

    def calc_bleaching(self,intmat):
        '''
        Takes the energy levels and the intensity matrix in order to find
        the bleaching processes that occur in an 2D IR experiment.
        
        @param noscill: number of oscillators
        @type noscill: integer
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''

        ble_x = [] # excitation coords
        ble_y = [] 
        ble_i = [] # intensity

        for i in range(len(intmat)):
            if intmat[0][i] != 0 and i<=self.noscill:
                
                y_coor = self.freqmat[0][i]-self.freqmat[0][0]
                ble_inten = intmat[0][i]
                
                for j in range(len(intmat)):
                    if intmat[0][j] != 0 and j<=self.noscill:
                        x_coor = self.freqmat[0][j]-self.freqmat[0][0]
                        ble_x.append(x_coor)
                        ble_y.append(y_coor)
                        ble_i.append(ble_inten)
                        
                        # print('Bleaching from energy level 0 to',i,'at (',x_coor,',',y_coor,') rcm and intensity: ',ble_inten)

        return (ble_x, ble_y, ble_i)
                      
    def calc_all_2d_process(self):
        '''
        Calculates all processes that can occur within a
        2D IR experiment from the energy levels and the
        intensity matrix. 
        
        @return: x- and y-coordinates and intensities of all processes
        @rtype: three tuples of lists

        '''
        intmat = self.calc_trans2int()
        
        exc = self.calc_excitation(intmat)
        ste = self.calc_stimulatedemission(intmat)
        ble = self.calc_bleaching(intmat)
        
        return exc, ste, ble
    
    
    
# TIME DOMAIN FUNCTIONS
    
class timedomain(Calc2dir_base):
    
    unitconvfactor = 0.188 # unit conversion factor
    
    def __init__(self, freqmat, dipoles,**params):
        '''
        Setting all parameters needed for the time domain 2D IR spectra calculations.
        
        @param n_t: number of grid points
        @type n_t: integer
        
        @param dt: spacing between grid points
        @type dt: float
        
        @param T2: dephasing time (experimental value)
        @type T2: float
        
        @param t2: time between two laser pulses
        @type t2: float
        
        @param pol: polarization condition
        @type pol: string
        
        @param pol_list: list of angles of polarization condition
        @type pol_list: list of integers
        
        '''
        super().__init__(freqmat, dipoles)
        
        if 'n_t' in params : 
            self.n_t = params.get('n_t')
            print('Set the number of time points (n_t) to',str(self.n_t)+'.')
        else : 
            self.n_t = 128
            print('Set the number of time points (n_t) to',self.n_t,'(default value).')
        
        if 'dt' in params : 
            self.dt = params.get('dt')
            print('Set the time step length (dt) to',self.dt,'ps.')
        else : 
            self.dt = 0.25
            print('Set the time step length (dt) to',self.dt,'ps (default value).')
        
        if 'T2' in params : 
            self.T2 = params.get('T2')
            print('Set the dephasing time (T2) to',self.T2,'ps.')
        else : 
            self.T2 = 2
            print('Set the zeropadded length (T2) to',self.T2,'ps (default value).')
        
        if 't2' in params : 
            self.t2 = params.get('t2')
            print('Set the population time (t2) to',self.t2,'ps.')
        else : 
            self.t2 = 0
            print('Set the population time (t2) to',self.t2,'ps (default value).')
        
        if 'pol' in params :
            self.polarization = params.get('pol')
            print('Set the polarization (pol) to',self.polarization,'.')
        else : 
            self.polarization = 'ZZZZ'
            print('Set the polarization (pol) to',self.polarization,'(default).')
        
        if 'pol_list' in params :
            self.pol_list = params.get('pol_list')
            print('Set the polarization angles (pol_list) to',self.pol_list,'.')
        else : 
            self.pol_list = self._get_pulse_angles(self.polarization)
            print('Set the polarization angles (pol_list) to',self.pol_list,'(calculated default).')
            
        
    def _get_secexc_dipoles(self): 
        '''
        Extracts the matrix for the excited state transition dipole moments.
        
        @return: excited state transition dipole moment
        @rtype: numpy array

        '''
        exc_trans = []

        for i in range(1,len(self.dipoles)):
            if i <= self.noscill:
                transcolumn = []
                for j in range(len(self.dipoles)):
                    if j > self.noscill :
                        transcolumn.append(self.dipoles[i][j])
                exc_trans.append(transcolumn)
                
        return np.asarray(exc_trans)
    
    def _get_omega_off(self):
        '''
        Sets the omega_off value that is used to center the peaks, 
        in order for better computation of the fourier transforms.
        
        @return: median of the first n_oscill (ground state) frequencies
        @rtype: integer
        
        '''
        return round(sum(self.freqmat[0][1:self.noscill+1])/len(self.freqmat[0][1:self.noscill+1]))
    
    def set_omega(self):
        '''
        In the frequency matrix, the first state is 0. 
        The states 1 to n_oscill are the first excited states.
        The states after state n_oscill are the second excited states.
        
        @return: lists of first and second excited states
        @rtype: lists of floats
        
        '''
        omega_init = self.freqmat[0][1:self.noscill+1]
        omega2_init = self.freqmat[0][self.noscill+1:]
        
        omega_off_value = self._get_omega_off()
        
        omega = [i-omega_off_value for i in omega_init]
        omega2 = [i-2*omega_off_value for i in omega2_init]
        
        if self.noscill != len(omega):
            print('The number of first excitation frequencies does not equal the number of oscillators.')
        if self._calc_nmodesexc() != len(omega2):
            print('The number of second excitation frequencies does not equal the number of excited states.')
        
        return omega, omega2
    
    def _get_pulse_angles(self,pol):
        '''
        Returns a list of different angles for different polarization conditions.
        E.g. for the <ZZZZ> polarization condition the list is [0,0,0,0].
        
        @param pol: polarization condition
        @type pol: string containing four symbols
        @example pol: 'ZZZZ'
        
        @return: list of angles for given polarization condition
        @rtype: list of integers
        
        '''
        
        pol_list = [0,0,0,0]
        
        for i, val in enumerate(pol):
            if val == pol[0]:
                pol_list[i] = 0
            if val != pol[0]:
                pol_list[i] = 90
        
        return pol_list
        
    def _calc_nmodesexc(self):
        '''
        n_modesexc = n_oscill + (n_oscill*(n_oscill-1))/2
        
        @return: number of excited modes
        @rtype: integer
        
        '''
        return int(self.noscill + (self.noscill*(self.noscill-1))/2)
    
    def calc_cos(self,vec1,vec2):
        '''
        calculates the cosine between two three-dimensional vectors
        
        @param vec1,vec2: two 3D vectors
        @type vec1,vec2: list of three floats 
        
        @return: angle between the vectors
        @rtype: float

        '''

        mu1 = LA.norm(vec1)
        mu2 = LA.norm(vec2)

        if mu1 != 0 and mu2 !=0:
            cos12 = ( vec1[0]*np.conj(vec2[0])+vec1[1]*np.conj(vec2[1])+vec1[2]*np.conj(vec2[2]) ) / (mu1*mu2)
        else:
            cos12 = 0

        return cos12
    
    def _calc_fourpoint_factors(self,pol_lst):
        '''
        Needs the list of angles of the polarization condition.
        Calculating parts of the four-point correlation function:
        row1 = 4 * cos theta_ab * cos theta_cd - cos theta_ac * cos theta_bd - cos theta_ad * cos theta_bc
        row2 = 4 * cos theta_ac * cos theta_bd - cos theta_ab * cos theta_cd - cos theta_ad * cos theta_bc
        row3 = 4 * cos theta_ad * cos theta_bc - cos theta_ab * cos theta_cd - cos theta_ac * cos theta_bd
        
        @param pol_lst: list of angles for a given polarization condition
        @type pol_lst: list of integers
        
        @return: three faktors for the four-point correlation function
        @rtype: three floats
        
        '''

        ab = np.deg2rad(pol_lst[0]-pol_lst[1])
        cd = np.deg2rad(pol_lst[2]-pol_lst[3])
        ac = np.deg2rad(pol_lst[0]-pol_lst[2])
        bd = np.deg2rad(pol_lst[1]-pol_lst[3])
        ad = np.deg2rad(pol_lst[0]-pol_lst[3])
        bc = np.deg2rad(pol_lst[1]-pol_lst[2])

        row1 = 4 * np.cos(ab) * np.cos(cd) - np.cos(ac) * np.cos(bd) - np.cos(ad) * np.cos(bc)
        row2 = 4 * np.cos(ac) * np.cos(bd) - np.cos(ab) * np.cos(cd) - np.cos(ad) * np.cos(bc)
        row3 = 4 * np.cos(ad) * np.cos(bc) - np.cos(ab) * np.cos(cd) - np.cos(ac) * np.cos(bd)

        return row1,row2,row3
    
    def calc_fourpointcorr(self,pathway,fak1,fak2,fak3,*mus):
        '''
        pathway : 'jjii', 'jiji', 'jiij', 'jikl'

        S = 1/30 * ( cos theta_alpha_beta  * cos theta_gamma_delta * fak1
                   - cos theta_alpha_gamma * cos theta_beta_delta  * fak2
                   - cos theta_alpha_delta * cos theta_beta_gamma  * fak3 )

        @param pathway: feynman pathway of a diagram
        @type pathway: string
        
        @param fak1/fak2/fak3: prefactor of the correlation function
        @type fak1/fak2/fak3: float
        
        @param mus: dipole moments
        @type mus: list of floats
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
    
    def calc_axes(self):
        '''
        Calculates the time axis into a frequency axis.
        
        @return: frequency axis
        @rtype: list of floats
        
        '''
        ticks = []
        n_zp = self.n_t * 2
        for i in range(0,n_zp):
            ticks.append( (i-(n_zp/2))*2*np.pi / (self.unitconvfactor*n_zp*self.dt) + self._get_omega_off() )
        return ticks
    
    def calc_diagrams(self):
        '''
        This computes the diagrams R_1 to R_6.
        R_1, R_2 and R_3 are rephasing diagrams and R_4, R_5 and R_6 are non-rephasing diagrams.
        It also computes angles for different (ZZZZ, ZZXX) polarization functions.
        
        @return: Feynman diagrams 
        @rtype: tuple of numpy arrays
        
        '''
        fak1, fak2, fak3 = self._calc_fourpoint_factors(self.pol_list)
        
        R1 = np.zeros((self.n_t,self.n_t),dtype=np.complex_)
        R2 = np.zeros_like(R1,dtype=np.complex_)
        R3 = np.zeros_like(R1,dtype=np.complex_)
        R4 = np.zeros_like(R1,dtype=np.complex_)
        R5 = np.zeros_like(R1,dtype=np.complex_)
        R6 = np.zeros_like(R1,dtype=np.complex_)
        
        t = np.arange(0,self.n_t*self.dt,self.dt)
        mu, mu2 = self.dipoles[0][1:self.noscill+1] , self._get_secexc_dipoles()
        
        omega, omega2 = self.set_omega()

        for j in range(self.noscill):
            for i in range(self.noscill):

                # print('i:',i,'j:',j)
                
                mui = LA.norm(mu[i])
                muj = LA.norm(mu[j])
                
                dipole = mui**2 * muj**2
                
                angle_jjii = self.calc_fourpointcorr('jjii',fak1,fak2,fak3,mu[i],mu[j])
                angle_jiji = self.calc_fourpointcorr('jiji',fak1,fak2,fak3,mu[i],mu[j])
                angle_jiij = self.calc_fourpointcorr('jiij',fak1,fak2,fak3,mu[i],mu[j])

                # print('mu_i:',mui,'mu_j:',muj)
                # print('dipole:',dipole)

                for jj,T3 in enumerate(t):
                    for ii,T1 in enumerate(t):
                        R1[ii][jj] -= angle_jiji*dipole * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                        R2[ii][jj] -= angle_jjii*dipole * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 - (T1+T3)/self.T2)
                        R4[ii][jj] -= angle_jiij*dipole * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                        R5[ii][jj] -= angle_jjii*dipole * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*omega[i]*self.unitconvfactor*T3 - (T1+T3)/self.T2)

                for k in range(self._calc_nmodesexc()):

                    # print('k:',k)

                    muik = LA.norm(mu2[i][k])
                    mujk = LA.norm(mu2[j][k])
                    dipole2 = mui*muj*muik*mujk

                    # rint('mu_ik:',muik,'mu_jk:',mujk)
                    # print('dipole2:',dipole2)


                    angle_jilk = self.calc_fourpointcorr('jilk',fak1,fak2,fak3,mu[i],mu[j],mu2[i][k],mu2[j][k])
                    angle_jikl = self.calc_fourpointcorr('jikl',fak1,fak2,fak3,mu[i],mu[j],mu2[i][k],mu2[j][k])

                    for jj,T3 in enumerate(t):
                        for ii,T1 in enumerate(t):
                            R3[ii][jj] += dipole2*angle_jilk * np.exp(   1j*omega[j]*self.unitconvfactor*T1 - 1j*(omega2[k]-omega[j])*self.unitconvfactor*T3 + 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)
                            R6[ii][jj] += dipole2*angle_jikl * np.exp( - 1j*omega[j]*self.unitconvfactor*T1 - 1j*(omega2[k]-omega[i])*self.unitconvfactor*T3 - 1j*(omega[j]-omega[i])*self.unitconvfactor*self.t2 - (T1+T3)/self.T2)

                # print()

        return R1,R2,R3,R4,R5,R6
    
    def calc_sum_diagram(self,R_a,R_b,R_c):
        '''
        Text.
        
        '''
        R = R_a + R_b + R_c
        
        for i in range(len(R)):
            R[0][i] = R[0][i]/(2+0j)
            R[i][0] = R[i][0]/(2+0j)
            
        return R
    
    def calc_2d_fft(self,R):
        '''
        Text.
        
        '''
        n_zp = self.n_t * 2
        R_ft = np.fft.ifft2(R,s=(n_zp,n_zp))
        
        return R_ft
    
    def get_absorptive_spectrum(self):
        '''
        Text.
        
        '''
        R1,R2,R3,R4,R5,R6 = self.calc_diagrams()
        
        R_r_ft = self.calc_2d_fft(self.calc_sum_diagram(R1,R2,R3))
        R_nr_ft = self.calc_2d_fft(self.calc_sum_diagram(R4,R5,R6))
        
        R = np.asarray( np.fft.fftshift((np.flipud(np.roll(R_r_ft,-1,axis=0))+R_nr_ft).real,axes=(0,1)) )
        
        axes = self.calc_axes()
        
        return R, axes