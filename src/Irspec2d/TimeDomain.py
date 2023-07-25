import numpy as np
from numpy import linalg as LA
from scipy import fft

from Irspec2d import *

# TIME DOMAIN FUNCTIONS
    
class timedomain(Calc2dir_base):
    
    ucf = 0.188 # unit conversion factor
    
    def __init__(self, freqs : np.ndarray, dipoles : np.ndarray, **params):
        '''
        Setting all parameters needed for the time domain 2D IR spectra calculations.
        
        @param dipoles: Matrix of transition dipole moments
        @type dipoles: list of lists of floats
        
        @param freqmat: Matrix of frequencies
        @type freqmat: list of lists of floats
        
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
        super().__init__(freqs, dipoles)
        
        
        if 'print_output' in params :
            self.print_output = params.get('print_output')
            if self.print_output : print('Prints all output. To suppress printed output use print_output=False.')
        else : 
            self.print_output = True
            if self.print_output : print('Prints all output (default). To suppress printed output use timedomain(freqs,dipoles,print_output=False).')
            
        if 'n_t' in params : 
            self.n_t = params.get('n_t')
            if self.print_output : print('Set the number of time points (n_t) to',str(self.n_t)+'.')
        else : 
            self.n_t = 128
            if self.print_output : print('Set the number of time points (n_t) to',self.n_t,'(default value).')
        
        if 'dt' in params : 
            self.dt = params.get('dt')
            if self.print_output : print('Set the time step length (dt) to',self.dt,'ps.')
        else : 
            self.dt = 0.25
            if self.print_output : print('Set the time step length (dt) to',self.dt,'ps (default value).')
        
        if 'T2' in params : 
            self.T2 = params.get('T2')
            if self.print_output : print('Set the dephasing time (T2) to',self.T2,'ps.')
        else : 
            self.T2 = 2
            if self.print_output : print('Set the dephasing time (T2) to',self.T2,'ps (default value).')
        
        if 't2' in params : 
            self.t2 = params.get('t2')
            if self.print_output : print('Set the population time (t2) to',self.t2,'ps.')
        else : 
            self.t2 = 0
            if self.print_output : print('Set the population time (t2) to',self.t2,'ps (default value).')
        
        if 'pol' in params :
            self.polarization = params.get('pol')
            if self.print_output : print('Set the polarization (pol) to',self.polarization,'.')
        else : 
            self.polarization = 'ZZZZ'
            if self.print_output : print('Set the polarization (pol) to',self.polarization,'(default).')
        
        if 'pol_list' in params :
            self.pol_list = params.get('pol_list')
            if self.print_output : print('Set the polarization angles (pol_list) to',self.pol_list,'.')
        else : 
            self.pol_list = self._get_pulse_angles(self.polarization)
            if self.print_output : print('Set the polarization angles (pol_list) to',self.pol_list,'(calculated default).')
        
        if 'omega_off' in params :
            self.omega_off = params.get('omega_off')
            if self.print_output : print('Set the omega offset value (omega_off) to',self.omega_off,'.')
        else : 
            self.omega_off = self._get_omega_off()
            if self.print_output : print('Set the omega offset value (omega_off) to',self.omega_off,'(calculated default).')
            
        
    def _get_secexc_dipoles(self) -> np.ndarray : 
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
    
    def _get_omega_off(self) -> int :
        '''
        Sets the omega_off value that is used to center the peaks, 
        in order for better computation of the fourier transforms.
        
        @return: median of the first n_oscill (ground state) frequencies
        @rtype: integer
        
        '''
        omega_off = round(sum(self.freqs[1:self.noscill+1])/len(self.freqs[1:self.noscill+1]))
        
        return omega_off
    
    def set_omega(self) -> list :
        '''
        In the frequency matrix, the first state is 0. 
        The states 1 to n_oscill are the first excited states.
        The states after state n_oscill are the second excited states.
        
        @return: lists of first and second excited states
        @rtype: lists of floats
        
        '''
        omega_init = self.freqs[1:self.noscill+1]
        omega2_init = self.freqs[self.noscill+1:]
        
        omega_off_value = self.omega_off
        
        omega = [i-omega_off_value for i in omega_init]
        omega2 = [i-2*omega_off_value for i in omega2_init]
        
        if self.noscill != len(omega):
            print('The number of first excitation frequencies does not equal the number of oscillators.')
        if self._calc_nmodesexc() != len(omega2):
            print('The number of second excitation frequencies does not equal the number of excited states.')
        
        return omega, omega2
    
    def _get_pulse_angles(self, pol : str) -> list :
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
        
    def _calc_nmodesexc(self) -> int :
        '''
        n_modesexc = n_oscill + (n_oscill*(n_oscill-1))/2
        
        @return: number of excited modes
        @rtype: integer
        
        '''
        n_modes_exc = int(self.noscill + (self.noscill*(self.noscill-1))/2)
        
        return n_modes_exc
    
    def calc_cos(self, vec1 : list, vec2 : list) -> float :
        '''
        calculates the cosine between two three-dimensional vectors
        
        @param vec1/vec2: two 3D vectors
        @type vec1/vec2: list of three floats 
        
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
    
    def _calc_fourpoint_factors(self, pol_lst : list) -> float :
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

        return row1, row2, row3
    
    def calc_fourpointcorr(self, pathway : str, fak1 : float, fak2 : float, fak3 : float, *mus) -> float:
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
    
    def calc_fourpointcorr_mat(self, pathway : str, fak1 : float, fak2 : float, fak3 : float, *mus) -> float:
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
    
    def calc_axes(self) -> np.ndarray :
        '''
        Calculates the time axis into a frequency axis.
        
        @return: frequency axis
        @rtype: list of floats
        
        '''
        n_zp = self.n_t * 2
        tks = np.linspace(0, n_zp-1, n_zp)
        ticks = (tks-(n_zp/2))*2*np.pi / (self.ucf*n_zp*self.dt) + self.omega_off
        
        return ticks
    
    def calc_diagrams(self) -> np.ndarray :
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
                
                mui = LA.norm(mu[i])
                muj = LA.norm(mu[j])
                
                dipole = mui**2 * muj**2
                
                f_jjii = self.calc_fourpointcorr('jjii',fak1,fak2,fak3,mu[i],mu[j]) * dipole
                f_jiji = self.calc_fourpointcorr('jiji',fak1,fak2,fak3,mu[i],mu[j]) * dipole
                f_jiij = self.calc_fourpointcorr('jiij',fak1,fak2,fak3,mu[i],mu[j]) * dipole
                
                _t = np.tile(t,(self.n_t,1))
                
                parta    = 1j*omega[j]*self.ucf*_t.T
                partb    = 1j*omega[i]*self.ucf*_t
                partb_R4 = 1j*omega[j]*self.ucf*_t
                partc    = 1j*(omega[j]-omega[i])*self.ucf*self.t2
                partd    = (_t.T+_t)/self.T2 
                
                R1 -= f_jiji * np.exp(   parta - partb    + partc - partd )
                R2 -= f_jjii * np.exp(   parta - partb            - partd )
                R4 -= f_jiij * np.exp( - parta - partb_R4 + partc - partd ) 
                R5 -= f_jjii * np.exp( - parta - partb            - partd ) 

                for k in range(self._calc_nmodesexc()):

                    muik = LA.norm(mu2[i][k])
                    mujk = LA.norm(mu2[j][k])
                    dipole2 = mui*muj*muik*mujk
                    
                    parta2    = 1j*omega[j]*self.ucf*_t.T
                    partb2_R3 = 1j*(omega2[k]-omega[j])*self.ucf*_t
                    partb2_R6 = 1j*(omega2[k]-omega[i])*self.ucf*_t
                    partc2    = 1j*(omega[j]-omega[i])*self.ucf*self.t2
                    partd2    = (_t.T+_t)/self.T2

                    f_jilk = self.calc_fourpointcorr('jilk',fak1,fak2,fak3,mu[i],mu[j],mu2[i][k],mu2[j][k]) * dipole2
                    f_jikl = self.calc_fourpointcorr('jikl',fak1,fak2,fak3,mu[i],mu[j],mu2[i][k],mu2[j][k]) * dipole2
                    
                    R3 += f_jilk * np.exp(   parta2 - partb2_R3 + partc2 - partd2 ) 
                    R6 += f_jikl * np.exp( - parta2 - partb2_R6 - partc2 - partd2 ) 
        
        return R1,R2,R3,R4,R5,R6
    
    def calc_sum_diagram(self, R_a : np.ndarray, R_b : np.ndarray, R_c : np.ndarray) -> np.ndarray :
        '''
        Calculates the sum of diagrams and divides the first row and column by two.
        
        @param R_i, i=a,b,c: Feynman diagrams
        @type R_i, i=a,b,c: numpy arrays
        
        @return: Sum of Feynman diagrams
        @rtype: numpy array
        
        '''
        R = R_a + R_b + R_c
        
        for i in range(len(R)):
            R[0][i] = R[0][i]/(2+0j)
            R[i][0] = R[i][0]/(2+0j)
            
        return R
    
    def calc_2d_fft(self, R : np.ndarray) -> np.ndarray :
        '''
        Calculates a two-dimensional Fourier transformation of a given array
        
        @param R: sum of Feynman diagrams
        @type R: numpy array
        
        @return: Fourier transformed sum of Feynman diagrams
        @rtype: numpy array
        
        '''
        n_zp = self.n_t * 2
        R_ft = fft.ifft2(R,s=(n_zp,n_zp))
        
        return R_ft
    
    def get_absorptive_spectrum(self) -> np.ndarray :
        '''
        Automatically calculates a fully absorption 2D IR spectrum.
        R(w3,t2,w1) = FFT2D ( Real ( R_r(t3,t2,t1)+R_nr(t3,t2,t1) ) )

        @return R: Resulting signal from fourier-transformed sum of Feynman diagrams
        @rtype R: numpy array

        @return axes: frequency axis
        @rtype axes: list of floats

        '''
        R1,R2,R3,R4,R5,R6 = self.calc_diagrams()

        R_r_ft = self.calc_2d_fft(self.calc_sum_diagram(R1,R2,R3))
        R_nr_ft = self.calc_2d_fft(self.calc_sum_diagram(R4,R5,R6))

        R = np.asarray( np.fft.fftshift((np.flipud(np.roll(R_r_ft,-1,axis=0))+R_nr_ft).real,axes=(0,1)) )

        axes = self.calc_axes()

        return R, axes
    
    def get_photon_echo_spectrum(self) -> np.ndarray :
        '''
        Automatically calculates a photon echo 2D IR spectrum.
        R(w3,t2,w1) = abs( FFT2D ( Real ( R_r(t3,t2,t1)) ) )
        
        @return R: Resulting signal from fourier-transformed sum of Feynman diagrams
        @rtype R: numpy array
        
        @return axes: frequency axis
        @rtype axes: list of floats
        
        '''
        R1,R2,R3,R4,R5,R6 = self.calc_diagrams()
        
        R_r_ft = self.calc_2d_fft(self.calc_sum_diagram(R1,R2,R3))
        R_r_ft = np.absolute(R_r_ft)
        
        R = np.asarray( np.fft.fftshift((np.flipud(np.roll(R_r_ft,-1,axis=0))).real,axes=(0,1)) )
        
        axes = self.calc_axes()
        
        return R, axes
    
    def get_correlation_spectrum(self) -> np.ndarray :
        '''
        Automatically calculates a correlation 2D IR spectrum.
        R(w3,t2,w1) = FFT2D ( Imag ( R_r(t3,t2,t1)+R_nr(t3,t2,t1) ) )
        
        @return R: Resulting signal from fourier-transformed sum of Feynman diagrams
        @rtype R: numpy array
        
        @return axes: frequency axis
        @rtype axes: list of floats
        
        '''
        R1,R2,R3,R4,R5,R6 = self.calc_diagrams()
        
        R_r_ft = self.calc_2d_fft(self.calc_sum_diagram(R1,R2,R3))
        R_nr_ft = self.calc_2d_fft(self.calc_sum_diagram(R4,R5,R6))
        
        R = np.asarray( np.fft.fftshift((np.flipud(np.roll(R_r_ft,-1,axis=0))+R_nr_ft).imag,axes=(0,1)) )
        
        axes = self.calc_axes()
        
        return R, axes