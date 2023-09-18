import numpy as np
import multiprocessing as mp
from numpy import linalg as LA
from scipy.constants import c
from scipy import fft

from Vibrations2D import *

# TIME DOMAIN FUNCTIONS
    
class timedomain(Calc2dir_base):
    
    ucf = c * 10**-10 * 2 * np.pi # unit conversion factor
    # speed of light in cm/ps = 0.0299792458 = c * 10^-10
    # accounts for the conversion of frequency axis from Hz to cm^-1.
    
    def __init__(self, freqs : np.ndarray, dipoles : np.ndarray, **params):
        '''
        Setting all parameters needed for the 
        time domain 2D IR spectra calculations.
        
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
            if self.print_output : print('Prints all output. '
                                         'To suppress printed output '
                                         'use print_output=False. ')
        else : 
            self.print_output = True
            if self.print_output : print('Prints all output (default). '
                                         'To suppress printed output use ' 
                                         'timedomain(freqs,dipoles, '
                                         'print_output=False).')
            
        if 'n_t' in params : 
            self.n_t = params.get('n_t')
            if self.print_output : print('Set the number of time points '
                                         '(n_t) to',str(self.n_t)+'.')
        else : 
            self.n_t = 128
            if self.print_output : print('Set the number of time points '
                                         '(n_t) to',self.n_t,'(default value).')
        
        if 'dt' in params : 
            self.dt = params.get('dt')
            if self.print_output : print('Set the time step length (dt) '
                                         'to',self.dt,'ps.')
        else : 
            self.dt = 0.25
            if self.print_output : print('Set the time step length (dt) '
                                         'to',self.dt,'ps (default value).')
        
        if 'T2' in params : 
            self.T2 = params.get('T2')
            if self.print_output : print('Set the dephasing time (T2) '
                                         'to',self.T2,'ps.')
        else : 
            self.T2 = 2
            if self.print_output : print('Set the dephasing time (T2) '
                                         'to',self.T2,'ps (default value).')
        
        if 't2' in params : 
            self.t2 = params.get('t2')
            if self.print_output : print('Set the population time (t2) '
                                         'to',self.t2,'ps.')
        else : 
            self.t2 = 0
            if self.print_output : print('Set the population time (t2) '
                                         'to',self.t2,'ps (default value).')
        
        if 'pol' in params :
            self.polarization = params.get('pol')
            if self.print_output : print('Set the polarization (pol) '
                                         'to',self.polarization,'.')
        else : 
            self.polarization = 'ZZZZ'
            if self.print_output : print('Set the polarization (pol) '
                                         'to',self.polarization,'(default).')
        
        if 'pol_list' in params :
            self.pol_list = params.get('pol_list')
            if self.print_output : print('Set the polarization angles '
                                         '(pol_list) to',self.pol_list,'.')
        else : 
            self.pol_list = self._get_pulse_angles(self.polarization)
            if self.print_output : print('Set the polarization angles '
                                         '(pol_list) to',self.pol_list,
                                         '(calculated default).')
        
        if 'omega_off' in params :
            self.omega_off = params.get('omega_off')
            if self.print_output : print('Set the omega offset value '
                                         '(omega_off) to',self.omega_off,'.')
        else : 
            self.omega_off = self._get_omega_off()
            if self.print_output : print('Set the omega offset value '
                                         '(omega_off) to',self.omega_off,
                                         '(calculated default).')
            
        
    
    def _get_omega_off(self) -> int :
        '''
        Sets the omega_off value that is used to center the peaks, 
        in order for better computation of the fourier transforms.
        
        @return: median of the first n_oscill (ground state) frequencies
        @rtype: integer
        
        '''
        # only consider ground state frequencies
        freqlist = self.freqs[1:self.noscill+1]
        # get the sum of the frequencies
        numerator = sum(freqlist)
        # get the number of elements in that list
        # this should be equal to the number of oscillators
        denominator = len(freqlist)
        
        omega_off = round(numerator/denominator)
        
        return omega_off
    
    def set_omega(self) -> np.ndarray :
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
            print('The number of first excitation frequencies does not equal '
                  'the number of oscillators.')
        if self._calc_nmodesexc() != len(omega2):
            print('The number of second excitation frequencies does not equal '
                  'the number of excited states.')
        
        return np.asarray(omega), np.asarray(omega2)
    
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
    
    def calc_axes(self) -> np.ndarray :
        '''
        Calculates the time axis into a frequency axis.
        
        @return: frequency axis
        @rtype: list of floats
        
        '''
        n_zp = self.n_t * 2
        tks = np.linspace(0, n_zp-1, n_zp)
        
        numerator = (tks-(n_zp/2))*2*np.pi
        denominator = (self.ucf*n_zp*self.dt)
        
        ticks = numerator/denominator + self.omega_off
        
        return ticks
    
    def calc_diagrams_slow(self) -> np.ndarray :
        '''
        This computes the diagrams R_1 to R_6.
        R_1, R_2 and R_3 are rephasing diagrams and R_4, R_5 and R_6 are non-rephasing diagrams.
        It also computes angles for different (ZZZZ, ZZXX) polarization functions.
        
        @return: Feynman diagrams 
        @rtype: tuple of numpy arrays
        
        '''
        n_exc_oscill = self._calc_nmodesexc() # get the number of doubly excited states and combination bands
        fak1, fak2, fak3 = self._calc_fourpoint_factors(self.pol_list)
        
        R1 = np.zeros((self.n_t,self.n_t),dtype=np.complex_)
        R2 = np.zeros_like(R1,dtype=np.complex_)
        R3 = np.zeros_like(R1,dtype=np.complex_)
        R4 = np.zeros_like(R1,dtype=np.complex_)
        R5 = np.zeros_like(R1,dtype=np.complex_)
        R6 = np.zeros_like(R1,dtype=np.complex_)
        
        t = np.arange(0,self.n_t*self.dt,self.dt)
        _t = np.tile(t,(self.n_t,1))
        partd    = (_t.T+_t)/self.T2
        
        mu, mu2 = self.dipoles[0][1:self.noscill+1] , self._get_secexc_dipoles()
        omega, omega2 = self.set_omega()
        
        for j in range(self.noscill):
            
            muj = LA.norm(mu[j])
            parta    = 1j*omega[j]*self.ucf*_t.T
            partb_R4 = 1j*omega[j]*self.ucf*_t
            
            for i in range(self.noscill):
                
                mui = LA.norm(mu[i])
                dipole = mui**2 * muj**2
                
                f_jjii = self.calc_fourpointcorr('jjii',fak1,fak2,fak3,mu[j],mu[i]) * dipole
                f_jiji = self.calc_fourpointcorr('jiji',fak1,fak2,fak3,mu[j],mu[i]) * dipole
                f_jiij = self.calc_fourpointcorr('jiij',fak1,fak2,fak3,mu[j],mu[i]) * dipole
                
                parta_R1 = 1j*omega[i]*self.ucf*_t.T
                partb    = 1j*omega[i]*self.ucf*_t
                partc    = 1j*(omega[j]-omega[i])*self.ucf*self.t2
                 
                R1 -= f_jiji * np.exp(   parta_R1 - partb    + partc - partd ) # SE
                R4 -= f_jiij * np.exp( - parta    - partb_R4 - partc - partd ) # SE
                R2 -= f_jjii * np.exp(   parta    - partb            - partd ) # GB
                R5 -= f_jjii * np.exp( - parta    - partb            - partd ) # GB

                for k in range(n_exc_oscill):

                    muik = LA.norm(mu2[i][k])
                    mujk = LA.norm(mu2[j][k])
                    dipole2 = mui*muj*muik*mujk
                    
                    partb2_R3 = 1j*(omega2[k]-omega[j])*self.ucf*_t
                    partb2_R6 = 1j*(omega2[k]-omega[i])*self.ucf*_t

                    f_jilk = self.calc_fourpointcorr('jilk',fak1,fak2,fak3,mu[i],mu[j],mu2[i][k],mu2[j][k]) * dipole2 # EA
                    f_jikl = self.calc_fourpointcorr('jikl',fak1,fak2,fak3,mu[i],mu[j],mu2[i][k],mu2[j][k]) * dipole2 # EA
                    
                    R3 += f_jilk * np.exp(   parta - partb2_R3 + partc - partd ) 
                    R6 += f_jikl * np.exp( - parta - partb2_R6 - partc - partd ) 
        
        return R1, R2, R3, R4, R5, R6
    
    def calc_diagrams(self):
        '''
        This computes the diagrams R_1 to R_6.
        R_1, R_2 and R_3 are rephasing diagrams and 
        R_4, R_5 and R_6 are non-rephasing diagrams.
        It also computes angles for different polarization conditions.
        (i.e. <ZZZZ>, <ZZXX>, <ZXZX>, <ZXXZ>)
        
        @return: Feynman diagrams 
        @rtype: numpy arrays
        
        '''
        # define the time axis t
        t = np.arange(0,self.n_t*self.dt,self.dt)
        # rewrite t as a matrix
        _t = np.tile(t,(self.n_t,1)) 
        # get the number of doubly excited states and combination bands
        n_exc_oscill = self._calc_nmodesexc() 
        fak1, fak2, fak3 = self._calc_fourpoint_factors(self.pol_list)
        
        # get the fundamental transitions omega and 
        # the higher excited states omega2
        omega, omega2 = self.set_omega() 
        # get the fundamental transition dipoles mu and 
        # the higher excited states mu2
        mu, mu2 = self.dipoles[0][1:self.noscill+1] , self._get_secexc_dipoles() 
        
        mu_norm = LA.norm(mu, axis=1)
        mu2_norm = LA.norm(mu2, axis=2)
        
        dipole = np.einsum('a,a,b,b->ab',
                           mu_norm, mu_norm,
                           mu_norm, mu_norm)
        dipole2 = np.einsum('a,b,ac,bc -> abc',
                            mu_norm, mu_norm,
                            mu2_norm, mu2_norm) 
        
        # Calculate the exponential parts of the diagrams R1 to R6
        # part A/B : 1j * omega[i/j] * ucf * t1/t3
        AB = np.einsum('a,bc->acb', 1j*omega*self.ucf, _t)
        BA = np.einsum('a,bc->abc', 1j*omega*self.ucf, _t)
        # part C : 1j * ( omega[j] - omega[i] ) * ucf * t2
        C = np.fromfunction(lambda i,j : 1j*(omega[j]-omega[i])
                            *self.ucf*self.t2, 
                            (self.noscill,self.noscill), dtype=int)
        # part D : ( t1 + t3 ) / T2
        D = (_t.T+_t)/self.T2 
        # part B2 : 1j * ( omega[k] - omega[i/j] ) * ucf * t3
        _B2 = np.fromfunction(lambda i,k : 1j*(omega2[k]-omega[i])
                              *self.ucf, 
                              (self.noscill,n_exc_oscill), dtype=int)
        B2 = np.einsum('ab,cd->abcd',_B2,_t)
        
        # Vectorize the four-point correlation functions in order to be able to 
        # calculate them on a grid using np.fromfunction
        vfourpoint = np.vectorize(lambda i,j,k,l : 
                                  self.calc_fourpointcorr_mat(fak1, fak2, fak3, 
                                                              mu[i], mu[j], 
                                                              mu[k], mu[l]),
                                  excluded=['fak1','fak2','fak3'])
        vfourpoint2 = np.vectorize(lambda i,j,k,l,m : 
                                   self.calc_fourpointcorr_mat(fak1, fak2, fak3, 
                                                               mu[i], mu[j], 
                                                               mu2[l][k], 
                                                               mu2[m][k]),
                                   excluded=['fak1','fak2','fak3'])
        
        # calculate the prefactors of the diagrams R1, R2 and R4, R5
        f_jjii = np.fromfunction(lambda i,j : vfourpoint(j,j,i,i),
                                 (self.noscill,self.noscill),dtype=int) * dipole
        f_jiji = np.fromfunction( lambda i,j : vfourpoint(j,i,j,i), 
                                 (self.noscill,self.noscill),dtype=int) * dipole
        f_jiij = np.fromfunction( lambda i,j : vfourpoint(j,i,i,j), 
                                 (self.noscill,self.noscill),dtype=int) * dipole
        
        # calculate the prefactors of the diagrams R3 and R6
        f_ji_kij = np.fromfunction( lambda i,j,k : vfourpoint2(i,j,k,i,j), 
                                   (self.noscill,self.noscill,n_exc_oscill), 
                                   dtype=int ) * dipole2
        f_ji_kji = np.fromfunction( lambda i,j,k : vfourpoint2(i,j,k,j,i), 
                                   (self.noscill,self.noscill,n_exc_oscill), 
                                   dtype=int ) * dipole2

        # Calculate the diagrams R
        R1 = np.einsum('ji,iab,iba,ij,ab -> ab', -f_jiji, 
                       np.exp(+AB), np.exp(-AB), np.exp(+C), np.exp(-D))
        R2 = np.einsum('ji,jab,iba,ab -> ab', -f_jjii, 
                       np.exp(+AB), np.exp(-AB), np.exp(-D))
        R3 = np.einsum('jik,jab,jkab,ij,ab -> ab', f_ji_kij, 
                       np.exp(+AB), np.exp(-B2), np.exp(+C), np.exp(-D))
        
        R4 = np.einsum('ji,jab,jba,ij,ab -> ab', -f_jiij, 
                       np.exp(-AB), np.exp(-AB), np.exp(-C), np.exp(-D))
        R5 = np.einsum('ji,jab,iba,ab -> ab', -f_jjii, 
                       np.exp(-AB), np.exp(-AB), np.exp(-D))
        R6 = np.einsum('jik,jab,ikab,ij,ab -> ab', f_ji_kji, 
                       np.exp(-AB), np.exp(-B2), np.exp(-C), np.exp(-D))
                    
        return R1, R2, R3, R4, R5, R6
    
    def calc_sum_diagram(self, R_a : np.ndarray, R_b : np.ndarray, R_c : np.ndarray) -> np.ndarray :
        '''
        Calculates the sum of diagrams and divides the 
        first row and column by two.
        
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
        Calculates a two-dimensional Fourier transformation 
        of a given array
        
        @param R: sum of Feynman diagrams
        @type R: numpy array
        
        @return: Fourier transformed sum of Feynman diagrams
        @rtype: numpy array
        
        '''
        n_zp = self.n_t * 2
        # prefactor dt^2 due to norm of the ifft2 function
        R_ft = self.dt**2 * fft.ifft2(R,s=(n_zp,n_zp),norm='forward') 
        
        return R_ft
    
    def get_absorptive_spectrum(self, speed=None) -> np.ndarray :
        '''
        Automatically calculates a fully absorption 2D IR spectrum.
        R(w3,t2,w1) = FFT2D ( Real ( R_r(t3,t2,t1)+R_nr(t3,t2,t1) ) )

        @return R: Resulting signal from fourier-transformed sum 
        of Feynman diagrams
        @rtype R: numpy array

        @return axes: frequency axis
        @rtype axes: list of floats

        '''
        # Calculate all diagrams
        if speed:
            R1,R2,R3,R4,R5,R6 = self.calc_diagrams()
        else:
            R1,R2,R3,R4,R5,R6 = self.calc_diagrams_slow()

        # Fourier-transform the sum of the rephasing diagrams
        R_r_ft = self.calc_2d_fft(self.calc_sum_diagram(R1,R2,R3))
        # Fourier-transform the sum of the non-rephasing diagrams
        R_nr_ft = self.calc_2d_fft(self.calc_sum_diagram(R4,R5,R6))
        
        # Flip the axis of the rephasing diagrams
        R_r_flip = np.flipud(np.roll(R_r_ft,-1,axis=0))
        
        R = np.fft.fftshift((R_r_flip+R_nr_ft).real,axes=(0,1)) 

        axes = self.calc_axes()

        return R, axes
    
    def get_photon_echo_spectrum(self) -> np.ndarray :
        '''
        Automatically calculates a photon echo 2D IR spectrum.
        R(w3,t2,w1) = abs( FFT2D ( Real ( R_r(t3,t2,t1)) ) )
        
        @return R: Resulting signal from fourier-transformed sum 
        of Feynman diagrams
        @rtype R: numpy array
        
        @return axes: frequency axis
        @rtype axes: list of floats
        
        '''
        # Calculate all diagrams
        R1,R2,R3,R4,R5,R6 = self.calc_diagrams()
        
        # Fourier-transform the sum of the rephasing diagrams
        R_r_ft = self.calc_2d_fft(self.calc_sum_diagram(R1,R2,R3))
        # Get the absolut values
        R_r_ft = np.absolute(R_r_ft)
        
        # Flip the axis of the rephasing diagrams
        R_r_flip = np.flipud(np.roll(R_r_ft, -1, axis=0))
        
        R = np.fft.fftshift(R_r_flip.real, axes=(0,1)) 
        
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
        # Calculate all diagrams
        R1,R2,R3,R4,R5,R6 = self.calc_diagrams()

        # Fourier-transform the sum of the rephasing diagrams
        R_r_ft = self.calc_2d_fft(self.calc_sum_diagram(R1,R2,R3))
        # Fourier-transform the sum of the non-rephasing diagrams
        R_nr_ft = self.calc_2d_fft(self.calc_sum_diagram(R4,R5,R6))
        
        # Flip the axis of the rephasing diagrams
        R_r_flip = np.flipud(np.roll(R_r_ft,-1,axis=0))
        
        R = np.fft.fftshift((R_r_flip+R_nr_ft).imag,axes=(0,1)) 

        axes = self.calc_axes()
        
        return R, axes
    
    
    
    