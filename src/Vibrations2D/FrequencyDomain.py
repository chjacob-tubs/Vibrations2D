import numpy as np
from numpy import linalg as LA
from scipy.constants import c

from Vibrations2D import Calc2dir_base
from Vibrations2D import spectra

# FREQUENCY DOMAIN FUNCTIONS

class frequencydomain(Calc2dir_base):
    
    ucf = c * 10**-10 * 2 * np.pi # unit conversion factor
    # speed of light in cm/ps = 0.0299792458 = c * 10^-10
    
    def __init__(self, freqs, dipoles, **params):
        '''
        Setting all parameters needed for the 
        frequency domain 2D IR spectra calculations.
        
        @param dipoles: Matrix of transition dipole moments
        @type dipoles: list of lists of floats
        
        @param freqmat: Matrix of frequencies
        @type freqmat: list of lists of floats
        
        @param n_grid: number of grid points
        @type n_grid: integer
        
        @param T2: dephasing time (experimental value)
        @type T2: float
        
        @param t2: time between two laser pulses
        @type t2: float
        @note t2: NOT YET IMPLEMENTED
        
        @param pol: polarization condition
        @type pol: string
        @note pol: Only works for <ZZZZ> and <ZZXX> so far
        
        @param pol_list: list of angles of polarization condition
        @type pol_list: list of integers
        
        '''
        super().__init__(freqs, dipoles)
        
        self.freqmat = self.generate_freqmat(self.freqs)
        self.nmodes = self.calc_nmodes()
        
        if 'print_output' in params :
            self.print_output = params.get('print_output')
            if self.print_output : print('Prints all output. '
                                         'To suppress printed output use '
                                         'print_output=False.')
        else : 
            self.print_output = True
            if self.print_output : print('Prints all output (default). '
                                         'To suppress printed output use '
                                         'frequencydomain(freqs,dipoles,'
                                         'print_output=False).')
            
        if 'n_grid' in params : 
            self.n_grid = params.get('n_grid')*2
            if self.print_output : print('Set the number of grid points '
                                         '(n_grid) to',str(self.n_grid)+'.')
        else : 
            self.n_grid = 256
            if self.print_output : print('Set the number of grid points '
                                         '(n_grid) to',
                                         self.n_grid,'(default value).')
        
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
            if self.print_output : print('Set the population time (t2) to',
                                         self.t2,'ps.')
        else : 
            self.t2 = 0
            if self.print_output : print('Set the population time (t2) to',
                                         self.t2,'ps (default value).')
        
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
            
    
    def calc_excitation(self) -> list :
        '''
        Takes the energy levels and the intensity matrix in order to find 
        the excited state absorption processes that occur in an 2D IR
        experiment. 
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        intmat = self.calc_trans2int()
        # excitation coords
        exc_x = [] 
        exc_y = [] 
        # excitation intensities
        exc_i = []

        for i in range(1,self.noscill+1):
            for j in range(self.noscill,len(intmat)):

                x_coor = self.freqmat[0][i]-self.freqmat[0][0]
                y_coor = self.freqmat[0][j]-self.freqmat[0][i]
                exc_inten = intmat[i][j]

                exc_y.append(y_coor)
                exc_x.append(x_coor)
                exc_i.append(exc_inten)

                if self.print_output: print('Excitation from energy level',i,
                                            'to',j,'at (',np.around(x_coor,2),
                                            ',',np.around(y_coor,2),') rcm and '
                                            'intensity: ',np.around(exc_inten,2))
                        
        return (exc_x, exc_y, exc_i)
    
    def calc_stimulatedemission(self) -> list :
        '''
        Takes the energy levels and the intensity matrix in order to find
        the stimulated emission processes that occur in an 2D IR experiment.
        In order to match the experiment the stimulated emission can only 
        happen in transition to the ground state energy level!
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities of transition
        @rtype: tuple of lists

        '''
        intmat = self.calc_trans2int()
        # stimulated emission coords
        emi_x = [] 
        emi_y = [] 
        # intensity
        emi_i = [] 

        for i in range(self.noscill+1):
            for j in range(len(intmat)):
                if j==0 and i>j:

                    x_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    y_coor = self.freqmat[0][i]-self.freqmat[0][j]
                    emi_inten = intmat[i][j]

                    emi_y.append(y_coor)
                    emi_x.append(x_coor)
                    emi_i.append(emi_inten)

                    if self.print_output: print('Stim. emission from energy '
                                                'level',i,'to',j,'at (',
                                                np.around(x_coor,2),',',
                                                np.around(y_coor,2),') rcm and '
                                                'intensity: ',
                                                np.around(emi_inten,2))
        return (emi_x, emi_y, emi_i)

    def calc_bleaching(self) -> list :
        '''
        Takes the energy levels and the intensity matrix in 
        order to find the bleaching processes that occur in 
        an 2D IR experiment.
        
        @param intmat: matrix of intensities
        @type intmat: numpy array or list of lists of numbers
        
        @return: x-coordinates, y-coordinates and intensities 
                 of transition
        @rtype: tuple of lists

        '''
        intmat = self.calc_trans2int()
        # bleaching coords
        ble_x = [] 
        ble_y = [] 
        # intensity
        ble_i = [] 

        for i in range(self.noscill+1):
            for j in range(len(intmat)):
                if i==0 and j>0 and j<=self.noscill:
                    for k in range(1,self.noscill+1):

                        x_coor = self.freqmat[0][k]-self.freqmat[0][i]
                        y_coor = self.freqmat[0][j]-self.freqmat[0][i]
                        ble_inten = -intmat[i][j]

                        ble_x.append(x_coor)
                        ble_y.append(y_coor)
                        ble_i.append(ble_inten)

                        if self.print_output: print('Bleaching from energy '
                                                    'level 0 to',j,'at (',
                                                    np.around(x_coor,2),',',
                                                    np.around(y_coor,2),') rcm '
                                                    'and intensity: ',
                                                    np.around(ble_inten,2))

        return (ble_x, ble_y, ble_i)
                      
    def calc_all_2d_process(self) -> list :
        '''
        Calculates all processes that can occur within a
        2D IR experiment from the energy levels and the
        intensity matrix. 
        
        @return: x- and y-coordinates and intensities of 
                 all processes
        @rtype: three tuples of lists

        '''
        intmat = self.calc_trans2int()
        
        exc_x = [] # excitation coords
        exc_y = [] 
        exc_i = [] # intensity
        
        emi_x = [] # stimulated emission coords
        emi_y = [] 
        emi_i = [] # intensity
        
        ble_x = [] # bleaching coords
        ble_y = [] 
        ble_i = [] # intensity
        
        for i in range(self.noscill+1):
            
            # look for stimulated emission
            if i > 0 :
                s_xy = abs(self.freqmat[i][0])
                s_i = intmat[i][0]

                emi_x.append(s_xy)
                emi_y.append(s_xy)
                emi_i.append(s_i)

                if self.print_output: print('emission',i,0,'  ( '+str(s_xy)
                                            +' | '+str(s_xy)+' )',' :',s_i)
                
                
            for j in range(len(intmat)):

                # look for excitation
                if i>0 and j>i:
                    e_x = self.freqmat[0][i]
                    e_y = self.freqmat[i][j]
                    e_i = intmat[i][j]
                    
                    exc_x.append(e_x)
                    exc_y.append(e_y)
                    exc_i.append(e_i)
                    
                    if self.print_output: print('excitation',i,j,'  ( '+str(e_x)
                                                +' | '+str(e_y)+' )',' :',e_i)

                # look for bleaching
                if i==0 and j>0 and j<=self.noscill:
                    for k in range(1,self.noscill+1):
                        b_x = self.freqmat[i][k]
                        b_y = self.freqmat[i][j]
                        b_i = -intmat[i][j]
                    
                        ble_x.append(b_x)
                        ble_y.append(b_y)
                        ble_i.append(b_i)
                        
                        if self.print_output: print('bleaching',i,j,'  ( '
                                                    +str(b_x)+' | '+str(b_y)
                                                    +' )',' :',b_i)

        exc = (exc_x,exc_y,exc_i)
        ste = (emi_x,emi_y,emi_i)
        ble = (ble_x,ble_y,ble_i)
        
        return exc, ste, ble
    
    def _get_2d_spectrum(self, xmin : float, xmax : float, steps=2000, gamma=5, ftype='lorentz') -> np.ndarray :
        '''
        Plots the simple 2D IR spectrum automatically.
        
        @param xmin/xmax: minimum or maximum value of the spectrum 
                          in both axes
        @type xmin/xmax: Float
        
        @param exc/ble/emi: lists of evaluated x- and y-coordinates and 
                            associated intensities
        @type exc/ble/emi: List of lists of floats
        
        @param steps: number of points for the x-axis
        @type steps: Integer
        
        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak
        
        @param ftype: Choses between gauss and lorentz function
        @type ftype: String
        
        @return: x, y and z values of the 2D plot
        @rtype: Lists of floats
        
        '''
        
        x = np.linspace(xmin, xmax, steps)
        z = np.zeros((steps,steps))
        
        exc, ste, ble = self.calc_all_2d_process()
        
        exc_x,exc_y,exc_i = exc
        emi_x,emi_y,emi_i = ste
        ble_x,ble_y,ble_i = ble

        for freq_x, freq_y, inten in zip(exc_x+emi_x+ble_x, 
                                         exc_y+emi_y+ble_y, 
                                         exc_i+emi_i+ble_i):
            if ftype.lower() == 'gauss':
                z += inten * spectra.gauss2d_func(x-freq_x,x-freq_y,gamma)
            if ftype.lower() == 'lorentz':
                z += inten * spectra.lorentzian2D(x-freq_x,x-freq_y,gamma)
            
        return x, z 
    
    def calculate_S(self, axis):
        '''
        Calculates the signal functions of the 
        ground state bleach, stimulated emission and excited state
        absorption processes. 
        
        '''
        # Get the axis in the correct units
        w = axis * self.ucf
        _w = np.tile(w,(self.n_grid,1))
        
        S1 = np.zeros((self.n_grid,self.n_grid),dtype=np.complex_)
        S2 = np.zeros((self.n_grid,self.n_grid),dtype=np.complex_)
        S3 = np.zeros((self.n_grid,self.n_grid),dtype=np.complex_)
        S4 = np.zeros((self.n_grid,self.n_grid),dtype=np.complex_)
        S5 = np.zeros((self.n_grid,self.n_grid),dtype=np.complex_)
        S6 = np.zeros((self.n_grid,self.n_grid),dtype=np.complex_)
        
        gamma = 1/self.T2
        
        fak1, fak2, fak3 = self._calc_fourpoint_factors(self.pol_list)
        # get the number of doubly excited states and combination bands
        n_exc_oscill = self._calc_nmodesexc() 
        omega, omega2 = self.freqs[1:self.noscill+1]*self.ucf, \
                        self.freqs[self.noscill+1:]*self.ucf
        # get the fundamental transition dipoles mu and the higher excited states mu2
        mu, mu2 = self.dipoles[0][1:self.noscill+1] , self._get_secexc_dipoles() 
        mu_norm = LA.norm(mu, axis=1)
        mu2_norm = LA.norm(mu2, axis=2)
        dipole = np.einsum('a,a,b,b->ab',mu_norm,mu_norm,mu_norm,mu_norm)
        dipole2 = np.einsum('a,b,ac,bc -> abc',mu_norm,mu_norm,
                                               mu2_norm,mu2_norm) 

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
        f_jjii = np.fromfunction( lambda i,j : vfourpoint(j,j,i,i),
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
        
        frac1 = lambda v,i : gamma / ( (v+i)**2 + gamma**2 )
        frac2 = lambda v,i : (v+i) / ( (v+i)**2 + gamma**2 )
        
        # Stimulated Emission
        S1_func = lambda w1,w3,i : (( frac1(w1,+i) + 1j*frac2(w1,+i) ) 
                                    * ( frac1(w3,-i) + 1j*frac2(w3,-i) ))
        S4_func = lambda w1,w3,i : (( frac1(w1,-i) + 1j*frac2(w1,-i) ) 
                                    * ( frac1(w3,-i) + 1j*frac2(w3,-i) ))
        # Ground State Bleaching
        S2_func = lambda w1,w3,i,j : (( frac1(w1,+j) + 1j*frac2(w1,+j) ) 
                                      * ( frac1(w3,-i) + 1j*frac2(w3,-i) ))
        S5_func = lambda w1,w3,i,j : (( frac1(w1,-j) + 1j*frac2(w1,-j) ) 
                                      * ( frac1(w3,-i) + 1j*frac2(w3,-i) ))
        # Excited State Absorption
        S3_func = lambda w1,w3,i,j : (( frac1(w1,+i) + 1j*frac2(w1,+i) ) 
                                      * ( frac1(w3,-j) + 1j*frac2(w3,-j) ))
        S6_func = lambda w1,w3,i,j : (( frac1(w1,-i) + 1j*frac2(w1,-i) ) 
                                    * ( frac1(w3,-j) + 1j*frac2(w3,-j) ))
        
        t2_param = lambda i,j : np.exp(1j * (j-i) * self.t2)

        for i in range(self.noscill): 
            for j in range(self.noscill):
            
                # Stimulated Emission R1+R4
                S1 += -f_jiji[i][j] * S1_func(-_w.T,_w,omega[i]) \
                                    * t2_param(omega[i],omega[j])
                S4 += -f_jiij[i][j] * S4_func( _w.T,_w,omega[i]) \
                                    * t2_param(omega[i],omega[j])
                
                # Ground State Bleach, R2+R5
                S2 += -f_jjii[i][j] * S2_func(-_w.T,_w,omega[i],omega[j])
                S5 += -f_jjii[i][j] * S5_func( _w.T,_w,omega[i],omega[j])
                
                for k in range(n_exc_oscill):
                    # Excited State Absorption, R3+R6
                    S3 += f_ji_kji[i][j][k] \
                          * S3_func(-_w.T,_w,omega[j],(omega2[k]-omega[j])) \
                          * t2_param(omega[i],omega[j])
                    S6 += f_ji_kij[i][j][k] \
                          * S6_func( _w.T,_w,omega[j],(omega2[k]-omega[i])) \
                          * t2_param(omega[j],omega[i])
                    
        return S1, S2, S3, S4, S5, S6

    
    def get_absorptive_spectrum(self, wmin=None, wmax=None) -> np.ndarray :
        '''
        Automatically calculates a fully absorptive 2D IR spectrum.
        
        @param xmin/xmax: minimum or maximum value of the 
                          spectrum in both axes
        @type xmin/xmax: Float
        
        @return: w, S
        @rtype: np.ndarray
        
        '''
        margin = 100
        if not wmin:
            wmin = self.freqs[1:self.noscill+1].min()-margin
        if not wmax:
            wmax = self.freqs[1:self.noscill+1].max()+margin
        
        w = np.linspace(wmin, wmax, self.n_grid)
            
        S1, S2, S3, S4, S5, S6 = self.calculate_S(w)
        S = S1.real + S2.real + S3.real + S4.real + S5.real + S6.real
        
        return S, w

    def get_photon_echo_spectrum(self) -> np.ndarray:
        '''
        Automatically calculates a photon echo 2D IR spectrum.
        S(w3,t2,w1) = abs( Real ( S_r(w3,t2,w1) ) 

        @return R: Resulting signal from fourier-transformed sum
        of Feynman diagrams
        @rtype R: numpy array

        @return axes: frequency axis
        @rtype axes: list of floats
        '''
        margin = 100
        if not wmin:
            wmin = self.freqs[1:self.noscill+1].min()-margin
        if not wmax:
            wmax = self.freqs[1:self.noscill+1].max()+margin
        
        w = np.linspace(wmin, wmax, self.n_grid)
            
        S1, S2, S3, S4, S5, S6 = self.calculate_S(w)
        S = S1.real + S2.real + S3.real 
        
        return abs(S), w