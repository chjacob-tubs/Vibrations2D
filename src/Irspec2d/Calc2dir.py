import numpy as np
import math
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
        
        if len(np.asarray(self.freqmat).shape) == 2:
            self.check_symmetry(self.freqmat)
        if len(np.asarray(self.freqmat).shape) == 1:
            self.freqmat = [freqmat]
        
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
            dipolemat = (np.reshape(olddipole,(len(olddipole),len(olddipole),3))).tolist()
        if len(np.asarray(olddipole).shape) == 3:
            dipolemat = olddipole
            
        return dipolemat
        
    def check_input(self):
        '''
        Compares the frequency matrix (n,n) and the transition dipole moment matrix (n,n,3).
        Due to the transition dipole moments being vectors, the length of the first two elements
        are compared. 
        
        '''
        assert np.asarray(self.freqmat).shape[0] == np.asarray(self.dipoles).shape[0], 'First axis of frequency matrix and transition dipole moment matrix do not have the same length.'
        assert np.asarray(self.freqmat).shape[1] == np.asarray(self.dipoles).shape[1], 'Second axis of frequency matrix and transition dipole moment matrix do not have the same length.'
            
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
        a = np.asarray(a)
        
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
                intenmat[i][j]= (LA.norm(self.dipoles[i][j]))**2 * intfactor * (self.freqmat[0][j]-self.freqmat[0][i])
                
        return intenmat.tolist()
    
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

    
class spectra():
    
    
    
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
        return np.concatenate((negspace,posspace)).tolist()
    
    @staticmethod
    def gauss_func(intensity,x,x0,halfwidth):
        '''
        Computes a single value at position x for a 
        1D gaussian type function.
        
        @param intensity: Intensity of a peak
        @type intensity: Float
        
        @param x: x-values 
        @type x: List of floats
        
        @param x0: Position of a peak
        @type x0: Float
        
        @param halfwidth: Parameter to control the width of the peaks
        @type halfwidth: Float
        @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
        @return: Corresponding y-values
        @rtype: List of floats
        
        '''        
        gamma = halfwidth / (2.0*math.sqrt(2.0*math.log(2.0)))
        return (intensity/(2.0 * math.pi * gamma**2)) * np.exp( - (x - x0)**2 / (2*gamma**2) )
    
    @staticmethod
    def gauss2d_func(intensity,x,x0,y,y0,halfwidth):
        '''
        Computes a single value at position x for a 
        1D gaussian type function.
        
        @param intensity: Intensity of a peak
        @type intensity: Float
        
        @param x: x-values 
        @type x: List of floats
        
        @param x0: Position of a peak
        @type x0: Float
        
        @param halfwidth: Parameter to control the width of the peaks
        @type halfwidth: Float
        @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
        @return: Corresponding y-values
        @rtype: List of floats
        
        '''        
        gamma = halfwidth / (2.0*math.sqrt(2.0*math.log(2.0)))
        return (intensity/(2.0 * math.pi * gamma**2)) * np.exp( - ((x - x0)**2 + (y - y0)**2) / (2*gamma**2) )
    
    @staticmethod
    def lorentz_func(intensity,x,x0,halfwidth):
        '''
        Computes a single value at position x for a 
        1D lorentzian type function.
        
        @param intensity: Intensity of a peak
        @type intensity: Float
        
        @param x: x-values 
        @type x: List of floats
        
        @param x0: Position of a peak
        @type x0: Float
        
        @param halfwidth: Parameter to control the width of the peaks
        @type halfwidth: Float
        @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
        @return: Corresponding y-values
        @rtype: List of floats
        
        '''
        return ( (intensity*halfwidth) / (2*math.pi) ) / ( (x-x0)**2 + (halfwidth/2)**2 )
    
    @staticmethod
    def lorentz2d_func(intensity,x,x0,y,y0,halfwidth):
        '''
        Computes a single value at grid x,y for a 
        2D lorentzian type function.
        
        @param intensity: Intensity of a peak
        @type intensity: Float
        
        @param x/y: x-values 
        @type x/y: List of floats
        
        @param x0/y0: Position of a peak
        @type x0/y0: Float
        
        @param halfwidth: Parameter to control the width of the peaks
        @type halfwidth: Float
        @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
        @return: Corresponding y-values
        @rtype: List of floats
        
        '''
        return ( (intensity*halfwidth) / (2*math.pi) ) / ( (x-x0)**2 + (y-y0)**2 + (halfwidth/2)**2 )
    
    @staticmethod
    def get_norm_1d_spectrum(xmin,xmax,freqs,ints,steps=5000,halfwidth=5,ftype='gauss'):
        '''
        Sums up all gauss/lorentz functions for each peak and sets the highest value to one.
        
        @param xmin/xmax: minimum and maximum value of the spectrum
        @type xmin/xmax: Float
        
        @param freqs/ints: frequencies and corresponding intensities
        @type freqs/ints: Lists of floats
        
        @param steps: number of points for the x-axis
        @type steps: Integer
        
        @param halfwidth: Parameter to control the width of the peaks
        @type halfwidth: Float
        @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
        @param ftype: Choses between gauss and lorentz function
        @type ftype: String
        
        @return: x and y values of the 1D plot
        @rtype: Lists of floats
        
        '''
        x = np.linspace(xmin,xmax,steps)
        y = np.zeros(steps)
        
        for freq, inten in zip(freqs,ints):
            if ftype.lower() == 'gauss':
                y += spectra.gauss_func(inten,x,freq,halfwidth)
            if ftype.lower() == 'lorentz':
                y += spectra.lorentz_func(inten,x,freq,halfwidth)

        y = list(map(lambda x : x/y.max(), y))
        return x,y
    
    @staticmethod
    def get_2d_spectrum(xmin,xmax,exc,ble,emi,steps=2000,halfwidth=15,ftype='gauss'):
        '''
        Text
        
        '''
        
        x = np.linspace(xmin, xmax, steps)
        y = np.linspace(xmin, xmax, steps)
        xx, yy = np.meshgrid(x, y)
        
        z = np.zeros((steps,steps))
        
        exc_x,exc_y,exc_i = exc
        emi_x,emi_y,emi_i = emi
        ble_x,ble_y,ble_i = ble
        
        y_vals = []
        y_vals.extend([exc_y,emi_y,ble_y])
        i_vals = []
        i_vals.extend([exc_i,emi_i,ble_i])

        
        for freq_x, freq_y, inten in zip(exc_x+emi_x+ble_x, exc_y+emi_y+ble_y, exc_i+emi_i+ble_i):
            if ftype.lower() == 'gauss':
                z += spectra.gauss2d_func(inten,xx,freq_x,yy,freq_y,halfwidth=5)
            if ftype.lower() == 'lorentz':
                z += spectra.lorentz2d_func(inten,xx,freq_x,yy,freq_y,halfwidth=5)
            
        
        return x.tolist(),y.tolist(),z.tolist()
    
    @staticmethod
    def norm_2d_spectrum(z,max_z):
        '''
        Text
        
        '''        
        for i in range(len(z)):
            for j in range(len(z)):
                z[i][j] = z[i][j] / max_z
        
        return z