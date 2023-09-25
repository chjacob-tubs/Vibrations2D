"""
Calculation module 2-dimensional IR:
Calc2dir base class and spectra class.
"""
# import math  # imported but unused
import numpy as np
from numpy import linalg as LA


class Calc2dir_base():
    '''
    This is supposed to calculate all kinds of different 2D IR spectra.
    '''

    def __init__(self, freqs: np.ndarray, dipoles: np.ndarray):
        '''
        Create settings for object to calculate 2D IR spectra.
        Checks if input data is a (skew-)symmetrical matrix.
        Calculates the number of
        fundamental frequencies = number of oscillators.

        @param dipoles: Matrix of transition dipole moments
        @type dipoles: list of lists of numbers

        @param freqmat: Matrix of frequencies
        @type freqmat: list of lists of numbers
        '''

        self.dipoles = np.squeeze(dipoles)

        if len(freqs.shape) == 2:
            self.freqs = freqs[0]
        if len(freqs.shape) == 1:
            self.freqs = freqs

        assert self.freqs.shape[0] == self.dipoles.shape[0], (
            'Frequency list and first axis of transition '
            'dipole moment matrix do not have the same length.'
            )
        assert self.check_symmetry(self.dipoles) == True, (
            'Given matrix is not '
            '(skew-)symmetrical. Please check!'
            )

        self.noscill = self.calc_num_oscill(self.calc_nmodes())

    def check_symmetry(self, a: np.ndarray, tol=1e-5):
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
            val = np.all(np.abs(abs(a)-np.transpose(abs(a), (1, 0, 2))) < tol)

        else:
            raise ValueError('The shape', a.shape,
                             'of the given matrix is not implemented in the '
                             'check_symmetry function.')

        return val

    def calc_nmodes(self) -> int:
        '''
        The number of modes equals the length of the
        frequency matrix in one direction.

        @return: number of modes
        @rtype: integer
        '''
        if len(self.dipoles) == len(self.freqs):
            n = int(len(self.freqs))
        else:
            raise ValueError('The matrices containing the frequencies and the '
                             'transition dipole moments do not have the same '
                             'length.')

        return n

    def _calc_nmodesexc(self) -> int:
        '''
        n_modesexc = n_oscill + (n_oscill*(n_oscill-1))/2

        @return: number of excited modes
        @rtype: integer
        '''
        n_modes_exc = int(self.noscill + (self.noscill*(self.noscill-1))/2)

        return n_modes_exc

    def calc_num_oscill(self, n: int) -> int:
        '''
        Calculates the number of oscillators n_oscill based on a
        given number of modes n. This is based on the assumption
        that there are
           n_modes = 1 + 2n_oscill + (n_oscill*(n_oscill-1))/2
        modes. There is one ground state (0) plus n first excited states
        plus n second excited states plus (n_oscill*(n_oscill-1))/2
        combination states.
        This leads to
           n_oscill = 1/2 * (-3 + sqrt(8*n - 1)).

        If there are
           n = 2n_oscill + (n_oscill*(n_oscill-1))/2
        modes, then
           n_oscill = 1/2 * (-3 + sqrt(8*n + 9)).

        @param n: number of modes
        @type n: integer

        @return: number of oscillators
        @rtype: integer
        '''
        noscill = (-3. + np.sqrt(8. * n + 1.)) / 2.

        assert n != 0, 'There are no modes, because nmodes=0.'

        if noscill-int(noscill) == 0:
            n_osc = int(noscill)
        else:
            new_noscill = (-3. + np.sqrt(8. * n + 9.)) / 2.
            if new_noscill-int(new_noscill) == 0:
                n_osc = int(new_noscill)
            else:
                raise ValueError('Number of Oscillators could'
                                 ' not be evaluated.')

        return n_osc

    def calc_trans2int(self) -> np.ndarray:
        '''
        Calculates the intensity matrix from the given
        transition dipole moment matrix and the given frequeny matrix.

        @return: intensity matrix
        @rtype: numpy.ndarray
        '''
        # factor to calculate integral absorption coefficient
        # having [cm-1] and [Debye] ; see Vibrations/Misc.py code
        intfactor = 2.5066413842056297
        dim = self.calc_nmodes()
        freqmat = np.tile(self.freqs, (dim, 1))
        intenmat = (np.linalg.norm(self.dipoles, axis=2)**2
                    * intfactor
                    * (freqmat - freqmat.T))

        return intenmat

    def generate_freqmat(self, freq: np.ndarray) -> np.ndarray:
        '''
        Generates a frequency matrix from given frequency list

        @param freq: list of frequencies
        @type freq: list

        @return: matrix of frequencies
        @rtype: np.array
        '''
        freqlist = np.tile(freq, (len(freq), 1))
        freqmat = freqlist - freqlist.T

        return freqmat

    def _get_pulse_angles(self, pol: str) -> list:
        '''
        Returns a list of different angles for
        different polarization conditions.
        E.g. for the <ZZZZ> polarization condition the list is [0,0,0,0].

        @param pol: polarization condition
        @type pol: string containing four symbols
        @example pol: 'ZZZZ'

        @return: list of angles for given polarization condition
        @rtype: list of integers
        '''

        pol_list = [0, 0, 0, 0]

        for i, val in enumerate(pol):
            if val == pol[0]:
                pol_list[i] = 0
            if val != pol[0]:
                pol_list[i] = 90

        return pol_list

    def calc_cos(self, vec1: list, vec2: list) -> float:
        '''
        calculates the cosine between two three-dimensional vectors

        @param vec1/vec2: two 3D vectors
        @type vec1/vec2: list of three floats

        @return: angle between the vectors
        @rtype: float
        '''

        mu1 = LA.norm(vec1)
        mu2 = LA.norm(vec2)

        if mu1.all() != 0 and mu2.all() != 0:
            cos12 = (np.dot(vec1, np.conj(vec2))) / (mu1*mu2)
        else:
            cos12 = 0

        return cos12

    def _calc_fourpoint_factors(self, pol_lst: list) -> float:
        '''
        Needs the list of angles of the polarization condition.
        Calculating parts of the four-point correlation function:
        row1 = 4 * cos theta_ab * cos theta_cd
               - cos theta_ac * cos theta_bd
               - cos theta_ad * cos theta_bc
        row2 = 4 * cos theta_ac * cos theta_bd
               - cos theta_ab * cos theta_cd
               - cos theta_ad * cos theta_bc
        row3 = 4 * cos theta_ad * cos theta_bc
               - cos theta_ab * cos theta_cd
               - cos theta_ac * cos theta_bd

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

        abcd = np.cos(ab) * np.cos(cd)
        acbd = np.cos(ac) * np.cos(bd)
        adbc = np.cos(ad) * np.cos(bc)

        row1 = 4 * abcd - acbd - adbc
        row2 = 4 * acbd - abcd - adbc
        row3 = 4 * adbc - abcd - acbd

        return row1, row2, row3

    def calc_fourpointcorr_mat(self, fak1: float, fak2: float, fak3: float,
                               mu_a, mu_b, mu_c, mu_d) -> float:
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
        @type mus: list of floats, [x, y, z]
        '''

        S1 = fak1 * self.calc_cos(mu_a, mu_b) * self.calc_cos(mu_c, mu_d)
        S2 = fak2 * self.calc_cos(mu_a, mu_c) * self.calc_cos(mu_b, mu_d)
        S3 = fak3 * self.calc_cos(mu_a, mu_d) * self.calc_cos(mu_b, mu_c)

        S = (S1 + S2 + S3) / 30

        return S

    def _get_secexc_dipoles(self) -> np.ndarray:
        '''
        Extracts the matrix for the excited state transition dipole moments.

        @return: excited state transition dipole moment
        @rtype: numpy array
        '''
        exc_trans = []

        for i in range(1, len(self.dipoles)):
            if i <= self.noscill:
                transcolumn = []
                for j in range(len(self.dipoles)):
                    if j > self.noscill:
                        transcolumn.append(self.dipoles[i][j])
                exc_trans.append(transcolumn)

        return np.asarray(exc_trans)

    @staticmethod
    def n2s(number: float) -> str:
        '''
        Takes a number with a decimal point and changes it to an underscore.

        @param number: any number
        @type number: float

        @return: number without decimal point
        @rtype: string
        '''
        if str(number).find('.') != -1:
            val = (str(number)[0:str(number).find('.')]
                   + '_'
                   + str(number)[str(number).find('.')+1:])
        else:
            val = str(number)

        return val


class spectra():
    '''
    Collection of useful functions for plotting (2D) IR spectra.
    '''

    @staticmethod
    def set_line_spacing(maximum: float, number: int) -> np.ndarray:
        '''
        Use this for matplotlib.pyplot contour plots in order to set the
        number of displayed lines.
        Example: plt.contour(x,y,z,set_line_spacing(abs(z.max()),20))

        @param maximum: maximum value of the plotted array
        @type maximum: float

        @param number: number of plotted contour lines for
                       the positive/negative values
        @type number: int

        @return: new values at which the lines are plotted
        @rtype: np.ndarray

        '''
        firstvalue = maximum/number
        negspace = np.linspace(-maximum, -firstvalue, number)
        posspace = np.linspace(firstvalue, maximum, number)

        return np.concatenate((negspace, posspace))

    @staticmethod
    def gauss_func(x: np.ndarray, gamma: float) -> np.ndarray:
        '''
        Computes a single value at position x for a 1D gaussian type function.

        @param intensity: Intensity of a peak
        @type intensity: Float

        @param x: x-values
        @type x: np.ndarray

        @param x0: Position of a peak
        @type x0: Float

        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak

        @return: Corresponding y-values
        @rtype: np.ndarray
        '''
        f = np.exp(- x**2 / (2*gamma**2))

        return f

    @staticmethod
    def gauss2d_func(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
        '''
        Computes a single value at position x for a 2D gaussian type function.

        @param intensity: Intensity of a peak
        @type intensity: Float

        @param x: x-values
        @type x: np.ndarray

        @param x0: Position of a peak
        @type x0: Float

        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak

        @return: Corresponding y-values
        @rtype: np.ndarray
        '''
        f1 = spectra.gauss_func(x, gamma)
        f2 = spectra.gauss_func(y, gamma)
        f = np.einsum('a,b->ab', f1, f2)

        return f

    @staticmethod
    def lorentz_func(x: np.ndarray, gamma: float) -> np.ndarray:
        '''
        Computes a single value at position x
        for a 1D lorentzian type function.

        @param intensity: Intensity of a peak
        @type intensity: Float

        @param x: x-values
        @type x: np.ndarray

        @param x0: Position of a peak
        @type x0: Float

        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak

        @return: Corresponding y-values
        @rtype: np.ndarray
        '''
        f = ((2 * gamma) / (x**2 + gamma**2))

        return f

    @staticmethod
    def lorentz_func_imag(x: np.ndarray, gamma: float) -> np.ndarray:
        '''
        Computes a single value at position x
        for a 1D lorentzian type function.

        @param intensity: Intensity of a peak
        @type intensity: Float

        @param x: x-values
        @type x: np.ndarray

        @param x0: Position of a peak
        @type x0: Float

        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak

        @return: Corresponding y-values
        @rtype: np.ndarray

        '''
        f = (2 * x) / (x**2 + gamma**2)

        return f

    @staticmethod
    def lorentzian2D(x: np.ndarray, y: np.ndarray,
                     gamma: float) -> np.ndarray:
        '''
        Computes a single value at grid x,y for a 2D lorentzian type function.

        @param intensity: Intensity of a peak
        @type intensity: Float

        @param x/y: x-values
        @type x/y: np.ndarray

        @param x0/y0: Position of a peak
        @type x0/y0: Float

        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak

        @return: Corresponding y-values
        @rtype: np.ndarray

        '''
        f1 = spectra.lorentz_func(x, gamma)
        f2 = spectra.lorentz_func(y, gamma)
        f = np.einsum('a,b->ab', f1, f2)

        return f

    @staticmethod
    def lorentzian2D_imag(x: np.ndarray, y: np.ndarray,
                          gamma: float) -> np.ndarray:
        '''
        Computes a single value at grid x,y for a 2D lorentzian type function.

        @param intensity: Intensity of a peak
        @type intensity: Float

        @param x/y: x-values
        @type x/y: np.ndarray

        @param x0/y0: Position of a peak
        @type x0/y0: Float

        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak

        @return: Corresponding y-values
        @rtype: np.ndarray
        '''
        f1 = spectra.lorentz_func_imag(x, gamma)
        f2 = spectra.lorentz_func_imag(y, gamma)
        f = np.einsum('a,b->ab', f1, f2)

        return f

    @staticmethod
    def get_1d_spectrum(xmin: float, xmax: float,
                        freqs: list, ints: list,
                        steps=5000, gamma=2, ftype='lorentz') -> np.ndarray:
        '''
        Sums up all gauss/lorentz functions for each peak.

        @param xmin/xmax: minimum and maximum value of the spectrum
        @type xmin/xmax: Float

        @param freqs/ints: frequencies and corresponding intensities
        @type freqs/ints: Lists of floats

        @param steps: number of points for the x-axis
        @type steps: Integer

        @param gamma: Parameter to control the width of the peaks
        @type gamma: Float
        @note gamma: Does not necessarily correlate to actual FWHM of a peak

        @param ftype: Choses between gauss and lorentz function
        @type ftype: String

        @return: x and y values of the 1D plot
        @rtype: Lists of floats
        '''
        x = np.linspace(xmin, xmax, steps)
        y = np.zeros(steps)

        for f, i in zip(freqs, ints):
            if ftype.lower() == 'gauss':
                y += i * spectra.gauss_func(x-f, gamma)
            if ftype.lower() == 'lorentz':
                y += i * spectra.lorentz_func(x-f, gamma)

        return x, y
