import pytest
import numpy as np
import math
from numpy import linalg as LA

import Vibrations2D.Calc2dir as Calc2dir


def test_set_line_spacing():
    '''
    Use this for matplotlib.pyplot contour plots in order to set the
    number of lines.

    Usage: set_line_spacing(maximum,number)
    Example: plt.contour(x,y,z,set_line_spacing(abs(z.max()),20))
    '''
    # Arrange
    expected_out1 = [-3. , -2.4, -1.8, -1.2, -0.6,
                     0.6, 1.2,  1.8, 2.4, 3. ]
    expected_out2 = [-10, -9, -8, -7, -6, -5, -4, -3,
                     -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Act
    output1 = Calc2dir.spectra.set_line_spacing(3,5)
    output2 = Calc2dir.spectra.set_line_spacing(10,10)
    # Assert
    np.testing.assert_almost_equal(output1, expected_out1, decimal=4)
    np.testing.assert_almost_equal(output2, expected_out2, decimal=4)


def test_gauss_func():
    '''
    Computes a single value at position x for a
    1D gaussian type function.

    Usage: gauss_func(x,gamma)
    '''
    # Arrange
    expected_out1 = 0.8824969025845955
    expected_out2 = 0.9902478635182347
    # Act
    output1 = Calc2dir.spectra.gauss_func(5,10)
    output2 = Calc2dir.spectra.gauss_func(7,50)
    # Assert
    np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
    np.testing.assert_almost_equal(output2, expected_out2, decimal=10)


def test_gauss2d_func():
    '''
    Computes a single value at position x for a
    1D gaussian type function.

    Usage: gauss2d_func(x,y,gamma)
    '''
    # Arrange
    x1 = np.linspace(-5, 5, 3)
    x2 = np.linspace(0, 10, 3)
    expected_out1 = [[0.91713079, 0.95766946, 0.91713079],
                     [0.95766946, 1.00000000, 0.95766946],
                     [0.91713079, 0.95766946, 0.91713079]]
    expected_out2 = [[0.60653066, 0.36787944, 0.08208500],
                     [1.00000000, 0.60653066, 0.13533528],
                     [0.60653066, 0.36787944, 0.08208500]]
    # Act
    output1 = Calc2dir.spectra.gauss2d_func(x1, x1, 17)
    output2 = Calc2dir.spectra.gauss2d_func(x1, x2, 5)
    # Assert
    np.testing.assert_almost_equal(output1, expected_out1, decimal=7)
    np.testing.assert_almost_equal(output2, expected_out2, decimal=7)


def test_lorentz_func():
    '''
    Computes a single value at position x for a
    1D lorentzian type function.

    Usage: lorentz_func(x,halfwidth)
    '''
    # Arrange
    expected_out1 = 0.38461538461538464
    expected_out2 = 0.1891891891891892
    # Act
    output1 = Calc2dir.spectra.lorentz_func(1, 5)
    output2 = Calc2dir.spectra.lorentz_func(5, 7)
    # Assert
    np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
    np.testing.assert_almost_equal(output2, expected_out2, decimal=10)


def test_lorentz_func_imag():
    # Arrange
    expected_out1 = 0.07692307692307693
    expected_out2 = 0.13513513513513514
    # Act
    output1 = Calc2dir.spectra.lorentz_func_imag(1,5)
    output2 = Calc2dir.spectra.lorentz_func_imag(5,7)
    # Assert
    np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
    np.testing.assert_almost_equal(output2, expected_out2, decimal=10)


def lorentz2d_func():
    '''
    Computes a single value at grid x,y for a
    2D lorentzian type function.

    Usage: lorentz2d_func(x,y,gamma)
    '''
    # Arrange
    x1 = np.linspace(-5,5,3)
    x2 = np.linspace(0,10,3)
    expected_out1 = [[ 0.0064, 0.0000, -0.0064],
                     [ 0.0000, 0.0000, 0.0000],
                     [-0.0064, 0.0000, 0.0064]]
    expected_out2 = [[ 0., -0.0016, -0.00246154],
                     [ 0., 0.0000,  0.00000000],
                     [ 0., 0.0016,  0.00246154]]

    # Act
    output1 = Calc2dir.spectra.lorentzian2D_imag(x1, x1, 10)
    output2 = Calc2dir.spectra.lorentzian2D_imag(x1, x2, 15)
    # Assert
    np.testing.assert_almost_equal(output1, expected_out1, decimal=10)
    np.testing.assert_almost_equal(output2, expected_out2, decimal=10)


def test_get_1d_spectrum():
    # Arrange
    path = 'testdata/'
    VCI_freq = np.load(path+'VCI_frequencies.npy')
    VCI_ints = np.load(path+'VCI_intensities.npy')
    pathref = 'testdata/reference_data/'
    Ref = np.load(pathref + 'spectra_get_1d_spectrum_ref.npz')
    x_1d_g_ref = Ref['x_1d_g']
    y_1d_g_ref = Ref['y_1d_g']
    # Act
    x_1d_g,y_1d_g = Calc2dir.spectra.get_1d_spectrum(2000,2200,
                                            VCI_freq[0],VCI_ints[0],
                                            ftype='gauss')
    # Assert
    np.testing.assert_almost_equal(x_1d_g, x_1d_g_ref)
    np.testing.assert_almost_equal(y_1d_g, y_1d_g_ref)
