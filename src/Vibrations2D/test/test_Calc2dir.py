import pytest
import numpy as np
# import math  # imported but unused
# from numpy import linalg as LA # imported but unused

import Vibrations2D.Calc2dir as Calc2dir


@pytest.fixture
def freq_dip_input():
    """
    Input data.
    Rounded data is from the DAR molecule calculated with a
    def2-TZVP basis set and B3-LYP functional.
    """
    path = 'testdata/'
    freqs = np.load(path + 'VCI_frequencies.npy')
    dips = np.load(path + 'VCI_dipolemoments.npy')
    return freqs, dips


def attribute_assert_Calc2dir_base(freqs, dipoles, noscill):
    """
    Checks attributes from Calc2dir_base class.
    Reference data from corresponding class.
    Rounded data is from the DAR molecule calculated with a
    def2-TZVP basis set and B3-LYP functional
    """
    # Arrange
    pathref = 'testdata/reference_data/'
    dips_ref = np.load(pathref + 'VCI_dipolemoments_ref.npy')
    freqs_ref = np.load(pathref + 'VCI_frequencies_ref.npy')
    # Assert
    np.testing.assert_almost_equal(freqs_ref, freqs)
    np.testing.assert_almost_equal(dips_ref, dipoles)
    assert 2 == noscill


def test_Calc2dir_base_init(freq_dip_input):
    """tests the initialization of Calc2dir_base."""
    # Arrange
    freqs, dips = freq_dip_input
    # Act
    Calc = Calc2dir.Calc2dir_base(freqs, dips)
    # Assert
    attribute_assert_Calc2dir_base(Calc.freqs, Calc.dipoles, Calc.noscill)


@pytest.fixture
def Calc2dir_base_2modes():
    '''
    Sets the testing object.
    Rounded data is from the DAR molecule calculated with a
    def2-TZVP basis set and B3-LYP functional.
    '''
    freq = np.load('testdata/VCI_frequencies.npy')
    dips = np.load('testdata/VCI_dipolemoments.npy')
    return Calc2dir.Calc2dir_base(freq, dips)


def test_check_symmetry(Calc2dir_base_2modes):
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
    # Arrange
    a = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    b = [[0, 1, 2], [-1, 2, 3], [-2, -3, 4]]
    c = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    # d = np.ones((1, 2, 3, 4))  # assigned but never used

    Calc = Calc2dir_base_2modes
    # Act and Assert
    assert Calc.check_symmetry(a) == True
    assert Calc.check_symmetry(b) == True
    assert Calc.check_symmetry(c) == False
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)
    try:
        Calc2dir_base_2modes.check_symmetry(c)
    except ValueError:
        pass
    except AssertionError:
        raise AssertionError('The check_symmetry function works even though'
                             ' the matrix shape is not implemented.')


def test_calc_nmodes(Calc2dir_base_2modes):
    '''
    The number of modes equals the length
    of the frequency matrix in one direction.
    '''
    # Arrange
    Calc = Calc2dir_base_2modes
    # Act
    nmodes = Calc.calc_nmodes()
    # Assert
    assert nmodes == 6
    # assert Calc2dir_base_3modes.calc_nmodes() == 10
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test_calc_num_oscill(Calc2dir_base_2modes):
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
    '''
    # Arrange
    Calc = Calc2dir_base_2modes
    n2 = Calc.calc_nmodes()  # --> test depends on other method
    # n3 = Calc2dir_base_3modes.calc_nmodes()

    # Act
    num_oscill = Calc.calc_num_oscill(n2)
    # Assert
    assert num_oscill == 2
    # assert Calc2dir_base_3modes.calc_num_oscill(n3) == 3
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test__calc_nmodesexc(Calc2dir_base_2modes):
    '''
    n_modesexc = n_oscill + (n_oscill*(n_oscill-1))/2

    @return: number of excited modes
    @rtype: integer
    '''
    # Arrange
    Calc = Calc2dir_base_2modes
    # Act
    base_2modes = Calc._calc_nmodesexc()
    # Assert
    assert base_2modes == 3
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test_calc_trans2int(Calc2dir_base_2modes):
    '''
    Calculates the intensity matrix from the given transition dipole
    moment matrix and the given frequeny matrix.
    '''
    # Arrange
    path_ref = 'testdata/reference_data/'
    intenmat_ref = np.load(path_ref + 'Calc2dir_calc_trans2int_ref.npy')
    Calc = Calc2dir_base_2modes
    # Act
    intenmat = Calc.calc_trans2int()
    # Assert
    np.testing.assert_almost_equal(intenmat, intenmat_ref)
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test_generate_freqmat(Calc2dir_base_2modes):
    '''
    Generates a frequency matrix from given frequency list

    @param freq: list of frequencies
    @type freq: list

    @return: matrix of frequencies
    @rtype: np.array
    '''
    # Arrange
    a = [1, 2, 3]
    b = np.asarray([[0,  1,  2],
                    [-1,  0,  1],
                    [-2, -1,  0]])

    c = [1, 2, 3, 4, 5]
    d = np.asarray([[0,  1,  2,  3,  4],
                   [-1,  0,  1,  2,  3],
                   [-2, -1,  0,  1,  2],
                   [-3, -2, -1,  0,  1],
                   [-4, -3, -2, -1,  0]])
    Calc = Calc2dir_base_2modes
    # Act
    freqmat_a = Calc.generate_freqmat(a)
    freqmat_c = Calc.generate_freqmat(c)
    # Assert
    np.testing.assert_almost_equal(freqmat_a, b)
    np.testing.assert_almost_equal(freqmat_c, d)
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test__get_pulse_angles(Calc2dir_base_2modes):
    '''
    Returns a list of different angles for different polarization conditions.
    E.g. for the <ZZZZ> polarization condition the list is [0,0,0,0].

    @param pol: polarization condition
    @type pol: string containing four symbols
    @example pol: 'ZZZZ'

    @return: list of angles for given polarization condition
    @rtype: list of integers
    '''
    # Arrange
    ZZZZ = 'ZZZZ'
    ZZYY = 'ZZYY'
    ZXXZ = 'ZXXZ'
    Calc = Calc2dir_base_2modes
    # Act
    pulse_anglesZZZZ = Calc._get_pulse_angles(ZZZZ)
    pulse_anglesZZYY = Calc._get_pulse_angles(ZZYY)
    pulse_anglesZXXZ = Calc._get_pulse_angles(ZXXZ)
    # Assert
    assert pulse_anglesZZZZ == [0, 0, 0, 0]
    assert pulse_anglesZZYY == [0, 0, 90, 90]
    assert pulse_anglesZXXZ == [0, 90, 90, 0]
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test_calc_cos(Calc2dir_base_2modes):
    '''
    calculates the cosine between two three-dimensional vectors

    @param vec1/vec2: two 3D vectors
    @type vec1/vec2: list of three floats

    @return: angle between the vectors
    @rtype: float
    '''
    # Arrange
    v1 = [1, 1, 1]
    v2 = [1, 2, 3]
    v3 = [0, 0, 0]
    Calc = Calc2dir_base_2modes
    # Act
    cos_v1_v2 = Calc.calc_cos(v1, v2)
    cos_v1_v3 = Calc.calc_cos(v1, v3)
    cos_v2_v2 = Calc.calc_cos(v2, v2)
    # Assert
    assert cos_v1_v2 == 0.9258200997725515
    assert cos_v1_v3 == 0
    assert cos_v2_v2 == 1
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test__calc_fourpoint_factors(Calc2dir_base_2modes):
    '''
    calculates the four point factors.
    For more details see corresponding docstrings in src/.
    '''
    # Arrange
    Calc = Calc2dir_base_2modes
    # Act
    fourpoint_factors_1 = Calc._calc_fourpoint_factors([0, 0, 0, 0])
    fourpoint_factors_2 = Calc._calc_fourpoint_factors([0, 0, 90, 90])
    fourpoint_factors_3 = Calc._calc_fourpoint_factors([0, 90, 90, 0])
    # Assert
    assert fourpoint_factors_1 == (2, 2, 2)
    assert fourpoint_factors_2 == (4, -1, -1)
    assert fourpoint_factors_3 == (-1, -1, 4)
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test_calc_fourpointcorr_mat(Calc2dir_base_2modes):
    '''
    calculates the four point correlation matrix.
    For more details see corresponding docstrings in src/
    '''
    # Arrange
    fak1 = 1
    fak2 = 1
    fak3 = 1
    mu_a = 1
    mu_b = 1
    mu_c = 1
    mu_d = 1
    Calc = Calc2dir_base_2modes
    # Act
    S = Calc.calc_fourpointcorr_mat(fak1, fak2, fak3, mu_a, mu_b, mu_c, mu_d)
    # Assert
    assert S == 0.1
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test__get_secexc_dipoles(Calc2dir_base_2modes):
    '''
    Extracts the matrix for the excited transition dipole moments.
    For more details see corresponding docstrings in src/.
    '''
    # Arrange
    path_ref = 'testdata/reference_data/'
    secexc_dipoles_ref = np.load(path_ref + 'Calc2dir_secexc_dipoles_ref.npy')
    Calc = Calc2dir_base_2modes
    # Act
    secexc_dipoles = Calc._get_secexc_dipoles()
    # Assert
    np.testing.assert_almost_equal(secexc_dipoles, secexc_dipoles_ref)
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)


def test_n2s(Calc2dir_base_2modes):
    '''
    Takes a number with a decimal point and changes it to an underscore.
    For more details see corresponding docstrings in src/.
    '''
    # Arrange
    Calc = Calc2dir_base_2modes
    # Act
    n2s = Calc.n2s(2.5)
    # Assert
    assert n2s == '2_5'
    attribute_assert_Calc2dir_base(Calc.freqs,
                                   Calc.dipoles,
                                   Calc.noscill)
