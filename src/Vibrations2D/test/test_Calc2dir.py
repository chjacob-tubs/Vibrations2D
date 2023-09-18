import pytest
import numpy as np
import math
from numpy import linalg as LA

import Vibrations2D.Calc2dir as Calc2dir

@pytest.fixture
def Calc2dir_base_2modes():
    '''
    Sets the testing object. 
    Rounded data is from the DAR molecule calculated with a 
    def2-TZVP basis set and B3-LYP functional. 
    '''
    freq = np.load('testdata/VCI_frequencies.npy')
    dips = np.load('testdata/VCI_dipolemoments.npy')
    return Calc2dir.Calc2dir_base(freq,dips)

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
    a = [[0,1,2],[1,2,3],[2,3,4]]
    b = [[0,1,2],[-1,2,3],[-2,-3,4]]
    c = [[1,1,1],[2,2,2],[3,3,3]]
    d = np.ones((1,2,3,4))

    assert Calc2dir_base_2modes.check_symmetry(a) == True
    assert Calc2dir_base_2modes.check_symmetry(b) == True
    assert Calc2dir_base_2modes.check_symmetry(c) == False

    try:
        Calc2dir_base_2modes.check_symmetry(c)
    except ValueError:
        pass 
    except:
        raise AssertionError('The check_symmetry function works even though the matrix shape is not implemented.')


def test_calc_nmodes(Calc2dir_base_2modes):
    '''
    The number of modes equals the length of the frequency matrix in one direction.

    '''
    assert Calc2dir_base_2modes.calc_nmodes() == 6
    # assert Calc2dir_base_3modes.calc_nmodes() == 10

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
    n2 = Calc2dir_base_2modes.calc_nmodes()
    # n3 = Calc2dir_base_3modes.calc_nmodes()
    assert Calc2dir_base_2modes.calc_num_oscill(n2) == 2
    # assert Calc2dir_base_3modes.calc_num_oscill(n3) == 3
    
def test__calc_nmodesexc(Calc2dir_base_2modes):
    '''
    n_modesexc = n_oscill + (n_oscill*(n_oscill-1))/2

    @return: number of excited modes
    @rtype: integer

    '''
    
    assert Calc2dir_base_2modes._calc_nmodesexc() == 3
    
def test_calc_trans2int(Calc2dir_base_2modes):
    '''
    Calculates the intensity matrix from the given transition dipole moment matrix 
    and the given frequeny matrix.

    '''
    pass 

def test_generate_freqmat(Calc2dir_base_2modes):
    '''
    Generates a frequency matrix from given frequency list

    @param freq: list of frequencies
    @type freq: list

    @return: matrix of frequencies
    @rtype: np.array

    '''
    a = [1,2,3]
    b = np.asarray([[ 0,  1,  2],
         [-1,  0,  1],
         [-2, -1,  0]])
    
    c = [1,2,3,4,5]
    d = np.asarray([[ 0,  1,  2,  3,  4],
         [-1,  0,  1,  2,  3],
         [-2, -1,  0,  1,  2],
         [-3, -2, -1,  0,  1],
         [-4, -3, -2, -1,  0]])
    
    np.testing.assert_almost_equal(Calc2dir_base_2modes.generate_freqmat(a),b)
    np.testing.assert_almost_equal(Calc2dir_base_2modes.generate_freqmat(c),d)
    
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
    ZZZZ = 'ZZZZ'
    ZZYY = 'ZZYY'
    ZXXZ = 'ZXXZ'
    
    assert Calc2dir_base_2modes._get_pulse_angles(ZZZZ) == [0,0,0,0]
    assert Calc2dir_base_2modes._get_pulse_angles(ZZYY) == [0,0,90,90]
    assert Calc2dir_base_2modes._get_pulse_angles(ZXXZ) == [0,90,90,0]
    
def test_calc_cos(Calc2dir_base_2modes):
    '''
    calculates the cosine between two three-dimensional vectors

    @param vec1/vec2: two 3D vectors
    @type vec1/vec2: list of three floats 

    @return: angle between the vectors
    @rtype: float

    '''
    
    v1 = [1,1,1]
    v2 = [1,2,3]
    v3 = [0,0,0]
    
    assert Calc2dir_base_2modes.calc_cos(v1,v2) == 0.9258200997725515
    assert Calc2dir_base_2modes.calc_cos(v1,v3) == 0
    assert Calc2dir_base_2modes.calc_cos(v2,v2) == 1
    
def test__calc_fourpoint_factors(Calc2dir_base_2modes):
    assert Calc2dir_base_2modes._calc_fourpoint_factors([0,0,0,0]) == (2, 2, 2)
    assert Calc2dir_base_2modes._calc_fourpoint_factors([0,0,90,90]) == (4, -1, -1)
    assert Calc2dir_base_2modes._calc_fourpoint_factors([0,90,90,0]) == (-1, -1, 4)
    
def test_calc_fourpointcorr_mat():
    pass

def test__get_secexc_dipoles():
    pass
    
def test_n2s(Calc2dir_base_2modes):
    '''
    Takes a number with a decimal point and changes it to an underscore. 
    '''        
    assert Calc2dir_base_2modes.n2s(2.5) == '2_5'
