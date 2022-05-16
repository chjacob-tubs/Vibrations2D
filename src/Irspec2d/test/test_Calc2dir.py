import pytest
import numpy as np
import math
from numpy import linalg as LA

import Irspec2d.Calc2dir as Calc2dir

class TestCalc2dir_base():
    '''
    This is supposed to test the calculation of all kinds of different 2D IR spectra.
    
    '''
    
    @pytest.fixture
    def freqmat_2modes(self):
        '''
        Sets the testing frequency matrix. 
        Rounded data is from the DAR molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        freq =  [[0,      1765,  1767, 3526, 3526, 3532],
                 [-1765,     0,     3, 1761, 1762, 1767],
                 [-1767,    -3,     0, 1758, 1759, 1765],
                 [-3526, -1761, -1758,    0,    1,    6],
                 [-3526, -1762, -1759,   -1,    0,    6],
                 [-3532, -1767, -1765,   -6,   -6,    0]]
        return freq
    
    @pytest.fixture
    def freqmat_3modes(self):
        '''
        Sets the testing frequency matrix. 
        Rounded data is from the GAAP molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        freq =  [[    0.,  1727.,  1739.,  1779.,  3449.,  3459.,  3476.,  3506.,  3518.,  3547.],
                 [-1727.,     0.,    12.,    52.,  1722.,  1732.,  1749.,  1779.,  1791.,  1820.],
                 [-1739.,   -12.,     0.,    40.,  1710.,  1719.,  1736.,  1767.,  1778.,  1808.],
                 [-1779.,   -52.,   -40.,     0.,  1670.,  1679.,  1696.,  1727.,  1738.,  1768.],
                 [-3449., -1722., -1710., -1670.,     0.,     9.,    26.,    57.,    68.,    98.],
                 [-3459., -1732., -1719., -1679.,    -9.,     0.,    17.,    48.,    59.,    88.],
                 [-3476., -1749., -1736., -1696.,   -26.,   -17.,     0.,    30.,    42.,    71.],
                 [-3506., -1779., -1767., -1727.,   -57.,   -48.,   -30.,     0.,    11.,    41.],
                 [-3518., -1791., -1778., -1738.,   -68.,   -59.,   -42.,   -11.,     0.,    29.],
                 [-3547., -1820., -1808., -1768.,   -98.,   -88.,   -71.,   -41.,   -29.,     0.]]
        return freq
        
    @pytest.fixture
    def dipoles_2modes(self):
        '''
        Sets the testing transition dipole moment matrix. 
        Rounded data is from the DAR molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        dips =  [[[-3.100e-03,  7.500e-03, -1.540e-02],
                  [-3.038e-01,  2.405e-01,  2.064e-01],
                  [-7.800e-03,  1.510e-02, -2.600e-02],
                  [ 6.000e-04, -9.000e-04,  1.600e-03],
                  [ 2.260e-02, -1.480e-02, -1.380e-02],
                  [ 1.000e-04, -3.000e-04,  9.000e-04]],

                 [[-3.038e-01,  2.405e-01,  2.064e-01],
                  [-2.400e-03,  5.900e-03, -1.320e-02],
                  [ 2.490e-02, -2.290e-02, -1.860e-02],
                  [ 4.007e-01, -3.181e-01, -2.721e-01],
                  [ 1.900e-03, -1.050e-02,  2.980e-02],
                  [-1.489e-01,  1.176e-01,  1.010e-01]],

                 [[-7.800e-03,  1.510e-02, -2.600e-02],
                  [ 2.490e-02, -2.290e-02, -1.860e-02],
                  [-2.400e-03,  5.800e-03, -1.290e-02],
                  [ 7.600e-03, -1.040e-02,  9.900e-03],
                  [ 3.009e-01, -2.390e-01, -2.050e-01],
                  [ 1.040e-02, -2.020e-02,  3.440e-02]],

                 [[ 6.000e-04, -9.000e-04,  1.600e-03],
                  [ 4.007e-01, -3.181e-01, -2.721e-01],
                  [ 7.600e-03, -1.040e-02,  9.900e-03],
                  [-4.000e-04,  3.200e-03, -1.170e-02],
                  [ 4.660e-02, -4.270e-02, -3.470e-02],
                  [ 3.000e-04, -3.000e-04, -3.000e-04]],

                 [[ 2.260e-02, -1.480e-02, -1.380e-02],
                  [ 1.900e-03, -1.050e-02,  2.980e-02],
                  [ 3.009e-01, -2.390e-01, -2.050e-01],
                  [ 4.660e-02, -4.270e-02, -3.470e-02],
                  [-3.100e-03,  5.600e-03, -9.700e-03],
                  [ 2.040e-02, -1.880e-02, -1.530e-02]],

                 [[ 1.000e-04, -3.000e-04,  9.000e-04],
                  [-1.489e-01,  1.176e-01,  1.010e-01],
                  [ 1.040e-02, -2.020e-02,  3.440e-02],
                  [ 3.000e-04, -3.000e-04, -3.000e-04],
                  [ 2.040e-02, -1.880e-02, -1.530e-02],
                  [-1.800e-03,  4.600e-03, -1.090e-02]]]
        return dips
        
    @pytest.fixture
    def dipoles_3modes(self):
        '''
        Sets the testing transition dipole moment matrix. 
        Rounded data is from the GAAP molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        dips = [[[-5.620e-02,  2.570e-02, -2.220e-02],
                [-2.180e-02, -2.549e-01,  1.943e-01],
                [ 1.735e-01,  3.190e-02,  8.000e-03],
                [-5.160e-02,  1.281e-01, -6.470e-02],
                [ 9.600e-03, -3.600e-03,  3.700e-03],
                [-2.200e-03, -1.820e-02,  1.330e-02],
                [-6.800e-03,  4.000e-04, -1.000e-03],
                [-7.000e-04, -5.700e-03,  3.300e-03],
                [ 9.000e-04,  6.300e-03, -3.300e-03],
                [ 2.700e-03, -1.190e-02,  5.900e-03]],

               [[-2.180e-02, -2.549e-01,  1.943e-01],
                [-3.560e-02,  2.410e-02, -1.760e-02],
                [-2.000e-03, -2.500e-02,  2.050e-02],
                [ 2.200e-03,  7.100e-03, -4.700e-03],
                [ 1.040e-02,  3.461e-01, -2.681e-01],
                [-1.763e-01, -7.080e-02,  2.260e-02],
                [ 3.000e-04,  5.950e-02, -4.550e-02],
                [-4.690e-02,  1.301e-01, -6.560e-02],
                [-4.600e-03,  2.800e-03, -6.000e-04],
                [ 1.000e-03, -8.100e-03,  4.200e-03]],

               [[ 1.735e-01,  3.190e-02,  8.000e-03],
                [-2.000e-03, -2.500e-02,  2.050e-02],
                [-3.650e-02,  2.480e-02, -1.850e-02],
                [-3.400e-03, -7.200e-03,  3.900e-03],
                [-3.680e-02,  2.320e-02, -2.490e-02],
                [ 2.120e-02,  2.491e-01, -1.907e-01],
                [ 2.432e-01,  4.820e-02,  8.000e-03],
                [-1.500e-03, -1.350e-02,  8.100e-03],
                [-4.800e-02,  1.291e-01, -6.520e-02],
                [-3.600e-03,  1.810e-02, -9.400e-03]],

               [[-5.160e-02,  1.281e-01, -6.470e-02],
                [ 2.200e-03,  7.100e-03, -4.700e-03],
                [-3.400e-03, -7.200e-03,  3.900e-03],
                [-5.440e-02,  2.000e-03, -9.600e-03],
                [ 2.600e-03,  1.300e-03, -6.000e-04],
                [-2.800e-03, -5.800e-03,  4.400e-03],
                [-3.200e-03, -2.400e-03,  1.700e-03],
                [-2.590e-02, -2.644e-01,  1.976e-01],
                [ 1.628e-01,  3.770e-02,  8.500e-03],
                [ 9.430e-02, -1.616e-01,  8.120e-02]],

               [[ 9.600e-03, -3.600e-03,  3.700e-03],
                [ 1.040e-02,  3.461e-01, -2.681e-01],
                [-3.680e-02,  2.320e-02, -2.490e-02],
                [ 2.600e-03,  1.300e-03, -6.000e-04],
                [-1.620e-02,  1.480e-02, -6.700e-03],
                [-3.500e-03, -3.840e-02,  3.140e-02],
                [ 6.000e-04,  1.900e-03, -1.500e-03],
                [-2.700e-03, -8.400e-03,  5.600e-03],
                [ 1.000e-04,  2.000e-04, -0.000e+00],
                [ 1.000e-04,  5.000e-04, -3.000e-04]],

               [[-2.200e-03, -1.820e-02,  1.330e-02],
                [-1.763e-01, -7.080e-02,  2.260e-02],
                [ 2.120e-02,  2.491e-01, -1.907e-01],
                [-2.800e-03, -5.800e-03,  4.400e-03],
                [-3.500e-03, -3.840e-02,  3.140e-02],
                [-1.510e-02,  3.200e-02, -2.100e-02],
                [ 2.300e-03,  2.870e-02, -2.350e-02],
                [ 3.700e-03,  7.500e-03, -4.300e-03],
                [-1.900e-03, -6.200e-03,  4.200e-03],
                [-4.000e-04, -1.000e-03,  6.000e-04]],

               [[-6.800e-03,  4.000e-04, -1.000e-03],
                [ 3.000e-04,  5.950e-02, -4.550e-02],
                [ 2.432e-01,  4.820e-02,  8.000e-03],
                [-3.200e-03, -2.400e-03,  1.700e-03],
                [ 6.000e-04,  1.900e-03, -1.500e-03],
                [ 2.300e-03,  2.870e-02, -2.350e-02],
                [-1.510e-02,  2.520e-02, -1.550e-02],
                [-4.000e-04, -1.600e-03,  1.200e-03],
                [-5.200e-03, -9.600e-03,  5.400e-03],
                [-5.000e-04, -8.000e-04,  4.000e-04]],

               [[-7.000e-04, -5.700e-03,  3.300e-03],
                [-4.690e-02,  1.301e-01, -6.560e-02],
                [-1.500e-03, -1.350e-02,  8.100e-03],
                [-2.590e-02, -2.644e-01,  1.976e-01],
                [-2.700e-03, -8.400e-03,  5.600e-03],
                [ 3.700e-03,  7.500e-03, -4.300e-03],
                [-4.000e-04, -1.600e-03,  1.200e-03],
                [-3.350e-02,  2.200e-03, -6.800e-03],
                [-1.100e-03, -2.200e-02,  1.870e-02],
                [-4.700e-03, -1.420e-02,  9.500e-03]],

               [[ 9.000e-04,  6.300e-03, -3.300e-03],
                [-4.600e-03,  2.800e-03, -6.000e-04],
                [-4.800e-02,  1.291e-01, -6.520e-02],
                [ 1.628e-01,  3.770e-02,  8.500e-03],
                [ 1.000e-04,  2.000e-04, -0.000e+00],
                [-1.900e-03, -6.200e-03,  4.200e-03],
                [-5.200e-03, -9.600e-03,  5.400e-03],
                [-1.100e-03, -2.200e-02,  1.870e-02],
                [-3.470e-02, -4.300e-03, -2.200e-03],
                [ 7.000e-03,  1.220e-02, -6.400e-03]],

               [[ 2.700e-03, -1.190e-02,  5.900e-03],
                [ 1.000e-03, -8.100e-03,  4.200e-03],
                [-3.600e-03,  1.810e-02, -9.400e-03],
                [ 9.430e-02, -1.616e-01,  8.120e-02],
                [ 1.000e-04,  5.000e-04, -3.000e-04],
                [-4.000e-04, -1.000e-03,  6.000e-04],
                [-5.000e-04, -8.000e-04,  4.000e-04],
                [-4.700e-03, -1.420e-02,  9.500e-03],
                [ 7.000e-03,  1.220e-02, -6.400e-03],
                [-4.890e-02, -1.740e-02,  1.000e-04]]]
        return dips
    
    @pytest.fixture
    def Calc2dir_base_2modes(self,freqmat_2modes,dipoles_2modes):
        '''
        Sets the testing object. 
        Rounded data is from the DAR molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        return Calc2dir.Calc2dir_base(freqmat_2modes,dipoles_2modes)
    
    @pytest.fixture
    def Calc2dir_base_3modes(self,freqmat_3modes,dipoles_3modes):
        '''
        Sets the testing object. 
        Rounded data is from the GAAP molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        return Calc2dir.Calc2dir_base(freqmat_3modes,dipoles_3modes)

    def set_dipoles_diff_shape_2modes(self):
        '''
        Sets the testing transition dipole moment matrix in the (m,m,3,1) shape. 
        Rounded data is from the DAR molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        dips_diff_shape = [[[[-3.100e-03,  7.500e-03, -1.540e-02]],
                          [[-3.038e-01,  2.405e-01,  2.064e-01]],
                          [[-7.800e-03,  1.510e-02, -2.600e-02]],
                          [[ 6.000e-04, -9.000e-04,  1.600e-03]],
                          [[ 2.260e-02, -1.480e-02, -1.380e-02]],
                          [[ 1.000e-04, -3.000e-04,  9.000e-04]]],

                         [[[-3.038e-01,  2.405e-01,  2.064e-01]],
                          [[-2.400e-03,  5.900e-03, -1.320e-02]],
                          [[ 2.490e-02, -2.290e-02, -1.860e-02]],
                          [[ 4.007e-01, -3.181e-01, -2.721e-01]],
                          [[ 1.900e-03, -1.050e-02,  2.980e-02]],
                          [[-1.489e-01,  1.176e-01,  1.010e-01]]],

                         [[[-7.800e-03,  1.510e-02, -2.600e-02]],
                          [[ 2.490e-02, -2.290e-02, -1.860e-02]],
                          [[-2.400e-03,  5.800e-03, -1.290e-02]],
                          [[ 7.600e-03, -1.040e-02,  9.900e-03]],
                          [[ 3.009e-01, -2.390e-01, -2.050e-01]],
                          [[ 1.040e-02, -2.020e-02,  3.440e-02]]],

                         [[[ 6.000e-04, -9.000e-04,  1.600e-03]],
                          [[ 4.007e-01, -3.181e-01, -2.721e-01]],
                          [[ 7.600e-03, -1.040e-02,  9.900e-03]],
                          [[-4.000e-04,  3.200e-03, -1.170e-02]],
                          [[ 4.660e-02, -4.270e-02, -3.470e-02]],
                          [[ 3.000e-04, -3.000e-04, -3.000e-04]]],

                         [[[ 2.260e-02, -1.480e-02, -1.380e-02]],
                          [[ 1.900e-03, -1.050e-02,  2.980e-02]],
                          [[ 3.009e-01, -2.390e-01, -2.050e-01]],
                          [[ 4.660e-02, -4.270e-02, -3.470e-02]],
                          [[-3.100e-03,  5.600e-03, -9.700e-03]],
                          [[ 2.040e-02, -1.880e-02, -1.530e-02]]],

                         [[[ 1.000e-04, -3.000e-04,  9.000e-04]],
                          [[-1.489e-01,  1.176e-01,  1.010e-01]],
                          [[ 1.040e-02, -2.020e-02,  3.440e-02]],
                          [[ 3.000e-04, -3.000e-04, -3.000e-04]],
                          [[ 2.040e-02, -1.880e-02, -1.530e-02]],
                          [[-1.800e-03,  4.600e-03, -1.090e-02]]]]
        return dips_diff_shape

    def set_dipoles_diff_shape_3modes(self):
        '''
        Sets the testing transition dipole moment matrix in the (m,m,3,1) shape. 
        Rounded data is from the GAAP molecule calculated with a 
        def2-TZVP basis set and B3-LYP functional. 
        '''
        dips_diff_shape = [[[[-5.620e-02,  2.570e-02, -2.220e-02]],
                            [[-2.180e-02, -2.549e-01,  1.943e-01]],
                            [[ 1.735e-01,  3.190e-02,  8.000e-03]],
                            [[-5.160e-02,  1.281e-01, -6.470e-02]],
                            [[ 9.600e-03, -3.600e-03,  3.700e-03]],
                            [[-2.200e-03, -1.820e-02,  1.330e-02]],
                            [[-6.800e-03,  4.000e-04, -1.000e-03]],
                            [[-7.000e-04, -5.700e-03,  3.300e-03]],
                            [[ 9.000e-04,  6.300e-03, -3.300e-03]],
                            [[ 2.700e-03, -1.190e-02,  5.900e-03]]],


                           [[[-2.180e-02, -2.549e-01,  1.943e-01]],
                            [[-3.560e-02,  2.410e-02, -1.760e-02]],
                            [[-2.000e-03, -2.500e-02,  2.050e-02]],
                            [[ 2.200e-03,  7.100e-03, -4.700e-03]],
                            [[ 1.040e-02,  3.461e-01, -2.681e-01]],
                            [[-1.763e-01, -7.080e-02,  2.260e-02]],
                            [[ 3.000e-04,  5.950e-02, -4.550e-02]],
                            [[-4.690e-02,  1.301e-01, -6.560e-02]],
                            [[-4.600e-03,  2.800e-03, -6.000e-04]],
                            [[ 1.000e-03, -8.100e-03,  4.200e-03]]],


                           [[[ 1.735e-01,  3.190e-02,  8.000e-03]],
                            [[-2.000e-03, -2.500e-02,  2.050e-02]],
                            [[-3.650e-02,  2.480e-02, -1.850e-02]],
                            [[-3.400e-03, -7.200e-03,  3.900e-03]],
                            [[-3.680e-02,  2.320e-02, -2.490e-02]],
                            [[ 2.120e-02,  2.491e-01, -1.907e-01]],
                            [[ 2.432e-01,  4.820e-02,  8.000e-03]],
                            [[-1.500e-03, -1.350e-02,  8.100e-03]],
                            [[-4.800e-02,  1.291e-01, -6.520e-02]],
                            [[-3.600e-03,  1.810e-02, -9.400e-03]]],


                           [[[-5.160e-02,  1.281e-01, -6.470e-02]],
                            [[ 2.200e-03,  7.100e-03, -4.700e-03]],
                            [[-3.400e-03, -7.200e-03,  3.900e-03]],
                            [[-5.440e-02,  2.000e-03, -9.600e-03]],
                            [[ 2.600e-03,  1.300e-03, -6.000e-04]],
                            [[-2.800e-03, -5.800e-03,  4.400e-03]],
                            [[-3.200e-03, -2.400e-03,  1.700e-03]],
                            [[-2.590e-02, -2.644e-01,  1.976e-01]],
                            [[ 1.628e-01,  3.770e-02,  8.500e-03]],
                            [[ 9.430e-02, -1.616e-01,  8.120e-02]]],


                           [[[ 9.600e-03, -3.600e-03,  3.700e-03]],
                            [[ 1.040e-02,  3.461e-01, -2.681e-01]],
                            [[-3.680e-02,  2.320e-02, -2.490e-02]],
                            [[ 2.600e-03,  1.300e-03, -6.000e-04]],
                            [[-1.620e-02,  1.480e-02, -6.700e-03]],
                            [[-3.500e-03, -3.840e-02,  3.140e-02]],
                            [[ 6.000e-04,  1.900e-03, -1.500e-03]],
                            [[-2.700e-03, -8.400e-03,  5.600e-03]],
                            [[ 1.000e-04,  2.000e-04, -0.000e+00]],
                            [[ 1.000e-04,  5.000e-04, -3.000e-04]]],


                           [[[-2.200e-03, -1.820e-02,  1.330e-02]],
                            [[-1.763e-01, -7.080e-02,  2.260e-02]],
                            [[ 2.120e-02,  2.491e-01, -1.907e-01]],
                            [[-2.800e-03, -5.800e-03,  4.400e-03]],
                            [[-3.500e-03, -3.840e-02,  3.140e-02]],
                            [[-1.510e-02,  3.200e-02, -2.100e-02]],
                            [[ 2.300e-03,  2.870e-02, -2.350e-02]],
                            [[ 3.700e-03,  7.500e-03, -4.300e-03]],
                            [[-1.900e-03, -6.200e-03,  4.200e-03]],
                            [[-4.000e-04, -1.000e-03,  6.000e-04]]],


                           [[[-6.800e-03,  4.000e-04, -1.000e-03]],
                            [[ 3.000e-04,  5.950e-02, -4.550e-02]],
                            [[ 2.432e-01,  4.820e-02,  8.000e-03]],
                            [[-3.200e-03, -2.400e-03,  1.700e-03]],
                            [[ 6.000e-04,  1.900e-03, -1.500e-03]],
                            [[ 2.300e-03,  2.870e-02, -2.350e-02]],
                            [[-1.510e-02,  2.520e-02, -1.550e-02]],
                            [[-4.000e-04, -1.600e-03,  1.200e-03]],
                            [[-5.200e-03, -9.600e-03,  5.400e-03]],
                            [[-5.000e-04, -8.000e-04,  4.000e-04]]],


                           [[[-7.000e-04, -5.700e-03,  3.300e-03]],
                            [[-4.690e-02,  1.301e-01, -6.560e-02]],
                            [[-1.500e-03, -1.350e-02,  8.100e-03]],
                            [[-2.590e-02, -2.644e-01,  1.976e-01]],
                            [[-2.700e-03, -8.400e-03,  5.600e-03]],
                            [[ 3.700e-03,  7.500e-03, -4.300e-03]],
                            [[-4.000e-04, -1.600e-03,  1.200e-03]],
                            [[-3.350e-02,  2.200e-03, -6.800e-03]],
                            [[-1.100e-03, -2.200e-02,  1.870e-02]],
                            [[-4.700e-03, -1.420e-02,  9.500e-03]]],


                           [[[ 9.000e-04,  6.300e-03, -3.300e-03]],
                            [[-4.600e-03,  2.800e-03, -6.000e-04]],
                            [[-4.800e-02,  1.291e-01, -6.520e-02]],
                            [[ 1.628e-01,  3.770e-02,  8.500e-03]],
                            [[ 1.000e-04,  2.000e-04, -0.000e+00]],
                            [[-1.900e-03, -6.200e-03,  4.200e-03]],
                            [[-5.200e-03, -9.600e-03,  5.400e-03]],
                            [[-1.100e-03, -2.200e-02,  1.870e-02]],
                            [[-3.470e-02, -4.300e-03, -2.200e-03]],
                            [[ 7.000e-03,  1.220e-02, -6.400e-03]]],


                           [[[ 2.700e-03, -1.190e-02,  5.900e-03]],
                            [[ 1.000e-03, -8.100e-03,  4.200e-03]],
                            [[-3.600e-03,  1.810e-02, -9.400e-03]],
                            [[ 9.430e-02, -1.616e-01,  8.120e-02]],
                            [[ 1.000e-04,  5.000e-04, -3.000e-04]],
                            [[-4.000e-04, -1.000e-03,  6.000e-04]],
                            [[-5.000e-04, -8.000e-04,  4.000e-04]],
                            [[-4.700e-03, -1.420e-02,  9.500e-03]],
                            [[ 7.000e-03,  1.220e-02, -6.400e-03]],
                            [[-4.890e-02, -1.740e-02,  1.000e-04]]]]
        return dips_diff_shape
    
    
    def test_read_dipolemat(self,Calc2dir_base_2modes,Calc2dir_base_3modes,dipoles_2modes,dipoles_3modes):
        '''
        The transition dipole moment matrix that is obtained by VIBRATIONS calculations
        has the shape (n,n,1,3). In order to use it in the following calculations it is
        reduced to the shape (n,n,3). 
        
        The test takes the (n,n,3) and a (n,n,1,3) shaped dipole matrix, uses the 
        read_dipolemat function and compares it to the given (n,n,3) matrix.         
        
        '''        
        assert Calc2dir_base_2modes.read_dipolemat(dipoles_2modes) == dipoles_2modes
        assert Calc2dir_base_2modes.read_dipolemat(self.set_dipoles_diff_shape_2modes()) == dipoles_2modes
        assert Calc2dir_base_3modes.read_dipolemat(dipoles_3modes) == dipoles_3modes
        assert Calc2dir_base_3modes.read_dipolemat(self.set_dipoles_diff_shape_3modes()) == dipoles_3modes

    def test_calc_nmodes(self,Calc2dir_base_2modes,Calc2dir_base_3modes):
        '''
        The number of modes equals the length of the frequency matrix in one direction.
        
        '''
        assert Calc2dir_base_2modes.calc_nmodes() == 6
        assert Calc2dir_base_3modes.calc_nmodes() == 10
    
    def test_calc_num_oscill(self,Calc2dir_base_2modes,Calc2dir_base_3modes):
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
        n3 = Calc2dir_base_3modes.calc_nmodes()
        assert Calc2dir_base_2modes.calc_num_oscill(n2) == 2
        assert Calc2dir_base_3modes.calc_num_oscill(n3) == 3

    def test_calc_trans2int(self,Calc2dir_base_2modes,Calc2dir_base_3modes):
        '''
        Calculates the intensity matrix from the given transition dipole moment matrix 
        and the given frequeny matrix.
        
        '''
        ints_2modes = [[    0,   852,     4,     0,     8,     0],
                       [ -852,     0,     0,  1482,     4,   204],
                       [   -4,     0,     0,     1,   836,     7],
                       [    0, -1482,    -1,     0,     0,     0],
                       [   -8,    -4,  -836,     0,     0,     0],
                       [    0,  -204,    -7,     0,     0,     0]]
        ints_3modes = [[ 0.000000e+00,  4.467572e+02,  1.359320e+02,  1.037159e+02,  1.027200e+00,  4.447700e+00,  4.130000e-01,  3.855000e-01,  4.532000e-01,  1.633400e+00],
                       [-4.467572e+02,  0.000000e+00,  3.160000e-02,  1.010000e-02,  8.277673e+02,  1.589211e+02,  2.459750e+01,  1.044772e+02,  1.318000e-01,  3.844000e-01],
                       [-1.359320e+02, -3.160000e-02,  0.000000e+00,  7.900000e-03,  1.076940e+01,  4.262564e+02,  2.679190e+02,  1.107800e+00,  1.035535e+02,  1.943900e+00],
                       [-1.037159e+02, -1.010000e-02, -7.900000e-03,  0.000000e+00,  3.690000e-02,  2.562000e-01,  8.040000e-02,  4.745587e+02,  1.220420e+02,  1.843626e+02],
                       [-1.027200e+00, -8.277673e+02, -1.076940e+01, -3.690000e-02,  0.000000e+00,  6.200000e-02,  4.000000e-04,  1.560000e-02,  0.000000e+00,  1.000000e-04],
                       [-4.447700e+00, -1.589211e+02, -4.262564e+02, -2.562000e-01, -6.200000e-02,  0.000000e+00,  5.890000e-02,  1.040000e-02,  8.800000e-03,  3.000000e-04],
                       [-4.130000e-01, -2.459750e+01, -2.679190e+02, -8.040000e-02, -4.000000e-04, -5.890000e-02,  0.000000e+00,  3.000000e-04,  1.560000e-02,  2.000000e-04],
                       [-3.855000e-01, -1.044772e+02, -1.107800e+00, -4.745587e+02, -1.560000e-02, -1.040000e-02, -3.000000e-04,  0.000000e+00,  2.510000e-02,  3.230000e-02],
                       [-4.532000e-01, -1.318000e-01, -1.035535e+02, -1.220420e+02, -0.000000e+00, -8.800000e-03, -1.560000e-02, -2.510000e-02,  0.000000e+00,  1.740000e-02],
                       [-1.633400e+00, -3.844000e-01, -1.943900e+00, -1.843626e+02, -1.000000e-04, -3.000000e-04, -2.000000e-04, -3.230000e-02, -1.740000e-02,  0.000000e+00]]
        
        np.testing.assert_almost_equal(Calc2dir_base_2modes.calc_trans2int(),ints_2modes, decimal=4)
        np.testing.assert_almost_equal(Calc2dir_base_3modes.calc_trans2int(),ints_3modes, decimal=4)
        
    def test_n2s(self,Calc2dir_base_2modes):
        '''
        Takes a number with a decimal point and changes it to an underscore. 
        '''        
        assert Calc2dir_base_2modes.n2s(2.5) == '2_5'


    
# class spectra(Calc2dir_base):
    
    
    
#     @staticmethod
#     def set_line_spacing(maximum,number):
#         '''
#         Use this for matplotlib.pyplot contour plots in order to set the 
#         number of lines. 
#         Example: plt.contour(x,y,z,set_line_spacing(abs(z.max()),20))
        
#         @param maximum: maximum value of the plotted array
#         @type maximum: float
        
#         @param number: number of plotted contour lines for the positive/negative values
#         @type number: int
        
#         @return: new values at which the lines are plotted
#         @rtype: list

#         '''
#         firstvalue = maximum/number
#         negspace = np.linspace(-maximum,-firstvalue,number)
#         posspace = np.linspace(firstvalue,maximum,number)
#         return np.concatenate((negspace,posspace))
    
#     @staticmethod
#     def gauss_func(intensity,x,x0,halfwidth):
#         '''
#         Computes a single value at position x for a 
#         1D gaussian type function.
        
#         @param intensity: Intensity of a peak
#         @type intensity: Float
        
#         @param x: x-values 
#         @type x: List of floats
        
#         @param x0: Position of a peak
#         @type x0: Float
        
#         @param halfwidth: Parameter to control the width of the peaks
#         @type halfwidth: Float
#         @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
#         @return: Corresponding y-values
#         @rtype: List of floats
        
#         '''        
#         gamma = halfwidth / (2.0*math.sqrt(2.0*math.log(2.0)))
#         return (intensity/(2.0 * math.pi * gamma**2)) * np.exp( - (x - x0)**2 / (2*gamma**2) )
    
#     @staticmethod
#     def gauss2d_func(intensity,x,x0,y,y0,halfwidth):
#         '''
#         Computes a single value at position x for a 
#         1D gaussian type function.
        
#         @param intensity: Intensity of a peak
#         @type intensity: Float
        
#         @param x: x-values 
#         @type x: List of floats
        
#         @param x0: Position of a peak
#         @type x0: Float
        
#         @param halfwidth: Parameter to control the width of the peaks
#         @type halfwidth: Float
#         @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
#         @return: Corresponding y-values
#         @rtype: List of floats
        
#         '''        
#         gamma = halfwidth / (2.0*math.sqrt(2.0*math.log(2.0)))
#         return (intensity/(2.0 * math.pi * gamma**2)) * np.exp( - ((x - x0)**2 + (y - y0)**2) / (2*gamma**2) )
    
#     @staticmethod
#     def lorentz_func(intensity,x,x0,halfwidth):
#         '''
#         Computes a single value at position x for a 
#         1D lorentzian type function.
        
#         @param intensity: Intensity of a peak
#         @type intensity: Float
        
#         @param x: x-values 
#         @type x: List of floats
        
#         @param x0: Position of a peak
#         @type x0: Float
        
#         @param halfwidth: Parameter to control the width of the peaks
#         @type halfwidth: Float
#         @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
#         @return: Corresponding y-values
#         @rtype: List of floats
        
#         '''
#         return ( (intensity*halfwidth) / (2*math.pi) ) / ( (x-x0)**2 + (halfwidth/2)**2 )
    
#     @staticmethod
#     def lorentz2d_func(intensity,x,x0,y,y0,halfwidth=5):
#         '''
#         Computes a single value at grid x,y for a 
#         2D lorentzian type function.
        
#         @param intensity: Intensity of a peak
#         @type intensity: Float
        
#         @param x/y: x-values 
#         @type x/y: List of floats
        
#         @param x0/y0: Position of a peak
#         @type x0/y0: Float
        
#         @param halfwidth: Parameter to control the width of the peaks
#         @type halfwidth: Float
#         @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
#         @return: Corresponding y-values
#         @rtype: List of floats
        
#         '''
#         return ( (intensity*halfwidth) / (2*math.pi) ) / ( (x-x0)**2 + (y-y0)**2 + (halfwidth/2)**2 )
    
#     @staticmethod
#     def get_norm_1d_spectrum(xmin,xmax,freqs,ints,steps=5000,halfwidth=5,ftype='gauss'):
#         '''
#         Sums up all gauss/lorentz functions for each peak and sets the highest value to one.
        
#         @param xmin/xmax: minimum and maximum value of the spectrum
#         @type xmin/xmax: Float
        
#         @param freqs/ints: frequencies and corresponding intensities
#         @type freqs/ints: Lists of floats
        
#         @param steps: number of points for the x-axis
#         @type steps: Integer
        
#         @param halfwidth: Parameter to control the width of the peaks
#         @type halfwidth: Float
#         @note halfwidth: Does not necessarily correlate to actual FWHM of a peak
        
#         @param ftype: Choses between gauss and lorentz function
#         @type ftype: String
        
#         @return: x and y values of the 1D plot
#         @rtype: Lists of floats
        
#         '''
#         x = np.linspace(xmin,xmax,steps)
#         y = np.zeros(steps)
        
#         for freq, inten in zip(freqs,ints):
#             if ftype.lower() == 'gauss':
#                 y += spectra.gauss_func(inten,x,freq,halfwidth)
#             if ftype.lower() == 'lorentz':
#                 y += spectra.lorentz_func(inten,x,freq,halfwidth)

#         y = list(map(lambda x : x/y.max(), y))
#         return x,y
    
#     @staticmethod
#     def get_2d_spectrum(xmin,xmax,exc,ble,emi,steps=2000,halfwidth=15,ftype='gauss'):
#         '''
#         Text
        
#         '''
        
#         x = np.linspace(xmin, xmax, steps)
#         y = np.linspace(xmin, xmax, steps)
#         xx, yy = np.meshgrid(x, y)
        
#         z = np.zeros((steps,steps))
        
#         exc_x,exc_y,exc_i = exc
#         emi_x,emi_y,emi_i = emi
#         ble_x,ble_y,ble_i = ble
        
#         y_vals = []
#         y_vals.extend([exc_y,emi_y,ble_y])
#         i_vals = []
#         i_vals.extend([exc_i,emi_i,ble_i])

        
#         for freq_x, freq_y, inten in zip(exc_x+emi_x+ble_x, exc_y+emi_y+ble_y, exc_i+emi_i+ble_i):
#             if ftype.lower() == 'gauss':
#                 z += spectra.gauss2d_func(inten,xx,freq_x,yy,freq_y,halfwidth=5)
#             if ftype.lower() == 'lorentz':
#                 z += spectra.lorentz2d_func(inten,xx,freq_x,yy,freq_y,halfwidth=5)
            
        
#         return x,y,z
    
#     @staticmethod
#     def norm_2d_spectum(z,max_z):
#         '''
#         Text
        
#         '''        
#         for i in range(len(z)):
#             for j in range(len(z)):
#                 z[i][j] = z[i][j] / max_z
        
#         return z