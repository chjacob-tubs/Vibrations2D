import pytest
import numpy as np

import Vibrations2D as V2D

# Calc2dir inherits its properties to TimeDomain,
# so we take some methods from test_Calc2dir
from .test_Calc2dir import freq_dip_input, attribute_assert_Calc2dir_base


def attribute_assert_TimeDomain(print_output, n_t, dt, T2, t2, polarization,
                                pol_list, omega_off):
    assert print_output == True
    assert n_t == 128
    assert dt == 0.25
    assert T2 == 2
    assert t2 == 0
    assert polarization == 'ZZZZ'
    assert pol_list == [0, 0, 0, 0]
    assert omega_off == 2097


def test_TimeDomain_init(freq_dip_input):
    """tests the initialization of Calc2dir_base."""
    # Arrange
    freqs, dips = freq_dip_input
    # Act
    TD = V2D.timedomain(freqs, dips)
    # Assert
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


@pytest.fixture
def timedomain_class(freq_dip_input):
    '''
    Sets the testing object.
    Rounded data is from the DAR molecule calculated with a
    def2-TZVP basis set and B3-LYP functional.
    '''
    freqs, dips = freq_dip_input
    tido = V2D.timedomain(freqs, dips)
    return tido


def test__get_omega_off(timedomain_class):
    # Arrange
    TD = timedomain_class
    # Act
    omega_off = TD._get_omega_off()
    # Assert
    assert omega_off == 2097
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_set_omega(timedomain_class):
    # Arrange
    TD = timedomain_class
    omg1_ref = np.array([-30.32134056,  30.71703512])
    omg2_ref = np.array([-71.47178168, -20.78185307,  53.27274221])
    # Act
    omg1, omg2 = TD.set_omega()
    # Assert
    np.testing.assert_almost_equal(omg1, omg1_ref)
    np.testing.assert_almost_equal(omg2, omg2_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_calc_fourpointcorr(timedomain_class):
    # Arrange
    TD = timedomain_class
    mu1 = [1, 1, 1]
    mu2 = [2, 2, 2]
    mu3 = [3, 3, 3]
    mu4 = [4, 4, 4]
    pathway1 = 'jjii'
    pathway2 = 'jikl'
    fak1 = 1
    fak2 = 2
    fak3 = 4
    # Act
    S1 = TD.calc_fourpointcorr(pathway1, fak1, fak2, fak3, mu1, mu2)
    S2 = TD.calc_fourpointcorr(pathway2, fak1, fak2, fak3, mu1, mu2, mu3, mu4)
    # Assert
    np.testing.assert_almost_equal(S1, 0.23333333333333345)
    np.testing.assert_almost_equal(S2, 0.2333333333333334)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_calc_axes(timedomain_class):
    # Arrange
    TD = timedomain_class
    pathref = 'testdata/reference_data/'
    ticks_ref = np.load(pathref + 'Timedomain_calc_axes_ref.npy')
    # Act
    ticks = TD.calc_axes()
    # Assert
    np.testing.assert_almost_equal(ticks, ticks_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_calc_diagrams(timedomain_class):
    # Arrange
    TD = timedomain_class
    pathref = 'testdata/reference_data/'
    R_ref = np.load(pathref + 'Timedomain_calc_diagrams_slow_ref.npz')
    R1_ref = R_ref['R1']
    R2_ref = R_ref['R2']
    R3_ref = R_ref['R3']
    R4_ref = R_ref['R4']
    R5_ref = R_ref['R5']
    R6_ref = R_ref['R6']
    # Act
    R1, R2, R3, R4, R5, R6 = TD.calc_diagrams_slow()
    # Assert
    np.testing.assert_almost_equal(R1, R1_ref)
    np.testing.assert_almost_equal(R2, R2_ref)
    np.testing.assert_almost_equal(R3, R3_ref)
    np.testing.assert_almost_equal(R4, R4_ref)
    np.testing.assert_almost_equal(R5, R5_ref)
    np.testing.assert_almost_equal(R6, R6_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_calc_sum_diagram(timedomain_class):
    # Arrange
    TD = timedomain_class
    pathref = 'testdata/reference_data/'
    R_ref = np.load(pathref + 'Timedomain_calc_sum_diagram_ref.npy')
    R_in = np.load(pathref + 'Timedomain_calc_diagrams_slow_ref.npz')
    R1_in = R_in['R1']
    R2_in = R_in['R2']
    R3_in = R_in['R3']
    # Act
    R = TD.calc_sum_diagram(R1_in, R2_in, R3_in)
    # Assert
    np.testing.assert_almost_equal(R, R_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_calc_2d_fft(timedomain_class):
    # Arrange
    TD = timedomain_class
    pathref = 'testdata/reference_data/'
    R_sum = np.load(pathref + 'Timedomain_calc_sum_diagram_ref.npy')
    R_ft_ref = np.load(pathref + 'Timedomain_calc_2d_fft_ref.npy')
    # Act
    R_ft = TD.calc_2d_fft(R_sum)
    # Assert
    np.testing.assert_almost_equal(R_ft, R_ft_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_get_absorptive_spectrum(timedomain_class):
    # Arrange
    TD = timedomain_class
    pathref = 'testdata/reference_data/'
    dat = np.load(pathref + 'Timedomain_get_absorptive_spectrum_ref.npz')
    R_ref = dat['R']
    axes_ref = dat['axes']
    # Act
    R, axes = TD.get_absorptive_spectrum()
    # Assert
    np.testing.assert_almost_equal(R, R_ref)
    np.testing.assert_almost_equal(axes, axes_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_get_photon_echo_spectrum(timedomain_class):
    # Arrange
    TD = timedomain_class
    pathref = 'testdata/reference_data/'
    dat = np.load(pathref + 'Timedomain_get_photon_echo_spectrum_ref.npz')
    R_ref = dat['R']
    axes_ref = dat['axes']
    # Act
    R, axes = TD.get_photon_echo_spectrum()
    # Assert
    np.testing.assert_almost_equal(R, R_ref)
    np.testing.assert_almost_equal(axes, axes_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)


def test_get_correlation_spectrum(timedomain_class):
    # Arrange
    TD = timedomain_class
    pathref = 'testdata/reference_data/'
    dat = np.load(pathref + 'Timedomain_get_correlation_spectrum_ref.npz')
    R_ref = dat['R']
    axes_ref = dat['axes']
    # Act
    R, axes = TD.get_correlation_spectrum()
    # Assert
    np.testing.assert_almost_equal(R, R_ref)
    np.testing.assert_almost_equal(axes, axes_ref)
    attribute_assert_Calc2dir_base(TD.freqs, TD.dipoles, TD.noscill)
    attribute_assert_TimeDomain(TD.print_output, TD.n_t, TD.dt,
                                TD.T2, TD.t2, TD.polarization,
                                TD.pol_list, TD.omega_off)
