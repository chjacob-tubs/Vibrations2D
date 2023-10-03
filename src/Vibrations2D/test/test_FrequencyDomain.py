import pytest
import numpy as np

import Vibrations2D as V2D

# FrequencyDomain class needs same inputs as Calcd2dir class,
# so we take some methods from test_Calc2dir
from .test_Calc2dir import freq_dip_input

# Reference data for
# Vibrations2D.frequencydomain class
pathref = 'testdata/reference_data/'
Refdat = np.load(pathref+'FrequencyDomain_class_init_ref.npz')
dipoles_ref = Refdat['dipoles']
freqs_ref = Refdat['freqs']
freqmat_ref = Refdat['freqmat']
RefFreqDom = {'dipoles': dipoles_ref,
              'freqs': freqs_ref,
              'noscill': 2,
              'freqmat': freqmat_ref,
              'nmodes': 6,
              'print_output': True,
              'n_grid': 256,
              'T2': 2,
              'polarization': 'ZZZZ',
              'pol_list': [0, 0, 0, 0]}


def attribute_assert_FreqDom(dipoles, freqs, noscill, freqmat,
                             nmodes, print_output, n_grid, T2,
                             polarization, pol_list):
    np.testing.assert_almost_equal(dipoles, RefFreqDom['dipoles'])
    np.testing.assert_almost_equal(freqs, RefFreqDom['freqs'])
    assert noscill == 2
    np.testing.assert_almost_equal(freqmat, RefFreqDom['freqmat'])
    assert nmodes == 6
    assert print_output == True
    assert n_grid == 256
    assert T2 == 2
    assert polarization == 'ZZZZ'
    assert pol_list == [0, 0, 0, 0]


def test_frequencydomain_init(freq_dip_input):
    # Arrange
    VCI_freq, VCI_dipo = freq_dip_input
    # Act
    # FD = FrequencyDomain
    FD = V2D.frequencydomain(VCI_freq, VCI_dipo)
    # Assert
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)


@pytest.fixture
def FreqDom(freq_dip_input):
    # Arrange
    VCI_freq, VCI_dipo = freq_dip_input
    # Act
    # FD = FrequencyDomain
    FreDo = V2D.frequencydomain(VCI_freq, VCI_dipo)
    return FreDo


def test_calc_excitation(FreqDom):
    # Arrange
    exci_ref = np.load(pathref+'FrequencyDomain_calc_excitation_ref.npy')
    FD = FreqDom
    # Act
    exci = FD.calc_excitation()
    # Assert
    np.testing.assert_almost_equal(exci, exci_ref)
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)


def test_calc_stimulatedemission(FreqDom):
    # Arrange
    emi_ref = ([2066.678659436857, 2127.717035117661],
               [2066.678659436857, 2127.717035117661],
               [-854.9721103319562, -713.160629954444])
    FD = FreqDom
    # Act
    emi = FD.calc_stimulatedemission()
    # Assert
    np.testing.assert_almost_equal(emi, emi_ref)
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)


def test_calc_bleaching(FreqDom):
    # Arrange
    ble_ref = ([2066.678659436857, 2127.717035117661, 2066.678659436857,
               2127.717035117661],
               [2066.678659436857, 2066.678659436857, 2127.717035117661,
               2127.717035117661],
               [-854.972110331956, -854.972110331956, -713.160629954444,
               -713.160629954444])
    FD = FreqDom
    # Act
    ble = FD.calc_bleaching()
    # Assert
    np.testing.assert_almost_equal(ble, ble_ref)
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)


def test_calc_all_2d_process(FreqDom):
    # Arrange
    Ref = np.load(pathref+'FrequencyDomain_calc_all_2d_process_ref.npz')
    exc_ref = Ref['exc']
    ste_ref = Ref['ste']
    ble_ref = Ref['ble']
    FD = FreqDom
    # Act
    exc, ste, ble = FD.calc_all_2d_process()
    # Assert
    np.testing.assert_almost_equal(exc, exc_ref)
    np.testing.assert_almost_equal(ste, ste_ref)
    np.testing.assert_almost_equal(ble, ble_ref)
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)


def test__get_2d_spectrum(FreqDom):
    # Arrange
    Ref = np.load(pathref+'FrequencyDomain__get_2d_spectrum_ref.npz')
    x_ref = Ref['x']
    z_ref = Ref['z']
    FD = FreqDom
    xmin = 1400
    xmax = 2100
    # Act
    x, z = FD._get_2d_spectrum(xmin, xmax)
    # Assert
    np.testing.assert_almost_equal(x, x_ref)
    np.testing.assert_almost_equal(z, z_ref)
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)


def test_calculate_S(FreqDom):
    # Arrange
    Ref = np.load(pathref+'FrequencyDomain_calculate_S_ref.npz')
    S1_ref = Ref['S1']
    S2_ref = Ref['S2']
    S3_ref = Ref['S3']
    S4_ref = Ref['S4']
    S5_ref = Ref['S5']
    S6_ref = Ref['S6']

    InRef = np.load(pathref+'FrequencyDomain_get_2d_spectrum_ref.npz')
    axis = InRef['w']

    FD = FreqDom
    # Act
    S1, S2, S3, S4, S5, S6 = FD.calculate_S(axis)
    # Assert
    np.testing.assert_almost_equal(S1, S1_ref)
    np.testing.assert_almost_equal(S2, S2_ref)
    np.testing.assert_almost_equal(S3, S3_ref)
    np.testing.assert_almost_equal(S4, S4_ref)
    np.testing.assert_almost_equal(S5, S5_ref)
    np.testing.assert_almost_equal(S6, S6_ref)
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)


def test_get_absorptive_spectrum(FreqDom):
    # Arrange
    Ref = np.load(pathref+'FrequencyDomain_get_2d_spectrum_ref.npz')
    w_ref = Ref['w']
    S_ref = Ref['S']
    FD = FreqDom
    # Act
    S, w = FD.get_absorptive_spectrum()
    # Assert
    np.testing.assert_almost_equal(w, w_ref)
    np.testing.assert_almost_equal(S, S_ref)
    attribute_assert_FreqDom(FD.dipoles,
                             FD.freqs,
                             FD.noscill,
                             FD.freqmat,
                             FD.nmodes,
                             FD.print_output,
                             FD.n_grid,
                             FD.T2,
                             FD.polarization,
                             FD.pol_list)
