import pytest
import numpy as np

import Vibrations2D as V2D

# Reference data for excitonmodel class
Ref_exciton_model = {
    'cmat': np.array([[2117.21685328, 30.37232339],
                      [30.37232339, 2116.77966672]]),
    'dipoles': np.array([[-0.40287641, -0.78631065, -0.18256666],
                         [0.69296726, -0.13235363, -0.56492409]]),
    'noscill': 2
    }


@pytest.fixture
def EXC_dat():
    path = 'testdata/'
    EXC_cmat = np.load(path+'Exciton_cmat_lm.npy')
    EXC_dipo = np.load(path+'Exciton_dipolemoments_lm.npy')
    return EXC_cmat, EXC_dipo


def attribute_assert_excitonmodel(cmat, dipoles, noscill):
    # Reference data
    ref_cmat = Ref_exciton_model['cmat']
    ref_dipoles = Ref_exciton_model['dipoles']
    ref_noscill = Ref_exciton_model['noscill']
    # Assert
    np.testing.assert_almost_equal(cmat, ref_cmat)
    np.testing.assert_almost_equal(dipoles, ref_dipoles)
    assert noscill == ref_noscill


def test_excitonmodel_init(EXC_dat):
    # Arrange
    EXC_cmat, EXC_dipo = EXC_dat
    # Act
    # ExcitonModel = EM : exitonmodel class
    EM = V2D.excitonmodel(EXC_cmat, EXC_dipo)
    # Assert
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)


@pytest.fixture
def excitonmodel_class(EXC_dat):
    EXC_cmat, EXC_dipo = EXC_dat
    EM = V2D.excitonmodel(EXC_cmat, EXC_dipo)
    return EM


# Reference states:
# EM = Vibrations2D.excitonmodel(EXC_cmat, EXC_dipo)
# -> See: test_excitonmodel_init
#
# states = Vibrations2D.EM.generate_sorted_states()

states_ref = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]


def test_generate_sorted_states(excitonmodel_class):
    # Arrange
    EM = excitonmodel_class
    # Act
    states = EM.generate_sorted_states()
    # Assert
    assert states == states_ref
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)


def test_eval_hamiltonian(excitonmodel_class):
    # Arrange
    EM = excitonmodel_class
    pathref = 'testdata/reference_data/'
    hamiltonian_ref = np.load(pathref+'Excitonmodel_eval_hamiltonian_ref.npy')
    # Act
    hamiltonian = EM.eval_hamiltonian(states_ref)
    # Assert
    np.testing.assert_almost_equal(hamiltonian, hamiltonian_ref)
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)


def test_eval_dipolmatrix(excitonmodel_class):
    # Arrange
    EM = excitonmodel_class
    pathref = 'testdata/reference_data/'
    dipmatrix_ref = np.load(pathref+'Excitonmodel_eval_dipmatrix_ref.npy')
    # Act
    dipmatrix = EM.eval_dipolmatrix(states_ref)
    # Assert
    np.testing.assert_almost_equal(dipmatrix, dipmatrix_ref)
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)


def test_add_anharmonicity(excitonmodel_class):
    # Arrange
    EM = excitonmodel_class
    pathref = 'testdata/reference_data/'
    filenameref = 'Excitonmodel_add_anharmonicity_ref.npy'
    add_anh_hamiltonian_ref = np.load(pathref+filenameref)
    hamiltonian_ref = np.load(pathref+'Excitonmodel_eval_hamiltonian_ref.npy')
    anharmonicity = 20
    # Act
    add_anh_hamiltonian = EM.add_anharmonicity(hamiltonian_ref, anharmonicity)
    # Assert
    np.testing.assert_almost_equal(add_anh_hamiltonian,
                                   add_anh_hamiltonian_ref)
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)


def test_add_all_anharmonicity(excitonmodel_class):
    # Arrange
    EM = excitonmodel_class
    pathref = 'testdata/reference_data/'
    filenameref = 'Excitonmodel_add_all_anharmonicity_ref.npy'
    add_anh_hamiltonian_ref = np.load(pathref+filenameref)
    hamiltonian_ref = np.load(pathref+'Excitonmodel_eval_hamiltonian_ref.npy')
    anharmonicity = 20
    # Act
    add_anh_hamiltonian = EM.add_all_anharmonicity(hamiltonian_ref,
                                                   anharmonicity)
    # Assert
    np.testing.assert_almost_equal(add_anh_hamiltonian,
                                   add_anh_hamiltonian_ref)
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)


def test_get_nm_freqs_dipolmat(excitonmodel_class):
    # Arrange
    EM = excitonmodel_class
    anharmonicity = 20
    pathref = 'testdata/reference_data/'
    Ref = np.load(pathref+'Excitonmodel_get_nm_freqs_dipolmat_ref.npz')
    freqs_lm_ref = Ref['freqs_lm']
    dipole_nm_ref = Ref['dipole_nm']
    #  Act
    freqs_lm, dipole_nm = EM.get_nm_freqs_dipolmat(anharmonicity)
    # Assert
    np.testing.assert_almost_equal(freqs_lm, freqs_lm_ref)
    np.testing.assert_almost_equal(abs(dipole_nm), abs(dipole_nm_ref))
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)


def test_get_nm_freqs_dipolmat_from_VSCF(excitonmodel_class):
    # Arrange
    EM = excitonmodel_class
    pathref = 'testdata/reference_data/'
    filename = 'Excitonmodel_get_nm_freqs_dipolmat_from_VSCF_ref.npz'
    Ref = np.load(pathref+filename)
    freqs_lm_ref = Ref['freqs_lm']
    dipole_nm_ref = Ref['dipole_nm']

    a = 2096.5548762343137
    b = 2096.25065139409
    aa = 4173.605100783076
    bb = 4172.998692247222
    EX_freq_new = [0, a, b, aa, bb, a+b]
    #  Act
    freqs_lm, dipole_nm = EM.get_nm_freqs_dipolmat_from_VSCF(EX_freq_new)
    # Assert
    np.testing.assert_almost_equal(freqs_lm, freqs_lm_ref)
    np.testing.assert_almost_equal(abs(dipole_nm), abs(dipole_nm_ref))
    attribute_assert_excitonmodel(EM.cmat, EM.dipoles, EM.noscill)
