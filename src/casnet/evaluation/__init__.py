import numpy as np
from pyscf import gto, mcscf

from casnet.config import CONV_TOL, N_STATES, NCAS, NELECAS, SORT_MO_INDICES

basis_dict = {
    "sto_6g": 36,
}


def run_casscf_calculation(geometry_file: str, guess_orbitals: np.ndarray, basis="sto-6g"):
    molecule = gto.M(atom=geometry_file, basis=basis, spin=0, symmetry=True)
    molecule.verbose = 0

    hartree_fock = molecule.RHF()
    n_states = N_STATES
    weights = np.ones(n_states) / n_states
    casscf = hartree_fock.CASSCF(ncas=NCAS, nelecas=NELECAS).state_average(weights)
    casscf.conv_tol = CONV_TOL

    conv, e_tot, imacro, imicro, iinner, _, _, mo_coeffs, mo_energies = casscf.kernel(
        guess_orbitals
    )
    return conv, e_tot, imacro, imicro, iinner, mo_coeffs, mo_energies


def compute_converged_casscf_orbitals(model_path: str, geometry_path: str, basis: str):
    molecule = gto.M(atom=geometry_path, basis=basis, spin=0, symmetry=True)
    molecule.verbose = 0

    hartree_fock = molecule.RHF()
    hartree_fock.kernel()

    n_states = N_STATES
    weights = np.ones(n_states) / n_states
    casscf = hartree_fock.CASSCF(ncas=NCAS, nelecas=NELECAS).state_average(weights)
    casscf.conv_tol = CONV_TOL

    mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
    mo = casscf.sort_mo(SORT_MO_INDICES, mo)

    _, _, _, _, _, _, _, mo_coeffs, mo_energies = casscf.kernel(mo)
    return mo_energies, mo_coeffs


def compute_casci_energy(geometry_path: str, orbitals: np.ndarray, basis: str) -> float:
    molecule = gto.M(atom=geometry_path, basis=basis, spin=0, symmetry=True)
    molecule.verbose = 0

    hartree_fock = molecule.RHF()

    n_states = N_STATES
    weights = np.ones(n_states) / n_states
    casci = hartree_fock.CASCI(ncas=NCAS, nelecas=NELECAS).state_average(weights)
    casci.conv_tol = CONV_TOL

    output = casci.kernel(orbitals)
    return output[0]


def compute_casscf_energy(geometry_path: str, basis: str) -> float:
    molecule = gto.M(atom=geometry_path, basis=basis, spin=0, symmetry=True)
    molecule.verbose = 0

    hartree_fock = molecule.RHF()
    hartree_fock.kernel()

    n_states = N_STATES
    weights = np.ones(n_states) / n_states
    casscf = hartree_fock.CASSCF(ncas=NCAS, nelecas=NELECAS).state_average(weights)
    casscf.conv_tol = CONV_TOL

    mo = mcscf.project_init_guess(casscf, hartree_fock.mo_coeff)
    mo = casscf.sort_mo(SORT_MO_INDICES, mo)

    _, e_tot, _, _, _, _, _, _, _ = casscf.kernel(mo)
    return e_tot


def compute_cas_orb_energies_from_mo(
    geometry_path: str, basis: str, predicted_mos: np.ndarray
) -> np.ndarray:
    molecule = gto.M(atom=geometry_path, basis=basis, spin=0, symmetry=True)
    molecule.verbose = 0

    hartree_fock = molecule.RHF()

    n_states = N_STATES
    weights = np.ones(n_states) / n_states
    casci = hartree_fock.CASCI(ncas=NCAS, nelecas=NELECAS).state_average(weights)
    casci.conv_tol = CONV_TOL

    casci.kernel(predicted_mos)
    _, _, mo_energy = casci.canonicalize(predicted_mos)
    return mo_energy
