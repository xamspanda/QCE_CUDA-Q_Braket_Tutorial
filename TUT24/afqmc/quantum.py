import json
import numpy as np
import pennylane as qml
from afqmc.utils.chemical_preparation import chemistry_preparation
from afqmc.utils.matchgate import construct_covariance
from afqmc.trial_wavefunction.quantum_ovlp import QTrial
from afqmc.qmc.quantum_shadow import cqa_afqmc
from pyscf import gto, scf


def run(time_steps: int, delta_tau: float, num_walkers: int = 2):
    print("Running QC-AFQMC calculation")

    # perform HF calculations
    mol = gto.M(atom="H 0. 0. 0.; H 0. 0. 0.75", basis="sto-3g")
    hf = mol.RHF()
    hf.kernel()

    prop = chemistry_preparation(mol, hf)

    # untar the shadows
    with open(f'inputs/results.json', 'r') as file:
        matchgates = json.load(file)['dataDictionary']
        Q_save = matchgates['Q_save']
        output = matchgates['output']

    # then reconstruct the Q_list
    def Q_reconstruct(signed_permutation):
        size = len(signed_permutation)
        Q = np.zeros((size, size))
        for i in range(size):
            Q[int(abs(signed_permutation[i]) - 1), i] = np.sign(signed_permutation[i])
        return Q

    Q_list = []
    for signed_permutation in Q_save:
        Q = Q_reconstruct(signed_permutation)
        Q_list.append(Q)

    outcomes = []
    for i in output:
        shadow_outcome = []
        for j in list(i.keys()):
            shadow_outcome.append([construct_covariance(j), i.get(j)])
        outcomes.append(shadow_outcome)

    shadow = (outcomes, Q_list)
    print("The matchgate shadows are successfully reproduced!")

    Angstrom_to_Bohr = 1.88973
    symbols = ["H", "H"]
    geometry = np.array([[0., 0., 0.], [0., 0., 0.75 * Angstrom_to_Bohr]])
    hamiltonian, _ = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=0, basis='sto-3g')

    psi0 = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    qtrial = QTrial(prop=prop, initial_state=[0, 1], ansatz_circuit=V_T, ifshadow=True, shadow=shadow)

    local_energies, weights = cqa_afqmc(
        num_walkers=num_walkers,
        num_steps=time_steps,
        dtau=delta_tau,
        trial=qtrial,
        hamiltonian=hamiltonian,
        psi0=psi0,
        max_pool=num_walkers,
    )

    print("Calculation finished!")

    return {
        "local_energies_real": np.real(local_energies).tolist(),
        "local_energies_imag": np.imag(local_energies).tolist(),
        "weights": np.real(weights).tolist(),
    }

# define the ansatz circuit
def V_T():
    qml.DoubleExcitation(0.12, wires=[0, 1, 2, 3])