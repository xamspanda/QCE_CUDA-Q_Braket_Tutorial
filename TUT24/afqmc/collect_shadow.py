import os, boto3
import numpy as np
import pennylane as qml
from braket.jobs import save_job_result
from braket.jobs.metrics import log_metric
from braket.jobs.environment_variables import get_hyperparameters
from afqmc.utils.matchgate import gaussian_givens_decomposition

sfn_client = boto3.client("stepfunctions")

num_qubits = 4

def main() -> None:
    print("Hybrid job started.")
    hyperparameters = get_hyperparameters()
    task_token = hyperparameters.get("TaskToken", None)
    print(f"Task token: {task_token}")

    try:
        print("Perform shadow tomography")
        shadow_size = int(hyperparameters["ShadowSize"])
        shots = int(hyperparameters["Shots"])
        
        dev = get_pennylane_device(n_wires=num_qubits, shots=shots)
        print("The device is successfully loaded.")

        @qml.qnode(dev)
        def hydrogen_shadow_circuit(Q):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.DoubleExcitation(0.12, wires=[0, 1, 2, 3])
            gaussian_givens_decomposition(Q)
            return qml.counts()

        q_list = []
        q_save = []
        for _ in range(shadow_size):
            q, signed_per = random_signed_permutation(2*num_qubits)
            q_save.append(signed_per.tolist())
            q_list.append(q)
        print("The random signed matrices are successfully generated.")

        output = []
        for ns in range(len(q_list)):
            # for each snapshot, add a random matchgate circuit
            # every output from sv1 is a 0-dimensional numpy ndarray
            counts = hydrogen_shadow_circuit(q_list[ns])
            counts = {key: int(value) for key, value in counts.items()}
            output.append(counts)
            
            log_metric(metric_name="remaining tasks", iteration_number=ns, value=shadow_size-ns-1)

        print('All the measurements are completed.')

        # savings require JSON serializable object
        save_job_result(
            {
                "output": output,
                "Q_save": q_save,
            },
        )
        if task_token:
            print("Sending task success to Step Functions...")
            sfn_client.send_task_success(taskToken=task_token, output="{}")

    except Exception as e:
        print(e)
        if task_token:
            print("Sending task failure to Step Functions...")
            sfn_client.send_task_failure(taskToken=task_token, output="{}")

    finally:
        print("Hybrid job completed.")


def random_signed_permutation(size):
    """Generating size 2n signed permutation matrix Q, from Borel group B(2n).
    This will save matchgate circuit depth compared to Orthogonal group;
    """
    q = np.zeros((size, size))
    permutation = np.random.permutation(size)
    save = np.array([i+1 for i in permutation])
    sign = np.random.randint(2, size=size)
    
    for i in range(size):
        q[permutation[i], i] = (-1)**sign[i]
        save[i] *= (-1)**sign[i]
    return q, save


def get_pennylane_device(n_wires: int, shots: int) -> qml.device:
    """Create Pennylane device from the `device` keyword argument of AwsQuantumJob.create().
    See https://docs.aws.amazon.com/braket/latest/developerguide/pennylane-embedded-simulators.html
    about the format of the `device` argument.

    Args:
        n_wires (int): number of qubits to initiate the local simulator.

    Returns:
        device: The Pennylane device
    """
    device_string = os.environ["AMZN_BRAKET_DEVICE_ARN"]
    prefix, device_name = device_string.split("/")
    device = qml.device(device_name, wires=n_wires, shots=shots)
    print("Using simulator: ", device.name)
    return device