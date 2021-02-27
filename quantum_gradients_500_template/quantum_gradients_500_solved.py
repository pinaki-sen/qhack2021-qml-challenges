#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #

    gradient = np.zeros([6], dtype=np.float64)

    # gradient
    s = 0.1
    for i in range(6):
        w = np.copy(params)
        w[i] += s
        exp_p = qnode(w)
        w[i] -= 2*s
        exp_m = qnode(w)
        gradient[i] = (exp_p - exp_m) / (2*np.sin(s))

    # Fubini-Study metric
    @qml.qnode(dev)
    def my_qnode(params):
        variational_circuit(params)
        return qml.state()

    s = np.pi/2

    base_state = my_qnode(params)
    conj_base_state = np.conj(base_state)

    met = np.zeros([6, 6], dtype=np.float64)

    for k in range(6):
        for l in range(6):
                w_pp = np.copy(params)
                w_pp[k] += s
                w_pp[l] += s

                w_pm = np.copy(params)
                w_pm[k] += s
                w_pm[l] -= s

                w_mp = np.copy(params)
                w_mp[k] -= s
                w_mp[l] += s

                w_mm =  np.copy(params)
                w_mm[k] -= s
                w_mm[l] -= s

                expval_pp = my_qnode(w_pp)
                pp = np.abs(np.dot(conj_base_state, expval_pp)) ** 2

                expval_pm = my_qnode(w_pm)
                pm = np.abs(np.dot(conj_base_state, expval_pm)) ** 2

                expval_mp = my_qnode(w_mp)
                mp = np.abs(np.dot(conj_base_state, expval_mp)) ** 2

                expval_mm = my_qnode(w_mm)
                mm = np.abs(np.dot(conj_base_state, expval_mm)) ** 2
                
                met[k][l] = (-pp + pm + mp - mm) / 8

    
    #print(np.round(qml.metric_tensor(qnode)(params), 8))
    #print(F)

    natural_grad = np.matmul(np.linalg.inv(met), gradient)

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
