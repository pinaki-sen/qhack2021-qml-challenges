#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    s = np.pi / 2

    expval_plus_list = []
    expval_minus_list = []
    
    
    for k in range(5):
        w_plus = np.copy(weights)
        w_plus[k] += s
        expval_plus = circuit(w_plus)
        expval_plus_list.append(expval_plus)

        w_minus = np.copy(weights)
        w_minus[k] -= s
        expval_minus = circuit(w_minus)
        expval_minus_list.append(expval_minus)
        
        gradient[k] = (expval_plus - expval_minus) / (2*np.sin(s))
    

    # normal_res = circuit(weights)
    # s = np.pi / 4

    for k in range(5):
        for l in range(k+1):
            #if (l<=k):
            #print(k,l)
            w_pp = np.copy(weights)
            w_pp[k] += s
            w_pp[l] += s

            w_pm = np.copy(weights)
            w_pm[k] += s
            w_pm[l] -= s

            w_mp = np.copy(weights)
            w_mp[k] -= s
            w_mp[l] += s

            w_mm =  np.copy(weights)
            w_mm[k] -= s
            w_mm[l] -= s

            #expval_pp = circuit(w_pp)
            #expval_pm = circuit(w_pm)
            #expval_mp = circuit(w_mp)
            #expval_mm = circuit(w_mm)
            if k == l:
                expval_pp = circuit(w_pp)
                expval_pm = circuit(w_pm)
                expval_mp = circuit(w_mp)
                expval_mm = expval_pp

            elif k == 2 or k == 4:
                expval_pp = circuit(w_pp)
                expval_pm = circuit(w_pm)
                expval_mp = -expval_pp
                expval_mm = -expval_pm
            elif k == 1:
                expval_pp = circuit(w_pp)
                expval_pm = circuit(w_pm)
                expval_mp = expval_pm
                expval_mm = expval_pp

            else:
                expval_pp = circuit(w_pp)
                expval_pm = circuit(w_pm)
                expval_mp = circuit(w_mp)
                expval_mm = circuit(w_mm)
                #print(expval_pp)
                #print(expval_pm)
                #print(expval_mp)
                #print(expval_mm)

                
            hessian[k][l] = (expval_pp + expval_mm - expval_mp - expval_pm) / (4*np.sin(s))


    # for k in range(5):
    #     expval_pp = expval_plus_list[k]
    #     expval_pm = normal_res
    #     expval_mp = normal_res
    #     expval_mm = expval_minus_list[k]

    #     hessian[k][k] = (expval_pp + expval_mm - expval_mp - expval_pm) / (4*np.sin(s))



    for k in range(5):
        for l in range(5):
            if (l>k):
                hessian[k][l] = hessian[l][k]


    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
