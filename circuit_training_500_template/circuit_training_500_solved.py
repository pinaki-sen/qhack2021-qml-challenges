#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    
    Y_test = [1,0,-1,0,-1,1,-1,-1,0,-1,1,-1,0,1,0,-1,-1,0,0,1,1,0,-1,0,0,-1,0,-1,0,0,1,1,-1,-1,-1,0,-1,0,1,0,-1,1,1,0,-1,-1,-1,-1,0,0]

    
    def get_angles(x):

        beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
        beta1 = -2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[2] ** 2 + x[1] ** 2 + 1e-12))
        beta2 = 2 * np.arcsin(np.sqrt(x[2] ** 2 + x[1] ** 2)/np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 ))

        return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

        # beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
        # beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
        # beta2 = 2 * np.arcsin(
        #     np.sqrt(x[2] ** 2 + x[3] ** 2)
        #     / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
        # )

        # return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

        
    def statepreparation_1(a):
        # qml.RY(a[0], wires=0)

        # qml.CNOT(wires=[0, 1])
        # qml.RY(a[1], wires=1)
        # qml.CNOT(wires=[0, 1])
        # qml.RY(a[2], wires=1)

        # qml.PauliX(wires=0)
        # qml.CNOT(wires=[0, 2])
        # qml.RY(a[3], wires=2)
        # qml.CNOT(wires=[0, 2])
        # qml.RY(a[4], wires=2)
        # qml.PauliX(wires=0)



        qml.RY(a[0], wires=0)

        qml.CNOT(wires=[0, 1])
        qml.RY(a[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[2], wires=1)

        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[3], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(a[4], wires=1)
        qml.PauliX(wires=0)


    def layer_1(W):
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        # qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
        
        qml.CNOT(wires=[0, 1])
        # qml.CNOT(wires=[0, 2])
        # qml.CNOT(wires=[1, 2])
        

    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev)
    def circuit_1(weights, x=None):
        # Feature mapping
        angle = get_angles(x)
        statepreparation_1(angle)
        # variational classifier
        for w in weights:
            layer_1(w)
        
        return qml.expval(qml.PauliZ(1))

    def classifier_training_1(params, x=None, y=None):
        weights = params[0]
        bias = params[1]

        out_probs = circuit_1(weights, x=x) + bias
        return (out_probs-y)**2


    def classifier_prediction_1(params, x=None):
        weights = params[0]
        bias = params[1]

        out_probs = circuit_1(weights, x=x) + bias

        # if(out_probs>0.3):
        #     return 1
        # elif(out_probs<0.3 and out_probs>-0.3):
        #     return 0
        # elif(out_probs<-0.3):
        #     return -1
        if(out_probs>0):
            return 1
        else:
            return -1

    def circuit_output_test(params, x=None):
        weights = params[0]
        bias = params[1]

        out_probs = circuit(weights, x=x) + bias

        return out_probs

    def cost_1(params, X, Y):

        y_pred = np.array([classifier_training_1(params, x=X[i], y=Y[i]) for i in range(len(Y))])

        cost = np.sum(y_pred) / len(Y)
        return cost


    def accuracy_1(params, x_train, y_train, iter):

        y_pred_train = np.array([classifier_prediction_1(params, x=x) for x in x_train])
        acc_train = np.sum(y_pred_train==y_train) / len(y_train)

        # y_pred_test = np.array([classifier_prediction(params, x=x) for x in x_test])
        # acc_test = np.sum(y_pred_test==y_test) / len(y_test)

        # print("Iter=> {}   train_cost=> {}   train_acc=> {}   test_acc=> {}".format(iter+1, cost(params, x_train, y_train), acc_train, acc_test))
        print("Iter=> {}   train_cost=> {}   train_acc=> {} ".format(iter+1, cost_1(params, x_train, y_train), acc_train))

    

    ###################
    ###################
    ###################

    def statepreparation_2(a):
        
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        
        qml.RZ(a[0], wires=0)
        qml.RZ(a[1], wires=1)
        qml.CNOT(wires=[0,1])
        qml.RZ(a[0]*a[1], wires=1)
        qml.CNOT(wires=[0,1])

        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        
        qml.RZ(a[1], wires=0)
        qml.RZ(a[2], wires=1)
        qml.CNOT(wires=[0,1])
        qml.RZ(a[1]*a[2], wires=1)
        qml.CNOT(wires=[0,1])
        


        # qml.RY(a[0], wires=0)

        # qml.CNOT(wires=[0, 1])
        # qml.RY(a[1], wires=1)
        # qml.CNOT(wires=[0, 1])
        # qml.RY(a[2], wires=1)

        # qml.PauliX(wires=0)
        # qml.CNOT(wires=[0, 1])
        # qml.RY(a[3], wires=1)
        # qml.CNOT(wires=[0, 1])
        # qml.RY(a[4], wires=1)
        # qml.PauliX(wires=0)


    def layer_2(W):
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        # qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
        
        qml.CNOT(wires=[0, 1])
        # qml.CNOT(wires=[0, 2])
        # qml.CNOT(wires=[1, 2])
        

    # dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev)
    def circuit_2(weights, x=None):
        # Feature mapping
        # angle = get_angles(x)
        statepreparation_2(x)
        
        # variational classifier
        for w in weights:
            layer_2(w)
        
        return qml.expval(qml.PauliZ(1))

    def classifier_training_2(params, x=None, y=None):
        weights = params[0]
        bias = params[1]

        out_probs = circuit_2(weights, x=x) + bias
        return (out_probs-y)**2


    def classifier_prediction_2(params, x=None):
        weights = params[0]
        bias = params[1]

        out_probs = circuit_2(weights, x=x) + bias

        # if(out_probs>0.3):
        #     return 1
        # elif(out_probs<0.3 and out_probs>-0.3):
        #     return 0
        # elif(out_probs<-0.3):
        #     return -1
        if(out_probs>0):
            return 1
        else:
            return -1

    def cost_2(params, X, Y):

        y_pred = np.array([classifier_training_2(params, x=X[i], y=Y[i]) for i in range(len(Y))])

        cost = np.sum(y_pred) / len(Y)
        return cost

    
    
    
    def accuracy_2(params, x_train, y_train, iter):

        y_pred_train = np.array([classifier_prediction_2(params, x=x) for x in x_train])
        acc_train = np.sum(y_pred_train==y_train) / len(y_train)

        # y_pred_test = np.array([classifier_prediction(params, x=x) for x in x_test])
        # acc_test = np.sum(y_pred_test==y_test) / len(y_test)

        # print("Iter=> {}   train_cost=> {}   train_acc=> {}   test_acc=> {}".format(iter+1, cost(params, x_train, y_train), acc_train, acc_test))
        print("Iter=> {}   train_cost=> {}   train_acc=> {} ".format(iter+1, cost_2(params, x_train, y_train), acc_train))
       


    #################################################################
    #################################################################

    X_train_1 = np.copy(X_train)
    Y_train_1 = []
    for i in range(len(Y_train)):
        if(Y_train[i]==0 or Y_train[i]==-1):
            Y_train_1.append(-1)
        elif(Y_train[i]==1):
            Y_train_1.append(1)
    Y_train_1 = np.array(Y_train_1)


    # X_train_2 = np.copy(X_train)
    # Y_train_2 = []
    # for i in range(len(Y_train)):
    #     if(Y_train[i]==-1 or Y_train[i]==1):
    #         Y_train_2.append(1)
    #     elif(Y_train[i]==0):
    #         Y_train_2.append(-1)
    # Y_train_2 = np.array(Y_train_2)


    # X_train_3 = np.copy(X_train)
    # Y_train_3 = []
    # for i in range(len(Y_train)):
    #     if(Y_train[i]==0 or Y_train[i]==1):
    #         Y_train_3.append(1)
    #     elif(Y_train[i]==-1):
    #         Y_train_3.append(-1)
    # Y_train_3 = np.array(Y_train_3)





    X_train_2 = []
    Y_train_2 = []
    for i in range(len(Y_train)):
        if(Y_train[i]==0):
            X_train_2.append(X_train[i])
            Y_train_2.append(1)
        elif(Y_train[i]==-1):
            X_train_2.append(X_train[i])
            Y_train_2.append(-1)

    X_train_2 = np.array(X_train_2)
    Y_train_2 = np.array(Y_train_2)
    
    


    
    # num_qubits = 2
    # num_layers = 2
    # params = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0)

    params_1 = (0.01 * np.random.randn(2, 2, 3), 0.0)
    params_2 = (0.01 * np.random.randn(2, 2, 3), 0.0)
    # params_3 = (0.01 * np.random.randn(2, 2, 3), 0.0)
    

    
    # optimizing first model
    iters = 5
    optimizer_1 = qml.NesterovMomentumOptimizer(stepsize=0.1)
    
    for iter in range(iters):
        params_1 = optimizer_1.step(lambda v: cost_1(v, X_train_1, Y_train_1), params_1)

        # accuracy_1(params_1, X_train_1, Y_train_1, iter)

    # print()
    # optimizing second model
    iters = 10
    optimizer_2 = qml.NesterovMomentumOptimizer(stepsize=0.1)
    
    for iter in range(iters):
        params_2 = optimizer_2.step(lambda v: cost_2(v, X_train_2, Y_train_2), params_2)

        # accuracy_2(params_2, X_train_2, Y_train_2, iter)

    # print()
    # # optimizing third model
    # iters = 10
    # optimizer_3 = qml.NesterovMomentumOptimizer(stepsize=0.1)
    
    # for iter in range(iters):
    #     params_3 = optimizer_3.step(lambda v: cost(v, X_train_3, Y_train_3), params_3)

    #     accuracy(params_3, X_train_3, Y_train_3, iter)


    # Y_pred = np.array([classifier_prediction(params, x=X_test[i]) for i in range(len(X_test))])
    

    # flag_circuit_out = np.array([circuit_output_test(params_2, x=X_train_2[i]) for i in range(len(X_train_2))])

    # for i in range(len(Y_train_2)):
    #     print(flag_circuit_out[i], Y_train_2[i])

    Y_pred = []
    for i in range(len(X_test)):
        tmp = classifier_prediction_1(params_1, x=X_test[i])
        if(tmp == 1):
            Y_pred.append(1)
        else:
            tmp = classifier_prediction_2(params_2, x=X_test[i])
            if(tmp == 1):
                Y_pred.append(0)
            elif(tmp==-1):
                Y_pred.append(-1)
    
    Y_pred = np.array(Y_pred)

    # Y_pred = []
    # for i in range(len(X_test)):
    #     tmp = classifier_prediction(params_1, x=X_test[i])
    #     if(tmp==1):
    #         Y_pred.append(1)
    #     else:
    #         tmp = classifier_prediction(params_2, x=X_test[i])
    #         if(tmp==-1):
    #             Y_pred.append(0)
    #         else:
    #             tmp = classifier_prediction(params_3, x=X_test[i])
    #             if(tmp==-1):
    #                 Y_pred.append(-1)
    #             else:
    #                 Y_pred.append(0) #worst case

    # Y_pred = np.array(Y_pred)

    # print("Final_Accuracy => ", np.sum(Y_pred==Y_test)/len(Y_test))

    predictions = Y_pred
    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
