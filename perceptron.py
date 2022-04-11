# Averaged Perceptron

import numpy as np

def positive(x, th, th0):
   return np.sign(th.T@x + th0)

def averaged_perceptron(data, labels, params = {}, hook = None):

    T = params.get('T', 100)
    (d, n) = data.shape

    theta = np.zeros((d, 1))
    theta_0 = np.zeros((1, 1))
    theta_s = np.zeros((d, 1))
    theta_0s = np.zeros((1, 1))
    for t in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if y * positive(x, theta, theta_0) <= 0.0:
                theta = theta + y * x
                theta_0 = theta_0 + y
            theta_s += theta
            theta_0s += theta_0
    return theta_s/(n*T), theta_0s/(n*T)


# Evaluating a classifier

def eval_classifier(learner, data_train, labels_train, data_test, labels_test):
    theta, theta_0 = learner(data_train, labels_train)
    #return score(data_test, labels_test, theta, theta_0)


# Evaluating algorithm
def eval_learning_alg(learner, data_gen, n_train, n_test, it):

    train_data, train_labels = data_gen(n_train)
    score = 0

    for i in range(it):
        test_data, test_labels = data_gen(n_test)
        score += eval_classifier(learner, train_data, train_labels, test_data, test_labels)

    return score / it

# Cross validation
def xval_learning_alg(learner, data, labels, k):
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                              data_test, labels_test)
    return score_sum/k