import numpy as np
import math

def length(v):
    sum = 0
    d = v.shape[0]
    for i in range(d):
        sum += v[i][0] ** 2
    return math.sqrt(sum)

def signed_dist(x, y, th, th0):
    return y * (np.dot(th.T, x) + th0) / length(th)


def Margin_Task(data, labels, blue_th, blue_th0, red_th, red_th0):

    blue_res = []
    red_res = []

    for i in range(data.shape[1]):
        blue_res.append(signed_dist(data[:, i], labels[:, i], blue_th, blue_th0))
        red_res.append(signed_dist(data[:, i], labels[:, i], red_th, red_th0))

    return {"Blue separator": [sum(blue_res), min(blue_res), max(blue_res)],
            "Red separator": [sum(red_res), min(red_res), max(red_res)]}


# gradient descent
def gd(f, df, x0, step_size_fn, max_iter):
    prev_x = x0
    fs = []; xs = []
    for i in range(max_iter):
        prev_f, prev_grad = f(prev_x), df(prev_x)
        fs.append(prev_f); xs.append(prev_x)
        if i == max_iter-1:
            return prev_x, fs, xs
        step = step_size_fn(i)
        prev_x = prev_x - step * prev_grad

# numerical gradient
def num_grad(f, delta=0.001):
    def df(x):
        d = x.shape
        res = []
        for i in range(d[0]):
            delta_i = np.zeros(shape=d)
            delta_i[i][0] = delta
            res.append([(f(x + delta_i) - f(x - delta_i)) / (2 * delta)])

        res = np.array(res)
        return res

    return df

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def hinge(v):
    return np.max(np.array([0, 1 - v], dtype='object'))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    d, n = x.shape
    res = []
    for i in range(n):
        h = (y[:, i] * (th.T@x[:, i] + th0))
        if h < 1:
            res.append(1 - h)
        else:
            res.append(0)

    return sum(res)

# SVM objective
def svm_obj(x, y, th, th0, lam):
    d, n = x.shape
    return ((1/n) * hinge_loss(x, y, th, th0) + (lam * length(th)))

# SVM gradient
def d_hinge(v):
    return np.where(v >= 1, 0, -1)

def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T, x) + th0))* y * x

def d_hinge_loss_th0(x, y, th, th0):
    return d_hinge(y*(np.dot(th.T, x) + th0)) * y

def d_svm_obj_th(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th(x, y, th, th0), axis = 1, keepdims = True) + lam * 2 * th

def d_svm_obj_th0(x, y, th, th0, lam):
    return np.mean(d_hinge_loss_th0(x, y, th, th0), axis = 1, keepdims = True)

def svm_obj_grad(X, y, th, th0, lam):
    grad_th = d_svm_obj_th(X, y, th, th0, lam)
    grad_th0 = d_svm_obj_th0(X, y, th, th0, lam)
    return np.vstack([grad_th, grad_th0])

# Batch SVM minimize
def batch_svm_min(data, labels, lam):
    def svm_min_step_size_fn(i):
       return 2/(i+1)**0.5
    init = np.zeros((data.shape[0] + 1, 1))

    def f(th):
      return svm_obj(data, labels, th[:-1, :], th[-1:,:], lam)

    def df(th):
      return svm_obj_grad(data, labels, th[:-1, :], th[-1:,:], lam)

    x, fs, xs = gd(f, df, init, svm_min_step_size_fn, 10)
    return x, fs, xs