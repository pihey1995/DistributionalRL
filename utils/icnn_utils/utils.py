import numpy as np
from scipy.optimize import minimize
import torch

# bundle methods by Pierre-Alexandre Kamienny

def logistic(x):
    return 1. / (1. + np.exp(-x))

def logexp1p(x):
    """ Numerically stable log(1+exp(x))"""
    y = np.zeros_like(x)
    I = x>1
    y[I] = np.log1p(np.exp(-x[I]))+x[I]
    y[~I] = np.log1p(np.exp(x[~I]))
    return y

def proj_newton(A,b,lam0=None):
    """ minimize_{lam>=0, sum(lam)=1} -(A*1 + b)^T*lam + sum(log(1+exp(A^T*lam)))"""
    k = A.shape[0]
    c = np.sum(A,axis=1, keepdims=True) + b

    cons = [{"type": "ineq", "fun": lambda x: x[i]} for i in range(k)] + [{"type": "ineq", "fun": lambda x: 1-x[i]} for i in range(k)] + [{"type": "ineq", "fun": lambda x: np.sum(x) - 1},{"type": "ineq", "fun": lambda x: 1-np.sum(x)}]
    def f(lam):
        ATlam = A.T.dot(lam)
        return (-c.T.dot(lam)[0] + np.sum(logexp1p(ATlam)))

    def grad(lam):
        ATlam = np.expand_dims(A.T.dot(lam),0)
        z = 1/(1+np.exp(-ATlam)) ##z should be size n (action size)
        return (-c + A.dot(z))

    def hess(lam):
        ATlam = A.T.dot(lam)
        z = 1/(1+np.exp(-ATlam))
        return (A*(z*(1-z))).dot(A.T)

    lam_f = minimize(fun=f, x0=np.ones((k,1))/k, method="SLSQP", jac=grad, hess=hess, constraints=cons)
    return lam_f["x"]

def proj_newton2(A,b,lam0=None):
    """ minimize_{lam>=0, sum(lam)=1} -(A*1 + b)^T*lam + sum(log(1+exp(A^T*lam)))"""
    k,d = A.shape
    c = A.dot(np.ones((d,1))) + b #np.sum(A, 1)

    cons = [{"type": "ineq", "fun": lambda x: x[i]} for i in range(k)] + [{"type": "ineq", "fun": lambda x: 1-x[i]} for i in range(k)] + [{"type": "ineq", "fun": lambda x: np.sum(x) - 1},{"type": "ineq", "fun": lambda x: 1-np.sum(x)}]
    def f(lam):
        ATlam = A.T.dot(lam)
        return (-np.dot(c.transpose(), lam)[0] + np.sum(logexp1p(ATlam)))

    def grad(lam):
        ATlam = A.T.dot(lam)
        z = 1/(1+np.exp(-ATlam))
        return (-c + A.dot(np.expand_dims(z,1)))

    def hess(lam):
        ATlam = A.T.dot(lam)
        z = 1/(1+np.exp(-ATlam))
        return (A*(z*(1-z))).dot(A.T)

    lam_f = minimize(fun=f, x0=np.ones((k,1))/k, method="SLSQP", jac=grad, hess=hess, constraints=cons)
    return lam_f["x"]