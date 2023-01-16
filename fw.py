import numpy as np

# maximal clique problem
# target function is xT*A*x + some regularization term
# with x being an n-dimensional simplex
# (feasible set consists of probability simplexes)
# and A - a {0,1} graph adjacency matrix

def maxclique_grad(A: np.ndarray, x: np.ndarray, penalty: str = 'l2', p: int = None, alpha: float = None, beta: float = None, eps: float = None):
    """
    Gradient calculation for the Maximal Clique.

    A: a R^NxN adjacency matrix;
    x: a R^N vector;
    penalty: {'l2', 'f1', 'f2'} string specifying the type of penalty to use; defaults to l2;
    p, alpha, beta, eps: penalty parameters.
    """
    dp = 2*x
    if str.lower(penalty) == 'f1':
        raise NotImplementedError
    if str.lower(penalty) == 'f2':
        if beta <= 0 or alpha < 0 or alpha >= (2/beta**2):
            raise ValueError
        dp = alpha*np.sum(-beta*np.exp(-beta*x))

    return np.dot((A + A.T), x) + dp

def maxclique_lmo(grad):
    """
    LMO for the maximal clique problem. 
    The feasible set is the unit simplex, so return a [0,...,1,...,0] vector with 1 in place of the lowest element of the gradient.
    """
    e = np.zeros(shape=grad.shape[-1])
    idxmin = np.argmin(grad)
    e[idxmin] = 1
    return e

def frankwolfe(A: np.ndarray, x_0:float, grad: callable=maxclique_grad, lmo: callable=maxclique_lmo, max_iter: int=1000, stepsize: float = None):
    '''
    Basic Frank-Wolfe algorithm.

    A: graph adjacency matrix;
    g: gradient f-n;
    stepsize: defaults to 2/(k+2), step defined as (1-stepsize)*x_k + stepsize*x_k_hat;
    max_iter: max. nr of iterations,
    x_0: starting point.
    '''
    x_hist = [x_0]
    s_hist = []
    for k in range(max_iter):
        s = lmo(grad(A, x_hist[-1]))
        gamma = 2/(k+2) if stepsize is None else stepsize
        x_next = (1-gamma)*x_hist[-1] + gamma*s

        x_hist.append(x_next)
        s_hist.append(s)
    return x_hist, s_hist