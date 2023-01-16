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
    dp = np.linalg.norm(x, ord=2)
    if str.lower(penalty) == 'f1':
        raise NotImplementedError
    if str.lower(penalty) == 'f2':
        if beta <= 0 or alpha < 0 or alpha > (2/beta^2):
            raise ValueError
        dp = alpha*np.sum(-beta*np.exp(-beta*x))
    return 2*np.dot(A, x) + dp

def maxclique_lmo(x, grad):
    """
    LMO for the maximal clique problem. 
    The feasible set is the unit simplex, so return a [0,...,1,...,0] vector with 1 in place of the lowest element of the gradient.
    """
    e = np.zeros_like(x)
    if grad is callable:
        idxmin = np.argmin(grad(x))
    elif grad is np.ndarray:
        idxmin = np.argin(grad(x))
    e[idxmin] = 1
    return e

def frankwolfe(stepsize: float, domain: np.ndarray, gr:callable, lmo: callable, max_iter: int, x_0:float):
    '''
    Basic Frank-Wolfe algorithm.

    stepsize: defaults to 2/(k+2), step defined as (1-stepsize)*x_k + stepsize*x_k_hat
    domain: feasible set;
    g: gradient f-n;
    max_iter: max. nr of iterations,
    x_0: starting point, defaults to a random element of the feasible set.
    '''
    x_hist = []
    s_hist = []
    # random starting point from the domain
    if x_0 is None:
        x_0  = np.random.rand()*(domain.shape[0] - 1)
    for k in range(max_iter):
        x_opt = lmo(x_hist[-1], gr)
        gamma = 2/(k+1) if stepsize is None else stepsize
        x_next = (1-gamma)*x_hist[-1] + gamma*x_opt

        x_hist.append(x_next)
        s_hist.append(x_opt)
    return x_hist, s_hist
