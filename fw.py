import numpy as np

# maximal clique problem
# target function is xT*A*x + some regularization term
# with x being an n-dimensional simplex
# (feasible set consists of probability simplexes)
# and A - a {0,1} graph adjacency matrix

#test
#test

def maxclique_target(A: np.ndarray, x:np.ndarray, p: int=0.5):
    '''
    Target function for the Maximal Clique Problem.
    '''

    return np.matmul(x.T,np.matmul(A,x)) + p*np.linalg.norm(x)**2

def maxclique_grad(A: np.ndarray, x: np.ndarray, penalty: str = 'l2', p: int = 0.5, alpha: float = None, beta: float = None, eps: float = None):
    """Gradient calculation for the Maximal Clique Problem.
    
    A: a R^NxN symmetric adjacency matrix;
    x: a R^N vector;
    penalty: {'l2', 'f1', 'f2'} string specifying the type of penalty to use; defaults to l2;
    p, alpha, beta, eps: penalty parameters.

    Note: in this problem, the goal is maximization, so the gradient must be multiplied by -1.
    """
    if str.lower(penalty) == 'l2':
        dp = 2*x*p
    elif str.lower(penalty) == 'f1':
        raise NotImplementedError
    elif str.lower(penalty) == 'f2':
        if beta <= 0 or alpha < 0 or alpha >= (2/beta**2):
            raise ValueError
        dp = alpha*np.sum(-beta*np.exp(-beta*x))
    else:
        raise ValueError('Unknown penalty: must be l2, f1 or f2')
    
    return 2 * A @ x + dp

def maxclique_lmo(grad):
    """
    LMO for the maximal clique problem. 
    The feasible set is the unit simplex, so return a [0,...,1,...,0] vector 
    with 1 in place of the lowest element of the gradient.
    """

    e = np.zeros(shape=grad.shape[-1])
    idxmin = np.argmin(grad)
    e[idxmin] = 1
    return e

def frankwolfe(x_0: float, grad=maxclique_grad, lmo=maxclique_lmo, max_iter: int = 10000, tol: float = 1e-5, stepsize: float = None):
    '''
    Basic Frank-Wolfe algorithm.

    x_0: starting point;
    grad: gradient f-n;
    lmo: linear oracle;
    max_iter: max. nr of iterations;
    stepsize: defaults to 2/(k+2), step defined as (1-stepsize)*x_k + stepsize*x_k_hat. Linesearch: TODO;
    '''

    x_hist = [x_0]
    s_hist = []
    for k in range(max_iter):
        g = grad(x=x_hist[-1])
        s = lmo(g)
        if np.abs((x_hist[-1] - s) @ g) < tol:
            break
        gamma = 2/(k+2) if stepsize is None else stepsize
        x_next = (1-gamma)*x_hist[-1] + gamma*s
        x_hist.append(x_next)
        s_hist.append(s)
    return x_hist, s_hist