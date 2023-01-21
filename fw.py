import numpy as np
from scipy.optimize import line_search

# maximal clique problem
# target function is xT*A*x + some regularization term
# with x being an n-dimensional simplex
# (feasible set consists of probability simplexes)
# and A - a {0,1} graph adjacency matrix

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

def frankwolfe(
    x_0: float, 
    grad=maxclique_grad, 
    lmo=maxclique_lmo,
    max_iter: int = 10000, 
    tol: float = 1e-3, 
    stepsize: float = None
    ):
    '''
    Basic Frank-Wolfe algorithm.

    x_0: starting point;
    grad: gradient f-n;
    lmo: linear oracle;
    max_iter: max. nr of iterations;
    stepsize: defaults to 2/(k+2), step defined as (1-stepsize)*x_k + stepsize*x_k_hat. Linesearch: TODO;
    '''

    x_hist = [x_0]
    s_hist = [x_0]
    for k in range(max_iter):
        g = grad(x=x_hist[-1])
        s = lmo(g)
        if (x_hist[-1] - s) @ g < tol:
            #print(f'Stopped by condition at {k}')
            break
        gamma = 2/(k+2) if stepsize is None else stepsize
        x_next = (1-gamma)*x_hist[-1] + gamma*s
        x_hist.append(x_next)
        s_hist.append(s)
    return x_hist, s_hist, k


def frankwolfe_awaysteps(
    x_0: float, 
    f=maxclique_target, 
    grad=maxclique_grad, 
    lmo=maxclique_lmo, 
    max_iter: int = 10000, 
    tol: float = 1e-3, 
    stepsize: float = None):

    x_hist = [x_0]
    s_t = [x_0]
    weights = dict()
    weights[np.where(x_0)[0][0]] = 1
    for k in range(max_iter):
        xt = x_hist[-1]
        g = grad(x=xt)
        s = lmo(g)
        # find fw direction
        d_fw = s - xt
        # check the FW gap
        gap_fw = -d_fw @ g
        if gap_fw < tol:
            break
        # find away-step direction
        v = s_t[np.argmax([g @ v for v in s_t],axis=0)]
        d_as = xt - v
        # choose the direction
        if gap_fw >= -g @ d_as:
            use_fw = True
            gamma_max = 1
            d = d_fw
        else:
            use_fw = False
            a_v =  weights[np.where(v)[0][0]]
            gamma_max = a_v/(1-a_v)
            d = d_as
        # Line Search for stepsize
        gamma, _, _ ,_ ,_ ,_ = line_search(f, grad, xt, d, g, extra_condition=lambda a, x, f, g: a <= gamma_max)
        x_next = xt + gamma*d
        if use_fw:
            # update weights for s
            if np.where(s)[0][0] in weights.keys():
                weights[np.where(s)[0][0]] += gamma
            else:
                s_t.append(s)
                weights[np.where(s)[0][0]] = gamma
        else:
            # update weights for v
            weights[np.where(v)[0][0]] -= gamma
        x_hist.append(x_next)
    return x_hist, s_t, k


def frankwolfe_pairwise(
    x_0: float, 
    f=maxclique_target, 
    grad=maxclique_grad, 
    lmo=maxclique_lmo, 
    max_iter: int = 10000, 
    tol: float = 1e-3, 
    stepsize: float = None):

    x_hist = [x_0]
    s_t = [x_0]
    weights = dict()
    weights[np.where(x_0)[0][0]] = 1
    for k in range(max_iter):
        xt = x_hist[-1]
        g = grad(x=xt)
        s = lmo(g)
        # find fw direction
        d_fw = s - xt
        # check the FW gap
        gap_fw = -d_fw @ g
        if gap_fw < tol:
            break
        # find away-step direction
        v = s_t[np.argmax([g @ v for v in s_t],axis=0)]
        # combine directions
        a_v =  weights[np.where(v)[0][0]]
        d = s - v
        # Line Search
        gamma, _, _ ,_ ,_ ,_ = line_search(f, grad, xt, d, g, extra_condition=lambda a, x, f, g: a <= a_v)
        x_next = xt + gamma*d
        # update weights for s
        if np.where(s)[0][0] in weights.keys():
            weights[np.where(s)[0][0]] += gamma
        else:
            s_t.append(s)
            weights[np.where(s)[0][0]] = gamma
        # update weights for v
        weights[np.where(v)[0][0]] -= gamma
        x_hist.append(x_next)
    return x_hist, s_t, k