import numpy as np
from scipy.optimize import line_search
from warnings import warn

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

def _maxclique_grad(A: np.ndarray, x: np.ndarray, penalty: str = 'l2', p: int = 0.5, alpha: float = None, beta: float = None, eps: float = None):
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

def _maxclique_lmo(grad):
    """
    LMO for the maximal clique problem. 
    The feasible set is the unit simplex, so return a [0,...,1,...,0] vector 
    with 1 in place of the lowest element of the gradient.
    """

    e = np.zeros(shape=grad.shape[-1])
    idxmin = np.argmin(grad)
    e[idxmin] = 1
    return e


def linesearch(f, g, d, xk, fk, gk, max_step=1, A=None, c1=1e-4, c2=0.9, maxiter=10000):
    """
    Line-Search algorithm for strong Frank-Wolfe conditions.
    Source: Wright and Nocedal, 'Numerical Optimization', 1999, p. 59-61

    f: target f-n;
    g: gradient of f;
    d: descent direction;
    xk: starting point;
    fk: value of target f-n at xk;
    gk: value of gradient at xk;
    max_step: maximum stepsize;
    A: matrix for the maxclique problem, used to compute alpha analytically for l2 penalty;
    """

    def phi(alpha): return f(x=xk+alpha*d)
    
    def phi_prime(alpha): return g(x=xk+alpha*d) @ d

    # if we have an L2 penalty, can compute the optimal stepsize analytically
    # by setting the derivative w.r.t. alpha to 0
    if not (A is None):
        return (gk.T @ d) / (d.T @ A @ d)
    # for different norms, use a numerical algorithm
    else:
        a = [0, max_step/2]
        i = 1
        phi_prev = fk
        for _ in range(maxiter):
            ai = a[i]
            phi_ai = phi(ai)
            if phi_ai > fk + c1*ai*gk or (phi_ai > phi_prev and i > 1):
                a_star = _ls_zoom(a[i-1], ai)
                return a_star
            phi_prime_ai = phi_prime(ai)
            if np.abs(phi_prime_ai) <= -c2 * gk:
                a_star = ai
                return a_star
            if phi_prime_ai >= 0:
                a_star = _ls_zoom(ai, a[i-1])
                return a_star
            a.append(ai*2)
        warn(f'Line search did not converge in maxiter={maxiter} iterations', RuntimeWarning)
        return ai

def _ls_zoom(a, b):
    raise NotImplementedError


def frankwolfe(
    x_0: float, 
    grad=_maxclique_grad, 
    lmo=_maxclique_lmo,
    max_iter: int = 10000, 
    tol: float = 1e-3, 
    stepsize: float = None
    ):
    """
    Basic Frank-Wolfe algorithm.

    x_0: starting point;
    grad: gradient f-n;
    lmo: linear oracle;
    max_iter: max. nr of iterations;
    stepsize: defaults to 2/(k+2), step defined as (1-stepsize)*x_k + stepsize*x_k_hat. Linesearch: TODO;
    """

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
    f,
    grad, 
    lmo,
    linesearch,
    max_iter: int = 10000, 
    tol: float = 1e-6, 
    stepsize: float = None):
    """
    Away-Steps Frank-Wolfe algorithm.

    x_0: starting point;
    f: target f-n;
    grad: gradient f-n;
    lmo: linear oracle;
    linesearch: line-search function;
    max_iter: max. nr of iterations;
    stepsize: defaults to 2/(k+2), step defined as (1-stepsize)*x_k + stepsize*x_k_hat. Linesearch: TODO;
    """

    x_hist = [x_0]
    s_hist = [x_0]
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
        v = s_hist[np.argmax([g @ v for v in s_hist],axis=0)]
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
        #gamma, _, _ ,_ ,_ ,_ = line_search(f, grad, xt, d, g, extra_condition=lambda a, x, f, g: a <= gamma_max)
        #gamma, _, _ ,_ ,_ ,_ = line_search(f, grad, xt, d, g)
        gamma = linesearch(grad=g,d=d)
        x_next = xt + gamma*d
        # update weights
        if use_fw:
            s_ind = np.where(s)[0][0]
            if gamma == 1:
            # if gamma is 1 (we removed all previous x's), we dont need any other vertices
                s_hist = [s]
                weights = {s_ind: 1}
            elif s_ind in weights.keys():
            # else if we already used s
                weights.update((vertex, weight*(1-gamma)) for vertex, weight in weights.items())
                weights[s_ind] += gamma
            else:
            # else if first encounter of s
                s_hist.append(s)
                weights.update((vertex, weight*(1-gamma)) for vertex, weight in weights.items())
                weights[s_ind] = gamma
        else:
            v_ind = np.where(v)[0][0]
            if gamma == gamma_max:
            # if we completely removed v from x
                weights.pop(v_ind)
                s_hist.pop(v)
            else:
                weights.update((vertex, weight*(1+gamma)) for vertex, weight in weights.items())
                weights[v_ind] -= gamma
        x_hist.append(x_next)
    return x_hist, s_hist, k


def frankwolfe_pairwise(
    x_0: float, 
    f=maxclique_target, 
    grad=_maxclique_grad, 
    lmo=_maxclique_lmo, 
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