import numpy as np
from warnings import warn

# maximal clique problem
# target function is xT*A*x + some regularization term
# with x being an n-dimensional simplex
# (feasible set consists of probability simplexes)
# and A - a {0,1} graph adjacency matrix

def maxclique_target(
    A: np.ndarray, x: np.ndarray, penalty: str = 'l2', p: int = 0.5,
    alpha: float = None, beta: float = None, eps: float = None
    ):
    """Target f-n for the Maximal Clique Problem.

    penalty: {'l2', 'f1', 'f2'} string specifying the type of penalty to use; defaults to l2;
    p, alpha, beta, eps: penalty parameters.

    Note: in this problem, the goal is maximization, so the f-n must be multiplied by -1.
    """
    if str.lower(penalty) == 'l2':
        p_term = p*np.linalg.norm(x)**2
    elif str.lower(penalty) == 'f1':
        raise NotImplementedError
    elif str.lower(penalty) == 'f2':
        p_term = alpha*np.sum(np.exp(-beta*x) - 1)
    
    return x.T @ (A @ x) + p_term

def maxclique_grad(
    A: np.ndarray, x: np.ndarray, penalty: str = 'l2',
    p: int = 0.5, alpha: float = None, beta: float = None, eps: float = None
    ):
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
            raise ValueError('Check constants!')
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



def _linesearch(f, g, d, xk, fk=None, gk=None, max_step=1, A=None, c1=0.1, c2=0.9, maxiter=10000):
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

    # if we have an L2 penalty, can compute the optimal stepsize analytically
    # by setting the derivative w.r.t. alpha to 0
    if gk is None:
        gk = g(xk)
    if not (A is None):
        return (gk.T @ d) / (d.T @ A @ d)
    # for different norms, use a numerical algorithm
    def phi(alpha): return f(x=xk+alpha*d)
    def phi_prime(alpha): return g(x=xk+alpha*d) @ d
    
    if fk is None:
        fk = f(x=xk)

    a = [0, max_step/2]
    i = 1
    phi_prev = fk
    for _ in range(maxiter):
        ai = a[i]
        phi_ai = phi(ai)
        phi_prime0 = phi_prime(0)
        if phi_ai > fk + c1*ai*phi_prime0 or (phi_ai > phi_prev and i > 1):
            a_star = _ls_zoom(a[i-1], ai, phi, phi_prime, fk, phi_prime0, c1, c2)
            return a_star
        phi_prime_ai = phi_prime(ai)
        if np.abs(phi_prime_ai) <= -c2 * phi_prime0:
            a_star = ai
            return a_star
        if phi_prime_ai >= 0:
            a_star = _ls_zoom(ai, a[i-1], phi, phi_prime, fk, phi_prime0, c1, c2)
            return a_star
        a.append(ai*2)
        phi_prev = phi_ai
    warn(f'Line search did not converge in maxiter={maxiter} iterations', RuntimeWarning)
    return ai

def _ls_zoom(a, b, phi, phi_prime, phi0, phi_prime0, c1=1e-4, c2=0.9, maxiter=50):
    a_new = 0
    for _ in range(maxiter):
        a_l = a if a < b else b
        a_h = b if a < b else a
        # if a1 is None:
        #     a1 = _min_quad(a_h, phi, phi0, phi_prime0)
        # else:

        a_new = a_l + 0.5*(a_h-a_l)
        if phi(a_new) > phi0 + c1*a_new*phi_prime0 or phi(a_new) > phi(a_l):
            a_h = a_new
        else:
            phi_pr_j = phi_prime(a_new)
            if np.abs(phi_pr_j) <= -c2*phi_prime0:
                return a_new
            if phi_pr_j * (a_h - a_l) >= 0:
                a_h = a_l
            a_l = a_new
    warn(f'Zoom f-n did not converge in maxiter={maxiter} iterations', RuntimeWarning)
    return a_new

def _min_quad(a0, phi, phi0=None, phi_prime0=None):
    return -phi_prime0 * a0**2 / (2*(phi(a0) - phi0 - phi_prime0*a0))

def _min_cub(a0, phi, phi0, phi_prime0, a1):
    t = 1/(a0**2 * a1**2 * (a1-a0))

    a = t * ((a0**2)*(phi(a1) - phi0 - phi_prime0*a1) - (a1**2) * (phi(a0) - phi0 - phi_prime0 * a0))
    b = t * ((-a0**3)*(phi(a1) - phi0 - phi_prime0*a1) + (a1**3) * (phi(a0) - phi0 - phi_prime0 * a0))

    return (-b + np.sqrt(b**2 - 3*a*phi_prime0))/(3*a)

def linesearch_armijo(f, g, d, xk, fk=None, gk=None, amax=1, A=None, c1=1e-4, c2=0.9, maxiter=1000):
    """
    Armijo Line-Search algorithm.
    Source: Wright and Nocedal, 'Numerical Optimization', 1999, p. 57

    f: target f-n;
    g: gradient of f;
    d: descent direction;
    xk: starting point;
    fk: value of target f-n at xk;
    gk: value of gradient at xk;
    max_step: maximum stepsize;
    A: matrix for the maxclique problem, used to compute alpha analytically for l2 penalty;
    """

    if gk is None:
        gk = g(xk)

    # if we have an L2 penalty, can compute the optimal stepsize analytically
    # by setting the derivative w.r.t. alpha to 0
    #if not (A is None):
    #    return (gk.T @ d) / (d.T @ A @ d)
    
    def phi(alpha): return f(x=xk+alpha*d)

    if fk is None:
        fk = f(x=xk)

    a0 = amax
    phi_prime_0 = gk.T @ d
    a = a0
    rho = 0.3
    inc = 1.01
    # a_l = 0
    # a_h = amax
    # while True:
    #     if phi(a) <= fk + c1*a*phi_prime_0:
    #         if np.abs(phi_prime(a)) >= c2 * phi_prime_0:
    #             return a
    #         a_l = a
    #     a_h = a
    #     a = (a_l + a_h)/2
    while phi(a) > fk + c1*a*phi_prime_0:
        a *= rho
    return a

def frankwolfe(
    x_0: np.ndarray,
    f=maxclique_target,
    grad=maxclique_grad, 
    lmo=maxclique_lmo,
    max_iter: int = 10000, 
    tol: float = 1e-5, 
    stepsize: float = 'fixed',
    penalty = 'l2',
    A=None
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
    use_analytical = (penalty == 'l2' and not (A is None))

    for k in range(max_iter):
        gk = grad(x=x_hist[-1])
        sk = lmo(gk)
        if (x_hist[-1] - sk) @ gk < tol:
            break
        if stepsize == 'fixed':
            gamma = 2/(k+2)
        elif stepsize == 'armijo':
            if use_analytical:
                dk = x_hist[-1] - sk
                gamma = (gk.T @ dk) / (dk.T @ A @ dk)
            else:
                gamma = linesearch_armijo(f, grad, (sk-x_hist[-1]), x_hist[-1], gk=gk)
        x_next = (1-gamma)*x_hist[-1] + gamma*sk
        x_hist.append(x_next)
        s_hist.append(sk)
    return x_hist, s_hist, k


def frankwolfe_awaysteps(
    x_0: np.ndarray, 
    f,
    grad, 
    lmo,
    max_iter: int = 10000, 
    tol: float = 1e-5, 
    stepsize: float = None,
    penalty: str = 'l2',
    A: np.ndarray = None):
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
    s_t = [x_0]

    weights = dict()
    weights[np.where(x_0)[0][0]] = 1

    num_stall_iter = 0
    gap_prev = 0
    gap = 0
    
    use_analytical = (penalty == 'l2' and not (A is None))

    for k in range(max_iter):
        xk = x_hist[-1]
        gk = grad(x=xk)
        s = lmo(gk)
        # find fw direction
        d_fw = s - xk
        # check the FW gap
        gap_fw = -d_fw @ gk
        v = s_t[np.argmax([gk @ v for v in s_t], axis=0)]
        d_as = xk - v
        # choose the direction
        gap_as = -gk @ d_as
        if gap_fw >= -gk @ d_as:
            use_fw = True
            gamma_max = 1
            dk = d_fw
            gap = gap_fw
        else:
            use_fw = False
            a_v =  weights[np.where(v)[0][0]]
            gamma_max = a_v/(1-a_v)
            dk = d_as
            gap = gap_as
        
        if np.abs(gap - gap_prev) < 1e-6:
            if num_stall_iter > 5:
                break
            num_stall_iter += 1
        gap_prev = gap

        # Line Search for stepsize; if using l2 penalty, find analytically
        if use_analytical:
            #gamma, _, _ ,_ ,_ ,_ = line_search(f, grad, xk, dk, gk, c1=1e-6, c2=0.9)
            gamma = (gk.T @ dk) / (dk.T @ A @ dk)
        else:
            gamma = linesearch_armijo(f=f, g=grad, d=dk, xk=xk, gk=gk, amax=gamma_max, c1=1e-4)
        #gamma = _linesearch(f=f, g=grad, d=dk, xk=xk, gk=gk, c1=1e-4, c2=0.9)
        x_next = xk + gamma*dk
        # update weights
        if use_fw:
            s_ind = np.where(s)[0][0]
            # if gamma is 1 (we removed all previous x's), we dont need any other vertices
            if abs(gamma - 1) < tol:
                s_t = [s]
                weights = {s_ind: 1}
            # else if we already used s
            elif s_ind in weights.keys():
                weights.update((vertex, weight*(1-gamma)) for vertex, weight in weights.items())
                weights[s_ind] += gamma
            # else if first encounter of s
            else:
                s_t.append(s)
                weights.update((vertex, weight*(1-gamma)) for vertex, weight in weights.items())
                weights[s_ind] = gamma
        else:
            v_ind = np.where(v)[0][0]
            if abs(gamma - gamma_max) < tol:
            # if we completely removed v from x
                weights.pop(v_ind)
                s_t.pop([np.all(v == x) for x in s_t].index(True))
            else:
                weights.update((vertex, weight*(1+gamma)) for vertex, weight in weights.items())
                weights[v_ind] -= gamma
        x_hist.append(x_next)
    return x_hist, s_t, k


def frankwolfe_pairwise(
    x_0: float, 
    f, 
    grad, 
    lmo, 
    max_iter: int = 10000,
    tol: float = 1e-5, 
    stepsize: float = None,
    penalty='l2',
    A: np.ndarray = None):

    x_hist = [x_0]
    s_t = [x_0]
    weights = dict()
    weights[np.where(x_0)[0][0]] = 1
    num_stall_iter = 0
    gap_prev = 0

    use_analytical = (penalty == 'l2' and not (A is None))

    for k in range(max_iter):
        xk = x_hist[-1]
        gk = grad(x=xk)
        s = lmo(gk)
        # find fw direction
        d_fw = s - xk
        # check the FW gap
        gap_fw = -d_fw @ gk
        # check stopping criterion
        if gap_fw < tol:
            break
        if np.abs(gap_fw - gap_prev) < 1e-6:
            if num_stall_iter > 5:
                break
            num_stall_iter += 1
        gap_prev = gap_fw

        # find away-step direction
        v = s_t[np.argmax([gk @ v for v in s_t],axis=0)]
        # combine directions
        a_v =  weights[np.where(v)[0][0]]
        dk = s - v
        # Line Search for stepsize; if using l2 penalty, find analytically
        if use_analytical:
            #gamma, _, _ ,_ ,_ ,_ = line_search(f, grad, xk, dk, gk, c1=1e-4, c2=0.9,maxiter=1000)
            #if gamma is None:
            gamma = (gk.T @ dk) / (dk.T @ A @ dk)
        else:
            gamma = linesearch_armijo(f=f, g=grad, d=dk, xk=xk, gk=gk, amax=a_v, c1=1e-4)
        x_next = xk + gamma*dk
        # update weights for s
        s_ind = np.where(s)[0][0]
        if s_ind in weights.keys():
            weights[s_ind] += gamma
        else:
            s_t.append(s)
            weights[s_ind] = gamma
        # update weights for v
        weights[np.where(v)[0][0]] -= gamma
        x_hist.append(x_next)
    return x_hist, s_t, k