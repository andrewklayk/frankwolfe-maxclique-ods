import numpy as np
from fw import maxclique_grad, maxclique_lmo, frankwolfe
from matplotlib import pyplot as plt

def make_test_graph(n, p):
    r = [0,1]
    A = np.tril(np.random.default_rng().choice(a=r, size=(n,n),p=[1-p, p]), k=-1)
    A = A + A.T
    return A

def draw_graph(A):
    import networkx as nx
    A_ = np.copy(A)
    G = nx.Graph(A_)
    nx.draw(G, labels={i:f'{i}' for i in range(A.shape[0])})


def main():
    n_graphs = 5
    for _ in range(n_graphs):
        n = 6
        A = make_test_graph(n, 0.75)
        draw_graph(A)
        x_0 = np.random.uniform(low=0.0, high=1.0, size=n)
        x_0 /= np.sum(x_0)

        x_hist, _ = frankwolfe(
            grad = lambda x: -maxclique_grad(A, x,penalty='f2', alpha=1, beta=0.1),
            lmo = maxclique_lmo, max_iter = 10000, x_0 = x_0)
        x = x_hist[-1]
        print(1./(1.-np.dot(np.dot(x.T,A),x)))
        print(x_hist[-1])
        plt.show()


if __name__ == "__main__":
    main()