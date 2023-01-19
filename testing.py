import numpy as np
from fw import maxclique_grad, maxclique_lmo, frankwolfe
from matplotlib import pyplot as plt
from scipy import stats

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

def graph_dict_to_matrix(d: dict, size: int):
    A = np.zeros(shape=(size,size))
    for v1 in d.keys():
        for v2 in d[v1]:
            A[v1-1,v2-1] = 1
    if not np.all(A == A.T):
        A += A.T
    return A

def read_dimacs_brock(fpath: str):
    clique = []
    graph = dict()
    graph_size = None
    with open(fpath) as f:
        reading_clique = False
        for line in f:
            l = line.strip('\n').split(' ')
            if reading_clique:
                if len(l) == 1 and l[0] == 'c':
                    reading_clique = False
                else:
                    for word in l:
                        if word.isnumeric():
                            clique.append(int(word))
            elif l[0] == 'e':
                v1 = int(l[1])
                v2 = int(l[2])
                if v1 in graph.keys():
                    graph[v1].append(v2)
                else:
                    graph[v1] = [v2]
            elif graph_size is None and 'Graph Size:' in line:
                graph_size = int(line.split(':')[1].split(',')[0])
            elif 'Clique Elements are:' in line:
                reading_clique = True
    return graph, graph_size, clique

def main():
    graph, graph_size, clique = read_dimacs_brock('DIMACS/brock200_2.clq')
    A = graph_dict_to_matrix(d=graph, size=graph_size)
    n_tries = 100
    res = []
    f_res = []
    for i in range(n_tries):
        x_0 = np.random.uniform(low=0.0, high=1.0, size=graph_size)
        x_0 /= np.sum(x_0)
        x_hist, _ = frankwolfe(
                grad = lambda x: -maxclique_grad(A, x,penalty='f2', p=0.5, alpha=1, beta=0.5),
                lmo = maxclique_lmo, max_iter = 10000, x_0 = x_0)
        x = x_hist[-1]
        res.append(x)
        f_res.append(1./(1.-np.dot(np.dot(x.T,A),x)))
    avg_clique_size = stats.mode(f_res)
    max_clique_size = np.max(f_res)
    print(f'Mode found: {avg_clique_size}, Largest found: {max_clique_size}, Largest actual: {len(clique)}')


    n_graphs = 0
    for _ in range(n_graphs):
        n = 6
        A = make_test_graph(n, 0.5)
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