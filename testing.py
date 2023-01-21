# math
import numpy as np
from fw import *
# read and save data and results
import pandas as pd
import os
# progress bar
from tqdm import tqdm
# visualisation
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

def graph_dict_to_matrix(d: dict, size: int):
    A = np.zeros(shape=(size,size))
    for v1 in d.keys():
        for v2 in d[v1]:
            A[v1-1,v2-1] = 1
    if not np.all(A == A.T):
        A += A.T
    return A


#TODO
def read_dimacs(fpath: str):
    pass

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
    n = 6
    A = make_test_graph(n, 0.6)
    draw_graph(A)
    plt.show()
    
    x_0 = np.zeros(n,dtype='float')
    x_0[np.random.randint(low=0, high=n)] = 1.
    x_hist, _, k = frankwolfe_pairwise(
                    f=lambda x: -maxclique_target(A, x),
                    grad = lambda x: -maxclique_grad(A, x, penalty='l2'),
                    lmo = maxclique_lmo, max_iter = 10000, x_0 = x_0, tol=1e-3)
    x = x_hist[-1]
    print(x)
    print(1./(1.-np.dot(np.dot(x.T,A),x)))








    graphs_dir = 'DIMACS'
    n_tries = 10
    results = dict()
    print('\n')

    for filename in os.listdir(graphs_dir):
        f = os.path.join(graphs_dir, filename)
        if 'brock' in filename:
            graph, graph_size, clique = read_dimacs_brock(f)
        else:
            continue
        A = graph_dict_to_matrix(d=graph, size=graph_size)
        res = []
        iters = []
        f_res = []
        graph_name = '.'.join(filename.split('.')[:-1])
        print(f'Working on {graph_name}')
        for i in tqdm(range(n_tries)):
            x_0 = np.zeros(graph_size,dtype='float')
            x_0[np.random.randint(low=0, high=graph_size)] = 1.
            x_hist, _, k = frankwolfe(
                #f=lambda x: -maxclique_target(A, x),
                grad = lambda x: -maxclique_grad(A, x, penalty='l2'),
                lmo = maxclique_lmo, max_iter = 10000, x_0 = x_0, tol=1e-4)
            x = x_hist[-1]
            res.append(x)
            iters.append(k)
            f_res.append(1./(1.-np.dot(np.dot(x.T,A),x)))
        avg_clique_size = np.mean(f_res)
        max_clique_size = np.max(f_res)
        cs_std = np.std(f_res)
        results[graph_name] = (avg_clique_size, max_clique_size, cs_std)
        #print(f'Mean found: {avg_clique_size}, Largest found: {max_clique_size}, Largest actual: {len(clique)}')
    pd.DataFrame(results).to_csv('mcresults.csv')


if __name__ == "__main__":
    main()