# math
import numpy as np
import fw
# save results
import pandas as pd
import os
# progress bar
from tqdm import tqdm
import re
# timing
import time

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


def read_dimacs_c(fpath: str):
    graph = dict()
    graph_size = None
    with open(fpath) as f:
        for line in f:
            l = line.strip('\n').split(' ')
            if l[0] == 'e':
                v1 = int(l[1])
                v2 = int(l[2])
                if v1 in graph.keys():
                    graph[v1].append(v2)
                else:
                    graph[v1] = [v2]
            elif graph_size is None and 'number of vertices' in line:
                graph_size = int(line.split(':')[1])
    return graph, graph_size

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
    # n = 6
    # A = make_test_graph(n, 0.6)
    # draw_graph(A)
    # plt.show()
    # graph_size = n
    # x_0 = np.zeros(n,dtype='float')
    # x_0[np.random.randint(low=0, high=n)] = 1.
    # for i in tqdm(range(n_tries)):
    #         # generate a random new starting point
    #         x_0 = np.zeros(graph_size,dtype='float')
    #         x_0[np.random.randint(low=0, high=graph_size)] = 1.
    #         # run
    #         norm = 'f2'
    #         x_hist, _, k = fw.frankwolfe_pairwise(
    #             f = lambda x: -fw.maxclique_target(A, x, penalty=norm, alpha=0.04, beta=5),
    #             grad = lambda x: -fw._maxclique_grad(A, x, penalty=norm,alpha=0.04, beta=5),
    #             lmo = fw._maxclique_lmo, 
    #             max_iter = 10000, x_0 = x_0, tol=1e-3)
    # x = x_hist[-1]
    # print(x)
    # print(1./(1.-np.dot(np.dot(x.T,A),x)))


    graphs_dir = 'DIMACS'
    n_tries = 10
    graphs = dict()
    results = dict()
    print('\n')
    np.random.seed(42)
    c_r = re.compile('C[0-9]+')
    for filename in os.listdir(graphs_dir):
        # read graph file
        f = os.path.join(graphs_dir, filename)
        if 'brock' in filename:
            graph, graph_size, clique = read_dimacs_brock(f)
        elif c_r.match(filename):
            graph, graph_size = read_dimacs_c(f)
        else:
            continue
        A = graph_dict_to_matrix(d=graph, size=graph_size)
        graph_name = '.'.join(filename.split('.')[:-1])
        graphs[graph_name] = A

    for graph_name, A in graphs.items():
        if graph_name in ['C4000.5', 'C2000.9']:
            continue
        res = []
        iters = []
        f_res = []
        timing = []
        print(f'Working on {graph_name}')
        # run the algorithm n_tries times
        graph_size = A.shape[0]
        for _ in tqdm(range(n_tries)):
            # generate a random new starting point
            # x_0 = np.random.rand(graph_size)
            # x_0 /= np.sum(x_0)
            x_0 = np.zeros(graph_size,dtype='float')
            x_0[np.random.randint(low=0, high=graph_size)] = 1.
            # run
            norm = 'l2'
            time_start = time.process_time_ns()
            x_hist, _, k = fw.frankwolfe(
                f = lambda x: -fw.maxclique_target(A, x, penalty=norm, alpha=0.04, beta=5),
                grad = lambda x: -fw.maxclique_grad(A, x, penalty=norm,alpha=0.04, beta=5),
                lmo = fw.maxclique_lmo, penalty=norm, stepsize='armijo',
                max_iter = 10000, x_0 = x_0, tol=1e-4, A=A
            )
            time_end = time.process_time_ns()
            timing.append((time_end - time_start) * 1e-9)
            x = x_hist[-1]
            res.append(x)
            iters.append(k)
            f_res.append(1./(1.-np.dot(np.dot(x.T,A),x)))
        avg_clique_size = np.mean(f_res)
        max_clique_size = np.max(f_res)
        cs_std = np.std(f_res)
        avg_time = np.mean(timing)
        results[graph_name] = (max_clique_size,avg_clique_size, cs_std, avg_time)
        #print(f'Mean found: {avg_clique_size}, Largest found: {max_clique_size}, Largest actual: {len(clique)}')
    res = pd.DataFrame(results, index=['Max','Mean','std', 'CPU time'])
    print(res)
    res.to_csv(f'normal_mcresults_{norm}.csv')


if __name__ == "__main__":
    main()