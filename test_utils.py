import numpy as np
import fw
import networkx as nx
import matplotlib.pyplot as plt

def make_test_graph(n, p):
    r = [0,1]
    A = np.tril(np.random.default_rng().choice(a=r, size=(n,n),p=[1-p, p]), k=-1)
    A = A + A.T
    for i in range(n):
        A[i,i] = 1
    return A

def draw_graph(A):
    A_ = np.copy(A)
    for i in range(A_.shape[0]):
        A_[i,i] = 0
    G = nx.Graph(A_)
    nx.draw(G, labels={i:f'{i}' for i in range(A.shape[0])})
    plt.show()



def main():
    A = make_test_graph(5, 0.6)
    draw_graph(A)
    x_0 = np.zeros_like(A[0]) + 0.
    x_0[0] = 1
    x_hist, s_hist = fw.frankwolfe(A, x_0)
    print(x_hist[-1])

if __name__ == "__main__":
    main()