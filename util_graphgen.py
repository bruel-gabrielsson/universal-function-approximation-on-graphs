import networkx as nx
import numpy as np
import random

def gen_cycle_graphs():
    list_graphs = []
    for i in [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]:
        g1 = nx.generators.classic.cycle_graph(i)
        g2 = nx.disjoint_union(nx.generators.classic.cycle_graph(i//2), nx.generators.classic.cycle_graph(i//2))
        list_graphs.append((g1, 0))
        list_graphs.append((g2, 1))

    return list_graphs

def gen_npab_graphs():
    list_graphs = []
    for i in range(2,20):
        g1 = nx.MultiGraph()
        g2 = nx.MultiGraph()
        for k in range(i):
            g1.add_edge(0, 1)
            g2.add_edge(0, 0)
        g2.add_node(1)

        list_graphs.append((g1, 0))
        list_graphs.append((g2, 1))

    return list_graphs

def erdos_renyi_graph():
    list_graphs = []
    for i in range(100):
        g1 = nx.generators.erdos_renyi_graph(10, 0.5)
        list_graphs.append((g1, i))

    return list_graphs

def random_regular_graph():
    list_graphs = []
    for i in range(10):
        #g1 = nx.generators.random_regular_graph(3, 10)
        g1 = nx.generators.configuration_model([2 for _ in range(8)], seed=i)
        list_graphs.append((g1, i)) # )) #%2))

    return list_graphs

def save_list_nx(list_nx, file_name):
    #from collections import defaultdict

    f = open(file_name,"w")
    f.write("%d\n" % len(list_nx))
    for i in range(len(list_nx)):
        g, label = list_nx[i]
        f.write("{} {}\n".format(len(g), label))

        nodes = [[] for _ in g] #defaultdict(list) # {} # [[] for _ in g]

        for u, v, weight in g.edges.data("weight"):
            #print(u,v)
            nodes[u].append(v)
            nodes[v].append(u)

        for j in range(len(nodes)):
            ngs = nodes[j]
            st = ' '.join([str(nbr) for nbr in ngs])
            label = random.randint(0, 10)
            f.write("{} {} {}\n".format(label,len(ngs),st)) # label, num_neg, negs, #random.randint(0, 5), label 0 should be label 1

        # print(nodes)
        # assert(False)

        # for n, nbrsdict in g.adjacency():
        #     ngs = nbrsdict.items()
        #     st = ' '.join([str(nbr) for nbr, eattr in ngs])
        #     f.write("{} {} {}\n".format(0,len(ngs),st)) # label, num_neg, negs

        # for n in g:
        #     ngs = g[n]
        #     st = ' '.join([str(nbr) for nbr in ngs])
        #     f.write("{} {} {}\n".format(0,len(ngs),st)) # label, num_neg, negs


    f.close()

if __name__ == "__main__":
    #cycs = gen_cycle_graphs()
    save_list_nx(erdos_renyi_graph(), "./ERDOSLABEL.txt")
