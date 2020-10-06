import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_degrees: a list of node degrees
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.order_edges = []
        self.num_levels = 0
        self.max_neighbor = 0
        self.levels_lengths = []
        self.max_node = 0


def load_data(dataset, degree_as_tag, type=""):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}
    max_deg = -1
    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph() # MultiGraph() #
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n
            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat
    if type=="Order":
        for g in g_list:
            g.label = label_dict[g.label]
    else:
        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)
            max_deg = max(max_deg, g.max_neighbor)

            g.label = label_dict[g.label]

            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges]) # undirected

            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    for g in g_list:
        g.node_degrees = list(dict(g.g.degree).values())

    #Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

        # import matplotlib.pyplot as plt
        # nx.draw(g.g)  # networkx draw()
        # plt.draw()
        # plt.show()
        # assert(False)

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict), max_deg

def load_data_order(name="MUTAG", degree_as_tag=False, sort_feats=4):
    from collections import defaultdict

    degree_as_tag = False
    graphs, num_classes, max_deg = load_data(name, degree_as_tag, type="Order")
    list_graphs = []
    for g in graphs:
        gr = np.zeros([len(g.g.edges), 3, 2])
        i = 0
        tags = g.node_tags
        for e in g.g.edges:
            deg = list(g.g.degree(e))
            arr = [[e[0], e[1]], [deg[0][1], deg[1][1]], [tags[e[0]], tags[e[1]]]]
            gr[i] = np.array(arr)
            i+=1

        gr = np.sort(gr, axis=2)[:,:,::-1]
        degAndFeat = gr.reshape(len(gr), -1)[:,2:2+sort_feats] #2:4
        tr = np.rot90(degAndFeat)
        inds = np.lexsort(tr)
        gr = gr[inds]
        gr = gr.reshape(len(gr), -1)

        g.order_edges = gr.astype(np.int)

        # import matplotlib.pyplot as plt
        # nx.draw(g.g)  # networkx draw()
        # plt.draw()
        # plt.show()
        #assert(False)

    return graphs, num_classes, max_deg

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list, train_idx, test_idx

def plot_progress(path):
    import os
    import matplotlib.pyplot as plt
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if (('.txt' in file) and not 'args' in file): # and (("1." in file) or ("2." in file) or ("3." in file))):
                print(file)
                files.append(os.path.join(r, file))
    print("num files ", len(files))
    data = np.loadtxt(files[0], delimiter=" ")
    it = 0
    its = data[:, 0]
    losses = data[:, 1].reshape(-1, 1)
    trains = data[:, 2].reshape(-1, 1)
    tests = data[:, 3].reshape(-1, 1)
    cols = data.shape[1]
    if cols > 4:
        acc_tests = data[:, 4].reshape(-1, 1)
    for file in files[1:]:
        fdata = np.loadtxt(file, delimiter=" ") #, delimiter="\t")
        #print(fdata.shape, losses.shape)
        losses = np.concatenate((losses, fdata[:,1].reshape(-1, 1)), axis=1)
        trains = np.concatenate((trains, fdata[:,2].reshape(-1, 1)), axis=1)
        tests = np.concatenate((tests, fdata[:,3].reshape(-1, 1)), axis=1)
        if cols > 4:
            acc_tests = np.concatenate((acc_tests, fdata[:,4].reshape(-1, 1)), axis=1)
        it += 1

    maxind = np.argmax(np.mean(tests, axis=1).flatten())
    print("train max {}".format(np.mean(np.max(trains, axis=0))))
    print(np.max(tests, axis=0))
    print(tests[maxind])
    print("[{}, {}, {}]".format(maxind, np.mean(tests[maxind]), np.std(tests[maxind].reshape(-1))), np.mean(np.max(tests, axis=0)))
    if cols > 4:
        maxind = np.argmax(np.mean(acc_tests, axis=1).flatten())
        print("accum [{}, {}, {}]".format(maxind, np.mean(acc_tests[maxind]), np.std(acc_tests[maxind].reshape(-1))), np.mean(np.max(acc_tests, axis=0)))
        plt.plot(data[:,0], np.mean(acc_tests, axis=1), c='y')

    plt.plot(data[:,0], np.mean(losses, axis=1), c='b')
    plt.plot(data[:,0], np.mean(trains, axis=1), c='g')
    plt.plot(data[:,0], np.mean(tests, axis=1), c='c')
    plt.show()

def plot_graph(edges, vals, ordering, special=[]):
    import networkx as nx
    #import numpy as np
    import matplotlib.pyplot as plt
    import pylab

    G = nx.DiGraph()

    G.add_edges_from([(e[0],e[1]) for e in edges], weight=1)
    values = [0.45 for node in G.nodes()]
    edge_labels={(o[0],o[1]):i for i,o in enumerate(ordering)}
    sp = []
    for s in special:
        sp.append((s[0], s[1])); sp.append((s[1], s[0]))
    edge_colors = ['black' if not edge in sp else 'red' for edge in G.edges()]
    print(edge_colors)
    print(len(edge_colors), len(G.edges()), len(edges))
    print(edges)
    pos=nx.planar_layout(G) 
    node_labels = {node:int(vals[node]) for node in G.nodes()}; nx.draw_networkx_labels(G, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nx.draw(G,pos, node_color = values, node_size=400, edge_color=edge_colors, edge_cmap=plt.cm.Reds)
    pylab.show()

def count_class_redundancy_real(dataset="MUTAG"):
    import scipy.misc
    graphs, num_classes, max_deg = load_data_order(dataset, False)
    class_redundancy_edges_only = []
    class_redundancy_edges = []
    graphs_edges = []
    graphs_nodes = []
    levels = []
    for g in graphs:
        #edges = g.order_edges[:,2:] #.astype(np.float64)
        #edge_ties, ecounts = np.unique(edges, axis=0, return_counts=True)
        #print(g.order_edges)
        supernode_ties = []
        prev_edge = g.order_edges[0]
        level = 0
        max_node_num = np.amax(g.order_edges[:,:2])
        node_to_group_num = np.arange(max_node_num+1)
        group_num_to_group = [[i] for i in range(max_node_num+1)]
        group_to_size = np.zeros(len(node_to_group_num))
        group_to_level = np.zeros(len(node_to_group_num))

        node_edge_ties = np.sum(np.all(g.order_edges[:,[2,4]] == g.order_edges[:,[3,5]], axis=1)).astype(np.float64)

        node_ties_edge = 0
        edge_permutations = 1
        graphs_edges.append(len(g.order_edges))
        graphs_nodes.append(max_node_num+1)

        for i in range(len(g.order_edges)):
            edge = g.order_edges[i]
            if not np.array_equal(edge[2:], prev_edge[2:]):
                prev_edge = edge
                edge_permutations = np.float64(edge_permutations) * np.prod(scipy.special.factorial(group_to_size).astype(np.float64))
                group_to_size = np.zeros(len(group_to_size))
            else:
                node_ties_edge += 1

            # count and combine into supernodes
            n1, n2 = edge[0], edge[1]
            g1, g2 = node_to_group_num[n1], node_to_group_num[n2]
            if g1 == g2:
                group_to_size[g1] += 1

            else:
                #if n1 not in group_num_to_group[node_to_group_num[n2]]:
                for node in group_num_to_group[g2]:
                    group_num_to_group[g1].append(node)
                    node_to_group_num[node] = g1
                group_to_size[g1] += group_to_size[g2] + 1
                group_to_size[g2] = 0

            group_to_level[g1] = max(group_to_level[g1], group_to_level[g2]) + 1

        levels.append(np.amax(np.array(group_to_level)))
        edge_permutations = np.float64(edge_permutations) * np.prod(scipy.special.factorial(group_to_size).astype(np.float64))
        class_redundancy_edges_only.append(edge_permutations)
        class_redundancy_edges.append(np.float64(edge_permutations)*2**(np.float64(node_edge_ties)))

    class_redundancy_edges = np.array(class_redundancy_edges).astype(np.float64)
    class_redundancy_edges_only = np.array(class_redundancy_edges_only).astype(np.float64)
    levels = np.array(levels)
    print(dataset)
    print("class_redundancy_edges_only", np.format_float_scientific(np.median(class_redundancy_edges_only)), np.format_float_scientific(np.mean(class_redundancy_edges_only)), np.std(class_redundancy_edges_only))
    print("class_redundancy_edges", np.format_float_scientific(np.median(class_redundancy_edges)), np.format_float_scientific(np.mean(class_redundancy_edges)), np.std(class_redundancy_edges))
    print("avg nodes", np.mean(np.array(graphs_nodes)), np.std(np.array(graphs_nodes)))
    print("avg edges", np.mean(np.array(graphs_edges)), np.std(np.array(graphs_edges)))
    print("levels", np.median(levels), np.mean(levels), np.std(levels))

def count_class_redundancy_GNN(dataset="MUTAG"):
    import scipy.misc
    #graphs, num_classes, max_deg = load_data_order(dataset, False)
    graphs, num_classes, max_deg = load_data(dataset, False)

    class_redundancy_nodes_only = []
    all_levels = []
    node_adjusts = []

    for g in graphs:
        tags, degrees = np.array(g.node_tags), np.array(g.node_degrees)
        comb = np.stack((np.arange(len(tags)), degrees, tags), axis=-1)
        comp = np.random.shuffle(comb)
        sort_var = comb[:,1:]
        tr = np.rot90(sort_var)
        inds = np.lexsort(tr)
        gr = comb #[inds]
        node_order = gr.reshape(len(gr), -1)

        supernode_ties = []
        #prev_edge = g.order_edges[0]
        prev_node = np.array(node_order[0][1:])
        node_permutations = 1
        level = 0
        #max_node_num = np.amax(g.order_edges[:,:2])
        max_node_num = np.amax(node_order[:,0])
        node_to_group_num = np.arange(max_node_num+1)
        group_num_to_group = [[i] for i in range(max_node_num+1)]
        group_to_size = np.ones(len(node_to_group_num)) # becuase each component has one node but no edge already
        group_to_level = np.zeros(len(node_to_group_num))

        processed_edges = []

        # print(g.neighbors)
        # print(node_order)
        # assert(False)
        node_adjust_permutations = 1 # Will be random

        for i in range(len(node_order)):
            node_info = np.array(node_order[i])
            node_id = node_info[0]
            center_group = node_to_group_num[node_id]
            this_neighbors = g.neighbors[node_id] + [node_id]
            this_groups = [node_to_group_num[_n] for _n in this_neighbors]
            this_levels = [group_to_level[_g] for _g in this_groups]
            this_level = max(this_levels) + 1

            if False: # not np.array_equal(prev_node, node_info[1:]):
                prev_node = node_info[1:]
                node_permutations = np.float64(node_permutations) * np.prod(scipy.special.factorial(group_to_size).astype(np.float64))
                group_to_size = np.ones(len(group_to_size))

            # count and combine into supernodes
            #group_to_size[center_group] += np.sum(np.array([group_to_size[node_to_group_num[_n]] for _n in g.neighbors[node_id]]))
            this_node_adjust_perms = 1 # start with one, because center always there
            #unique_groups = np.unique(np.array(this_groups)) # can we really do this?
            #if len(unique_groups) == 1 and unique_groups[0] == center_group: continue # This is not true
            seen_one_new_edge = False
            seen_group_ids = []
            for neigh_id in this_neighbors:
                neigh_group_id = node_to_group_num[neigh_id]
                if ((node_id, neigh_id) in processed_edges) or ((neigh_id, node_id) in processed_edges):
                    # we have seen this before
                    continue

                seen_one_new_edge = True
                if neigh_group_id == center_group:
                    continue

                #print("SSSSS")
                if neigh_group_id not in seen_group_ids: # don't want to count group nodes twice
                    this_node_adjust_perms += 1
                    group_to_size[center_group] += group_to_size[neigh_group_id]
                    group_to_size[neigh_group_id] = 0

                    temp = group_num_to_group[neigh_group_id]
                    for nid in temp:
                        group_num_to_group[center_group].append(nid)
                        node_to_group_num[nid] = center_group
                    seen_group_ids.append(neigh_group_id)

                processed_edges.append((node_id, neigh_id))

            if seen_one_new_edge == False:
                continue # if we have seen no new edge, level doesn't have to be increased

            # for gid in unique_groups:
            #     if gid == center_group: # If within same, then that group of nodes has already been included
            #
            #         continue
            #     else:
            #         this_node_adjust_perms += 1
            #         group_to_size[center_group] += group_to_size[gid]
            #         group_to_size[gid] = 0
            #
            #         temp = group_num_to_group[gid]
            #         for nid in temp:
            #             group_num_to_group[center_group].append(nid)
            #             node_to_group_num[nid] = center_group

            #node_adjust_permutations = scipy.special.factorial(np.float64(max_node_num + 1))
            node_adjust_permutations = np.float64(node_adjust_permutations) * np.prod(scipy.special.factorial(this_node_adjust_perms).astype(np.float64))
            group_to_level[center_group] = this_level

        #print(len(g.g), group_to_size)
        #assert(False)

        node_adjusts.append(node_adjust_permutations)
        node_permutations = np.float64(node_permutations) * np.prod(scipy.special.factorial(group_to_size).astype(np.float64))
        all_levels.append(np.amax(np.array(group_to_level)))
        class_redundancy_nodes_only.append(node_permutations)

    class_redundancy_nodes_only = np.array(class_redundancy_nodes_only).astype(np.float64)
    # class_redundancy_edges_only = np.array(class_redundancy_edges_only).astype(np.float64)
    all_levels = np.array(all_levels)
    node_adjusts = np.array(node_adjusts)
    print(dataset)
    print("class_redundancy_nodes_only", np.format_float_scientific(np.median(class_redundancy_nodes_only)), np.format_float_scientific(np.mean(class_redundancy_nodes_only)), np.std(class_redundancy_nodes_only))
    #print("class_redundancy_edges", np.format_float_scientific(np.median(class_redundancy_edges)), np.format_float_scientific(np.mean(class_redundancy_edges)), np.std(class_redundancy_edges))
    #print("avg nodes", np.mean(np.array(graphs_nodes)), np.std(np.array(graphs_nodes)))
    #print("avg edges", np.mean(np.array(graphs_edges)), np.std(np.array(graphs_edges)))
    print("node_adjusts", np.format_float_scientific(np.median(node_adjusts)), np.format_float_scientific(np.mean(node_adjusts)), np.format_float_scientific(np.std(node_adjusts)))
    print("levels", np.median(all_levels), np.mean(all_levels), np.std(all_levels))


if __name__ == "__main__":
    print("####################################")
    #count_class_redundancy_real("COLLAB")
    #count_class_redundancy_GNN("COLLAB")
    plot_progress("./results/_tree_halfflows5l_Order_LSTM_NCI1_64_128_0.2_deg_both_0.01_big")
