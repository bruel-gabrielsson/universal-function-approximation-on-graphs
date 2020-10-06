import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention

import numpy as np
import sys
sys.path.append("models/")
import copy
from util import plot_graph

import ordercpp

import time

class GraphOrder(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, max_deg, network, sorting, size, type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphOrder, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.middle_dim  = hidden_dim
        self.node_dim = hidden_dim
        if self.type == "OrderP":
            self.node_dim = hidden_dim // 2

        self.sorting = sorting
        self.use_LSTM = False
        self.size = size
        self.type = type # ["Order", "OrderP"]
        print("---Sorting--- ", self.sorting)
        print(self.type)
        if network == "LSTM":
            print("--Using LSTM--")
            self.use_LSTM = True

        if self.size == "small":
            self.firstNet = nn.Sequential(
                nn.Linear(self.input_dim, self.node_dim),
                #nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(self.node_dim),
            )
        else:
            self.firstNet = nn.Sequential(
                nn.Linear(self.input_dim, self.node_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(self.node_dim),
            )

        class TreeLSTM(nn.Module):
            def __init__(self, in_dim, mem_dim):
                super(TreeLSTM, self).__init__()
                self.in_dim = in_dim
                self.mem_dim = mem_dim
                # sharing weights, make batch just bigger
                self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
                self.fh = nn.Linear(self.mem_dim, self.mem_dim)
                self.layernorm = torch.nn.LayerNorm(self.mem_dim)

            def node_forward(self, inputs):
                child_h = inputs[:,:,0,:] # [169, 2, 32]
                child_c = inputs[:,:,1,:]
                child_h_sum = torch.sum(child_h, dim=1, keepdim=False) # [169, 32]
                iou = self.iouh(child_h_sum) # self.ioux(inputs) + self.iouh(child_h_sum)
                i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
                i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u) # torch.Size([169, 32]) torch.Size([169, 32]) torch.Size([169, 32])
                f = torch.sigmoid(self.fh(child_h)) # torch.Size([169, 2, 32]), should be shared weights!
                fc = torch.mul(f, child_c) #
                c = torch.mul(i, u) + torch.sum(fc, dim=1, keepdim=False) # torch.Size([169, 32])
                c = self.layernorm(c)
                h = torch.mul(o, torch.tanh(c)) # torch.Size([169, 32])
                res = torch.stack([h, c], 1)
                return res

            # torch.Size([169, 2, 2, 32]) batch, branch, hc, hidden_dim
            def forward(self, inputs):
                res = self.node_forward(inputs)
                return res

        class TreePLSTM(nn.Module):
            def __init__(self, in_dim, mem_dim):
                super(TreePLSTM, self).__init__()
                self.in_dim = in_dim
                self.mem_dim = mem_dim
                # sharing weights, make batch just bigger
                self.iouh = nn.Linear(self.in_dim, 3 * self.mem_dim)
                self.fh = nn.Linear(self.in_dim, self.mem_dim)
                self.layernorm = torch.nn.LayerNorm(self.mem_dim)

            def node_forward(self, xh, c):
                child_h = xh # inputs[:,:,0,:] # [169, 2, 32]
                child_c = c # inputs[:,:,1,:]
                child_h_sum = torch.sum(child_h, dim=1, keepdim=False) # [169, 32]
                iou = self.iouh(child_h_sum) # self.ioux(inputs) + self.iouh(child_h_sum)
                i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
                i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u) # torch.Size([169, 32]) torch.Size([169, 32]) torch.Size([169, 32])
                f = torch.sigmoid(self.fh(child_h)) # torch.Size([169, 2, 32]), should be shared weights!
                fc = torch.mul(f, child_c) #

                c = torch.mul(i, u) + torch.sum(fc, dim=1, keepdim=False) # torch.Size([169, 32])
                c = self.layernorm(c)
                h = torch.mul(o, torch.tanh(c)) # torch.Size([169, 32])
                res = torch.stack([h, c], 1)
                return res

            # torch.Size([169, 2, 2, 32]) batch, branch, hc, hidden_dim
            def forward(self, xh, c):
                res = self.node_forward(xh, c)
                return res

        if self.type == "Order":
            self.treelstm = TreeLSTM(self.hidden_dim, self.hidden_dim)
        else:
            # (x,h,s) -> h, c
            self.treelstm = TreePLSTM(self.node_dim+self.hidden_dim+1, self.hidden_dim) #TreeLSTM(self.hidden_dim, self.hidden_dim)
            lstmxx = torch.nn.LSTMCell(self.hidden_dim+1, self.node_dim, bias=True)
            layernormxx = torch.nn.LayerNorm(self.node_dim)
            def lstmx(x,hc):
                h,c = lstmxx(x,hc)
                return h, layernormxx(c)
            self.lstmx = lstmx

        # Simpler models
        # self.persistNet = nn.Sequential(
        #     nn.Linear(self.hidden_dim*2, self.hidden_dim),
        #     nn.Sigmoid(),
        #     # ####################################################
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.Sigmoid(),
        # )

        # self.featureNet = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     #nn.Dropout(p=self.final_dropout),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Linear(self.hidden_dim, self.middle_dim),
        #     # nn.LeakyReLU(0.2, inplace=True),
        #     # nn.Linear(self.hidden_dim, self.middle_dim),
        # )

        if self.size == "small":
            self.classifyNet = nn.Sequential(
                nn.Linear(self.middle_dim, self.hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(self.hidden_dim),
                nn.Dropout(p=self.final_dropout),
                # #nn.ReLU(inplace=True),
                # ############### +-+-+-+-+-+-+
                # nn.Linear(self.hidden_dim, self.hidden_dim),
                # nn.LeakyReLU(0.2, inplace=True),
                # # # #nn.ReLU(inplace=True),
                # nn.BatchNorm1d(self.hidden_dim),
                # nn.Dropout(p=self.final_dropout),
                # ################ +-+-+-+-+-+-+
                nn.Linear(self.hidden_dim, self.output_dim),
                #nn.Tanh()
            )
        else:
            self.classifyNet = nn.Sequential(
                nn.Linear(self.middle_dim, self.hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(self.hidden_dim),
                nn.Dropout(p=self.final_dropout),
                # #nn.ReLU(inplace=True),
                # ############### +-+-+-+-+-+-+
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                # # #nn.ReLU(inplace=True),
                nn.BatchNorm1d(self.hidden_dim),
                nn.Dropout(p=self.final_dropout),
                # ################ +-+-+-+-+-+-+
                nn.Linear(self.hidden_dim, self.output_dim),
                #nn.Tanh()
            )

    def TREELSTM(self, x): # torch.Size([1, 2, 32]) torch.Size([1, 32])
        # torch.Size([169, 2, 2, 32])
        # first = x[:,0,:,:] # should be lower level if exist
        # second = x[:,1,:,:]
        return self.treelstm(x)

    def LSTM1(self, nodesf, nodesc, cconcat):
        #print(nodesf.shape, cconcat.shape) # torch.Size([19, 2, 32]) torch.Size([19, 32])
        _x, _h, _c = nodesf.unsqueeze(1), cconcat.unsqueeze(0), nodesc.unsqueeze(0)  # torch.Size([18, 1, 32]) torch.Size([1, 18, 32]) torch.Size([1, 18, 32])
        out, (h_n, c_n) = self.lstm1(_x, (_h, _c))
        res = torch.cat((h_n, c_n), 0).transpose(0, 1) # torch.Size([19, 2, 32])
        return res

    def LSTM2(self, nodesf, nodesc, cconcat):
        #print(nodesf.shape, cconcat.shape) # torch.Size([19, 2, 32]) torch.Size([19, 32])
        _x, _h, _c = nodesf.unsqueeze(1), cconcat.unsqueeze(0), nodesc.unsqueeze(0) # torch.Size([18, 1, 32]) torch.Size([1, 18, 32]) torch.Size([1, 18, 32])
        out, (h_n, c_n) = self.lstm2(_x, (_h, _c))
        res = torch.cat((h_n, c_n), 0).transpose(0, 1) # torch.Size([19, 2, 32])
        return res

    def LSTM(self, x):
        first = x[:,0,:,:] # should be lower level if exist
        second = x[:,1,:,:]
        _x = first[:,0,:]
        _h = second[:,0,:]
        c_to_sum = torch.stack([first[:,1,:], second[:,1,:]], 1)
        _c = torch.sum(c_to_sum, 1) / 2.0 # /2.0 to prevent blowing up (does affect gradient in backprop, signals need to be duplicated to travel far?)
        out, (h_n, c_n) = self.lstm(_x.unsqueeze(1), (_h.unsqueeze(0), _c.unsqueeze(0)))
        res = torch.cat((h_n, c_n), 0).transpose(0, 1)
        #res1 = self.lstm_dropout(res)
        #print(res)
        return res # nn.Sigmoid()(res) #torch.torch.clamp(res, min=-1.5, max=1.5) # Not to get overflows, doesn't make any sense gives def of LSTM. Obs /2.0

    def __preprocess_order(self, batch_graph, sorting="deg_one"): # sorting=["deg_one", "deg_both", "all"]
        # [[2. 1. 2. 2. 0. 0.], ... ]   [n1, n2, deg1, deg2, label1, label2], sorted for each pair with largest first
        option = 0
        powerful = False
        maxlevel = -1 # if -1
        if sorting == "deg_both": option = 1
        if sorting == "all": option = 2
        if sorting == "none": option = -1
        if self.type == "OrderP": powerful = True

        pass_to_cpp = [g.order_edges.astype(np.int32) for g in batch_graph]
        levels, a_levels, b_levels, cppsparseinds = ordercpp.preprocess_order(pass_to_cpp, option, powerful, maxlevel)

        a, b = torch.transpose(torch.LongTensor(cppsparseinds), 0, 1), torch.ones(len(cppsparseinds))
        sparse = torch.sparse.LongTensor(a, b, torch.Size([len(batch_graph), len(cppsparseinds)]))
        return levels, a_levels, b_levels, sparse

    def forward(self, batch_graph):
        node_feats = []
        node_sparse = []
        use_node_feats_in_sum = False
        if use_node_feats_in_sum:
            acc = 0
            for i in range(len(batch_graph)):
                g = batch_graph[i]
                node_feats.append(g.node_features)
                num_feats = len(g.node_features)
                node_sparse.extend([[i, acc + _i] for _i in range(num_feats)])
                acc += num_feats

            # print(torch.LongTensor(node_sparse)) # torch.Size([568, 2])
            node_sparse_m = torch.sparse.LongTensor(torch.transpose(torch.LongTensor(node_sparse), 0, 1), torch.ones(len(node_sparse)), torch.Size([len(batch_graph), len(node_sparse)])).to(self.device)
            X_concat = torch.cat(node_feats, 0).to(self.device)
            X_concat = self.firstNet(X_concat)
            node_sum = torch.spmm(node_sparse_m, X_concat)
            #print(node_sum.shape) # torch.Size([32, 16])
            if node_sum.shape[1] < self.hidden_dim:
                node_sum = torch.nn.functional.pad(node_sum, (0, self.hidden_dim - node_sum.shape[1]))
        else:
            X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
            X_concat = self.firstNet(X_concat) # torch.nn.ConstantPad1d((0, self.hidden_dim - self.input_dim), 0)(X_concat.unsqueeze(1))

        X_concat = X_concat.squeeze(1)
        Node_features = X_concat.clone() # torch.Size([535, 32])
        if self.use_LSTM:
            if self.type == "Order":
                X_concat = torch.stack((X_concat, torch.zeros(X_concat.shape[0], self.hidden_dim).to(self.device)), 1)
            else:
                X_concat = torch.stack((torch.zeros(X_concat.shape[0], self.hidden_dim).to(self.device), torch.zeros(X_concat.shape[0], self.hidden_dim).to(self.device)),1) # torch.Size([572, 2, 32]) h,c

        order_list, a_levels, b_levels, sparse = self.__preprocess_order(batch_graph, self.sorting)
        features = []
        for i in range(len(order_list)):
            indices = order_list[i].to(self.device)
            node_labels_inds = indices[:,:2] # torch.Size([1200, 2])
            h_inds = indices[:,2:4] # torch.Size([1200, 2])
            batch = X_concat[h_inds] # torch.Size([1188, 2, 64])
            #print(batch.shape) # torch.Size([178, 2, 32], h and c, torch.Size([169, 2, 2, 32]), edge, h, c
            if self.type != "Order": # torch.Size([169, 2, 1, 32]) torch.Size([169, 2, 2, 32]) torch.Size([169, 2, 1, 1])
                intersections = indices[:,-1] # torch.Size([1200])
                intersec_tensor = intersections.view(-1,1,1)
                intersec_tensor = torch.stack((intersec_tensor,intersec_tensor), 1).type(torch.FloatTensor).to(self.device)
                #print(Node_features[node_labels_inds].shape) # torch.Size([1397, 2, 32])
                tag_batch = Node_features[node_labels_inds].unsqueeze(2) # [:,:,[0],:]  # torch.Size([162, 2, 2, 32]) -> torch.Size([162, 2, 1, 32])
                #print(tag_batch.shape)
                x_h_s = torch.cat((tag_batch, batch[:,:,[0],:], intersec_tensor), 3).squeeze(2) # torch.Size([170, 2, 1, 33])
                c = batch[:,:,[1],:].squeeze(2)
                #print(x_h_s.shape, c.shape) # torch.Size([176, 2, 65]) torch.Size([176, 2, 32])
                #batch = torch.stack([tag_batch, batch], dim=1) # torch.Size([167, 2, 2, 32])
                #print(x_h_s.shape, c.shape) # torch.Size([1490, 2, 65]) torch.Size([1490, 2, 32])
                #print(x_h_s.shape, c.shape) # torch.Size([162, 2, 9]) torch.Size([162, 2, 4])
                next = self.treelstm(x_h_s, c) #
                #print(next.shape) # torch.Size([164, 2, 32])
                features.append(next[:,1,:]) # Only the h's now!
                # Node_features[node_labels_inds] = next[:,[0,1],:] # updating the node features

            elif self.use_LSTM:
                next = self.TREELSTM(batch) # TREELSTM,  torch.Size([192, 2, 8])
                #print(Node_features[node_labels_inds].shape,  next[:,0,:].shape) # torch.Size([192, 2, 8]) torch.Size([192, 8])
                #Node_features[node_labels_inds] = next[:,0,:] #
                #print(next.shape, Node_features[node_labels_inds].shape) # torch.Size([169, 3, 32]) torch.Size([169, 2, 32])
                #assert(False)
                features.append(next[:,0,:])
                # if self.type == "Order":
                #     features.append(next[:,0,:])
                # else:
                #     features.append(next[:,2,:]) # Only the c's now!
                #     # Node_features[node_labels_inds] = next[:,[0,1],:] # updating the node features
                #     next = next[:,2,:]

            else:
                batch = batch.view(batch.shape[0], -1)
                next = self.persistNet(batch)
                features.append(next)

            X_concat = torch.cat([X_concat, next], 0)

            if self.type == "OrderP":
                # [node, cid]
                ''' LSTM each node '''

                this_a = a_levels[i]
                this_b = b_levels[i] # torch.Size([169, 2]), node ind, c ind
                if True:
                    if len(this_a) > 0 or len(this_b) > 0:
                        #input, (h_0, c_0)
                        input_x = torch.cat((torch.zeros((this_a.shape[0],1)), torch.ones((this_b.shape[0],1))), 0).to(self.device)

                        c = torch.cat((Node_features[this_a[:,0]], Node_features[this_b[:,0]]),0)
                        hcomp = torch.cat((X_concat[this_a[:,1]][:,0,:], X_concat[this_b[:,1]][:,0,:]), 0) # use h's

                        all_input = torch.cat((hcomp, input_x), 1)
                        #print(all_input.shape, h.shape, c.shape) # torch.Size([358, 33]) torch.Size([358, 32]) torch.Size([179, 64])
                        h = torch.zeros_like(c)
                        h_1, c_1 = self.lstmx(all_input, (h,c)) # torch.Size([354, 32]) torch.Size([354, 32])
                        all_inds = torch.cat((this_a[:,0], this_b[:,0]), 0)

                        Node_features[all_inds] = c_1 # torch.stack([h_1,c_1],1) # torch.Size([328, 2, 32])

                else:
                    ''' Just summing '''
                    if (len(this_a[:,0]) > 0):
                        Node_features[this_a[:,0]] += X_concat[this_a[:,1]]

        features = torch.cat(features, dim=0)
        sparse = sparse.to(self.device)
        features = nn.Dropout(p=self.final_dropout)(features)

        per_graph = torch.spmm(sparse, features)
        if use_node_feats_in_sum:
            per_graph += node_sum

        score = self.classifyNet(per_graph)

        return score
