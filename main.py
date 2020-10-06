import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from util import load_data, load_data_order, separate_data
import sys
sys.path.append("models/")
from graphorder import GraphOrder

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        # if batch is bigger than the data, we want repeats
        if len(train_graphs) < args.batch_size:
            selected_idx = np.remainder(np.random.permutation(args.batch_size), len(train_graphs))  #[:args.batch_size]
        else:
            selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        #compute loss
        loss = criterion(output, labels)

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clip this before updating
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        #report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum/total_iters
    print("loss training: %f" % (average_loss))

    return average_loss

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch, num_shuffles=5):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    if len(test_graphs) == 0:
        test_graphs = train_graphs

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    for i in range(num_shuffles-1):
        output += pass_data_iteratively(model, test_graphs)
    output = output / float(num_shuffles)

    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test_acc = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f acc_test: %f" % (acc_train, acc_test, acc_test_acc))

    return acc_train, acc_test, acc_test_acc

def main(obj):
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=50,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--degree_as_tag', action="store_true",
    					help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "",
                                        help='output file')
    parser.add_argument('--type', type = str, default = "",
                                        help='What type of model to use, Order or something else')
    parser.add_argument('--sorting', type = str, default = "deg_both",
                                        help='What type of model to use, Order or something else')
    parser.add_argument('--size', type = str, default = "small",
                                        help='Size of model, "big" or "small"')
    args = parser.parse_args()
    #set up seeds and gpu device
    args.num_layers = -1 # not used
    args.learn_eps = False # not used
    args.dont_split = False
    for key in obj:
        setattr(args, key, obj[key])

    print(args)

    USE_SAVE_IDX = True
    print("USE_SAVE_IDX ", USE_SAVE_IDX)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print(device)

    def prepare_data_order(args):
        graphs, num_classes, max_deg = load_data_order(args.dataset, args.degree_as_tag)
        if USE_SAVE_IDX:
            try:
              train_idx = np.loadtxt('./dataset/%s/10fold_idx/train_idx-%d.txt' % (args.dataset, args.fold_idx+1), dtype=np.int32).tolist()
              test_idx = np.loadtxt('./dataset/%s/10fold_idx/test_idx-%d.txt' % (args.dataset, args.fold_idx+1), dtype=np.int32).tolist()
              train_graphs, test_graphs = [graphs[i] for i in train_idx], [graphs[i] for i in test_idx]
            except:
                print("Could not load dataset indicies")
                if num_classes >= len(graphs) or args.dont_split:
                    train_graphs, test_graphs, train_idx, test_idx = graphs, [], 0, 0
                else:
                    train_graphs, test_graphs, train_idx, test_idx = separate_data(graphs, args.seed, args.fold_idx)
        else:
            if num_classes >= len(graphs) or args.dont_split:
                train_graphs, test_graphs, train_idx, test_idx = graphs, [], 0, 0
            else:
                train_graphs, test_graphs, train_idx, test_idx = separate_data(graphs, args.seed, args.fold_idx)
        return train_graphs, test_graphs, train_idx, test_idx, num_classes, max_deg


    if args.type == "Order" or args.type == "OrderP":
        print("--- ORDER ---")
        train_graphs, test_graphs, train_idx, test_idx, num_classes, max_deg = prepare_data_order(args)
        model = GraphOrder(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, max_deg, args.network, args.sorting, args.size, args.type, device).to(device)

    if not args.filename == "":
        f = open(args.filename[:-3] + "_args.txt","w+")
        f.write(str(args))
        f.write("\n")
        f.write("TEST\n" + str(test_idx) + "\nTRAIN\n" + str(train_idx))

    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    if not args.filename == "":
        f = open(args.filename,"w+")

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch)
        acc_train, acc_test, acc_test_acc = test(args, model, device, train_graphs, test_graphs, epoch)

        if not args.filename == "":
            f.write("%d %f %f %f %f\n" % (epoch, avg_loss, acc_train, acc_test, acc_test_acc))
        print("")

if __name__ == '__main__':
    print("--- start ---")
    import os
    dataset = "MUTAG"
    epochs = 350
    type = "OrderP" # ["Order", "OrderP"]
    size = "big" #["big", "small"]
    lr = 0.01 # 0.01
    for sorting in ["all"]: # ["deg_one", "deg_both", "all", "none"]
        for network in ["LSTM"]:
            for final_dropout in [0]:
                for hidden_dim in [64]:
                    for batch_size in [64]:
                        prefix = "_tree_halfflows5l_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(type, network, dataset, hidden_dim, batch_size, final_dropout, sorting, lr, size)
                        if not os.path.exists("./{}".format(prefix)): os.mkdir("./{}".format(prefix))
                        for i in range(10):
                            # Putting them here instead
                            torch.manual_seed(0)
                            np.random.seed(0)

                            filename = "./{}/{}_type{}_idx{}_h{}_b{}_d{}.txt".format(prefix, dataset, type, i, hidden_dim, batch_size, final_dropout)
                            if os.path.exists(filename): continue # OBS JUST FOR NOW
                            obj = {'sorting':sorting, 'type':type, 'network':network, 'fold_idx':i,
                                'dataset': dataset, 'hidden_dim': hidden_dim, 'batch_size':batch_size,
                                'epochs': epochs, 'filename': filename, 'final_dropout': final_dropout,
                                'lr': lr, 'size': size}
                            main(obj)
