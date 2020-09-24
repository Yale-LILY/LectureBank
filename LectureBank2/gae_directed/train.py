from __future__ import division
from __future__ import print_function

import argparse
import time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
import datetime

from torch import optim

from model import GCNModelVAE,GCNModelVAE_Semi
from optimizer import loss_function, loss_function_relation_semi,loss_function_label,loss_function_relation
from utils import load_test_10_percent, mask_train_edges, preprocess_graph, load_test, my_load_data_tfidf, make_test_edges, \
    load_train_90_percent, my_eval, my_load_data_p2v,my_eval_test,my_load_data_tfidf_semi,my_load_data_p2v_semi


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# epochs was 200
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--ds', type=str, default='tfidf', help='type of dataset: tfidf, p2v')
parser.add_argument('--class_dim', type=int, default=322, help='Number of topics')
parser.add_argument('--labels', type=str, default='n', help='If add labels for training.')
parser.add_argument('--wmd', type=str, default='n', help='If use wmd for adjacency matrix.')

args = parser.parse_args()


def gae_for(args,iter='0.txt'):
    # print("Using {} dataset".format(args.ds))
    # adj_cd, features = load_data(args.ds)

    'Load features!'
    if args.ds.startswith('tf'):
        if args.labels == 'y':

            adj_cd, adj_dd, features, tags_nodes = my_load_data_tfidf_semi(args.wmd)
        else:
            adj_cd, adj_dd, features = my_load_data_tfidf(args.wmd)
    else:
        #
        if args.labels == 'y':
            adj_cd, adj_dd, features, tags_nodes = my_load_data_p2v_semi(args.wmd)
        else:
            adj_cd, adj_dd, features = my_load_data_p2v(args.wmd)
            # adj_cd, adj_dd, features = my_load_data_p2v()


    'Load test adjacency matrix'
    adj_test = load_test()
    # adj_test = load_test_10_percent(iter)


    n_nodes, feat_dim = features.shape
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig_cd = adj_cd

    'do again for adj_dd'
    adj_orig_dd = adj_dd

    adj_train_cd, train_edges, val_edges, val_edges_false = mask_train_edges(adj_cd)

    'do again for adj_dd'
    adj_train_dd, train_edges_dd, _, _ = mask_train_edges(adj_dd)  #

    adj_cd = adj_train_cd
    adj_dd = adj_train_dd

    test_edges, test_edges_false = make_test_edges(adj_train_cd, adj_test)

    # Some preprocessing: calculate norm
    adj_norm_cd = preprocess_graph(adj_cd)
    'For loss function: add diag values'
    adj_label_cd = adj_train_cd + sp.eye(adj_train_cd.shape[0])
    adj_label_cd = torch.FloatTensor(adj_label_cd.toarray())
    pos_weight_cd = float(adj_cd.shape[0] * adj_cd.shape[0] - adj_cd.sum()) / adj_cd.sum()
    norm_cd = adj_cd.shape[0] * adj_cd.shape[0] / float((adj_cd.shape[0] * adj_cd.shape[0] - adj_cd.sum()) * 2)

    'do it again for adj_dd'
    adj_norm_dd = preprocess_graph(adj_dd)
    adj_label_dd = adj_train_dd + sp.eye(adj_train_dd.shape[0])
    adj_label_dd = torch.FloatTensor(adj_label_dd.toarray())
    pos_weight_dd = float(adj_dd.shape[0] * adj_dd.shape[0] - adj_dd.sum()) / adj_dd.sum()
    norm_dd = adj_dd.shape[0] * adj_dd.shape[0] / float((adj_dd.shape[0] * adj_dd.shape[0] - adj_dd.sum()) * 2)


    if args.labels == 'y':
        model = GCNModelVAE_Semi(feat_dim, args.hidden1, args.hidden2, args.dropout, args.class_dim)
    else:
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print ('Now start training...')
    hidden_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        # import pdb;pdb.set_trace

        if args.labels == 'y':
            recovered, mu, logvar, pred_nodes = model(features, [adj_norm_cd, adj_norm_dd])
            loss = loss_function_relation_semi(preds=recovered, labels=(adj_label_cd, adj_label_dd),
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=(norm_cd,norm_dd), pos_weight=(pos_weight_cd,pos_weight_dd),pred_nodes=pred_nodes, tags_nodes = tags_nodes)
        else:
            recovered, mu, logvar = model(features, [adj_norm_cd, adj_norm_dd])
            loss = loss_function_relation(preds=recovered, labels=(adj_label_cd, adj_label_dd),
                                       mu=mu, logvar=logvar, n_nodes=n_nodes,
                                       norm=(norm_cd,norm_dd), pos_weight=(pos_weight_cd,pos_weight_dd))

        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()

        acc_curr, p,r,f1,map_curr, roc_curr = my_eval(hidden_emb, (adj_orig_cd,adj_orig_dd), val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(map_curr),"val_ac=", "{:.5f}".format(acc_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    acc_score, p,r,f1, map_score, roc_score = my_eval_test(hidden_emb,(adj_orig_cd,adj_orig_dd), test_edges, test_edges_false)
    # print('Test ROC score: ' + str(roc_score))
    # print('Test AP score: ' + str(ap_score))
    # print ("Test accuracy ", "{:.5f}".format(acc_score))
    # print ('P {:.5f}, R {:.5f}, F1 {:.5f}'.format(p,r,f1))
    print ('Acc, P, R, F1, MAP, AUC')
    print ('{:5f},{:5f},{:5f},{:5f},{:5f},{:5f}'.format(acc_score,p,r,f1,map_score,roc_score))

    return acc_score,p,r,f1,map_score,roc_score



if __name__ == '__main__':

    'We run '
    seeds = [25,2019,5,13,22]
    iterations = [str(x) + '.txt' for x in range(0, 5)]

    # import pdb;pdb.set_trace()
    results = defaultdict(list)
    for iter,seed in zip(iterations,seeds):
        # if you are suing GPU
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        acc, pre, rec, f1, map, auc = gae_for(args,iter)

        results['acc'].append(acc)
        results['pre'].append(pre)
        results['rec'].append(rec)
        results['f1'].append(f1)
        results['map'].append(map)
        results['auc'].append(auc)

    # get average
    output=[]
    flags = '_'.join([args.ds,str(args.epochs),'semi?'+args.labels,'wmd?'+args.wmd])

    output.append(flags)
    for k,v in results.items():
        avg = float("{0:.5f}".format(sum(v)/5))
        output.append(str(avg))
        print (k,avg)

    # add time
    # tm = datetime.datetime.now().isoformat()
    # output.append(tm)

    print (','.join(output))