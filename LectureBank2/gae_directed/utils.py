import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from sklearn.metrics import roc_auc_score, average_precision_score,accuracy_score,precision_recall_fscore_support

from sklearn.preprocessing import normalize

np.random.seed(17)

def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
        # objects.append(
        #     pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def parse_index_file(filename):
    """
    Parse the index file.

    Args:
        filename: (str): write your description
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    """
    Convert a sparse sparse matrix to a sparse matrix.

    Args:
        sparse_mx: (todo): write your description
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges_ori(adj):
    """
    Generate edges of a random edge.

    Args:
        adj: (todo): write your description
    """
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        """
        Return true if two arrays are equal.

        Args:
            a: (int): write your description
            b: (int): write your description
            tol: (float): write your description
        """
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    '''
    Add self connection, normalization
    :param adj:
    :return:
    '''


    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])

    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    """
    Get roc curve

    Args:
        emb: (todo): write your description
        adj_orig: (todo): write your description
        edges_pos: (todo): write your description
        edges_neg: (todo): write your description
    """
    def sigmoid(x):
        """
        Return the sigmoid

        Args:
            x: (float): write your description
        """
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


### we add our functions ####
'Our own functions'
THD = 0.5  # loading threshold, similarity
ID_SHIFT=1695

N_CONCEPT = 322
THD_WMD = 0.8

def my_load_data():
    """
    My_load data. feature.

    Args:
    """
    # load features and adj matrix
    # feature_path='test_data/features.tsv'
    # adj_path = 'test_data/adj_matrix.tsv'

    feature_path = 'test_data/cora.x'
    adj_path = 'test_data/cora.adj'

    features = []
    id_mapping = {}
    count = 0
    with open(feature_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')

            if len(line[1:-1]) == 1433:
                id_mapping[line[0]] = count
                count += 1
                features.append([float(x) for x in line[1:-1]])
            else:
                features.append([0. for _ in range(0, 1433)])

    n_nodes = len(features)
    n_dim = len(features[0])
    features = np.asarray(features, dtype=np.float64)
    features = sp.csr_matrix(features)

    rows = []
    cols = []
    values = []
    with open(adj_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if (line[0] in id_mapping.keys()) and (line[1] in id_mapping.keys()):
                rows.append(id_mapping[line[0]])
                cols.append(id_mapping[line[1]])
                # values.append(int(line[2]))
                values.append(1)
    # add row and col

    adj = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = sp.lil_matrix(adj)

    '''
    (Pdb) type(adj)
    <class 'scipy.sparse.csr.csr_matrix'>
    ***** adj.get_shape(): (3327, 3327)
    (Pdb) type(features)
    <class 'scipy.sparse.lil.lil_matrix'>
    ***** features.get_shape() (3327, 3703)
    '''
    # import pdb;
    # pdb.set_trace()
    return adj, features

def load_relations(rel_path,n_nodes):
    '''
    We load tsv file into sparse matrix
    :return: sparse matrix.
    '''

    rows = []
    cols = []
    values = []

    with open(rel_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if float(line[2]) > THD:
                rows.append(line[0])
                cols.append(line[1])
                values.append(float(line[2]))

    # add row and col
    adj = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = sp.lil_matrix(adj)

    # import pdb;pdb.set_trace()

    return adj

def load_relations_all(rel_path1, rel_path2,n_nodes):
    '''
    We load tsv file into sparse matrix
    :return: sparse matrix.
    '''

    rows = []
    cols = []
    values = []
    types = []
    '1st relations'
    with open(rel_path1, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if float(line[2]) > THD:
                rows.append(line[0])
                cols.append(line[1])
                values.append(float(line[2]))

    '2nd relations'
    with open(rel_path2, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if float(line[2]) > THD:
                rows.append(line[0])
                cols.append(line[1])
                values.append(float(line[2]))

    # add row and col
    adj = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = sp.lil_matrix(adj)

    return adj

def _load_wmd_adj():
    '''
    This method will load the wmd adj matrix
    :return: adj matrix <class 'scipy.sparse.lil.lil_matrix'>
    '''
    adj_path = '/home/lily/zl379/SP19/AAN/LectureBank/feature_matrix/wmd/full_wmd_adj.npz'
    adj_dense = np.load(adj_path)['arr_0']


    print ('Loading wmd adj...')
    'all relations'
    # adj = sp.lil_matrix(adj_dense)

    # 'dd' relations
    adj_dd = adj_dense.copy()
    for i in range(0,len(adj_dense)):
        for j in range(len(adj_dense)-N_CONCEPT,len(adj_dense)):
            adj_dd[i,j] = 0
    for i in range(len(adj_dense)-N_CONCEPT,len(adj_dense)):
        for j in range(0, len(adj_dense)-N_CONCEPT):
            adj_dd[i,j] = 0

    'filter'
    adj_dd[adj_dd < 0.7] = 0
    # import pdb;
    # pdb.set_trace()

    adj_dd = sp.lil_matrix(adj_dd)

    print ('Finished doc-doc relations')
    # dc relations
    adj_cd = adj_dense.copy()
    for i in range(0, len(adj_dense) - N_CONCEPT):
        for j in range(0, len(adj_dense) - N_CONCEPT):
            adj_cd[i, j] = 0
    for i in range(len(adj_dense) - N_CONCEPT, len(adj_dense)):
        for j in range(len(adj_dense) - N_CONCEPT, len(adj_dense)):
            adj_cd[i, j] = 0

    adj_cd[adj_cd < 0.6] = 0
    adj_cd = sp.lil_matrix(adj_cd)
    print ('Finished doc-con relations')

    # import pdb;pdb.set_trace()
    '''
    (Pdb) adj = sp.lil_matrix(adj_dense)
    (Pdb) adj
    <2017x2017 sparse matrix of type '<class 'numpy.float64'>'
        with 2033131 stored elements in LInked List format>
    (Pdb) type(adj)
    <class 'scipy.sparse.lil.lil_matrix'>
    '''

    return adj_cd, adj_dd



def my_load_data_tfidf(if_wmd='n'):
    '''
    This is the code to load tfidf features.
    :return:
    '''
    print ('Loading TF-IDF data..')
    # load features and adj matrix
    base_path = '/home/lily/zl379/SP19/AAN/LectureBank/feature_matrix/TF_IDF/'
    feature_path = base_path + 'tf_features.tsv'
    adj_cd_path = base_path + 'tf_rel_cd.tsv'
    adj_dd_path = base_path + 'tf_rel_dd.tsv'


    features = []
    tags = []
    with open(feature_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            'ignoring id col'
            features.append([float(x) for x in line[1:-1]])
            tags.append(line[-1])

    features = np.asarray(features, dtype=np.float64)
    features = sp.csr_matrix(features)  # shape(1717x322)
    n_nodes, n_dim = features.get_shape()

    # load together
    # adj= load_relations_all(adj_cd_path, adj_dd_path,n_nodes)

    if if_wmd == 'y':
        adj_cd, adj_dd = _load_wmd_adj()
    else:
        # load one by one
        adj_cd = load_relations(adj_cd_path, n_nodes)
        adj_dd = load_relations(adj_dd_path, n_nodes)

    '''
    (Pdb) adj
    <2039x2039 sparse matrix of type '<class 'numpy.float64'>'
            with 2020 stored elements in LInked List format>
    (Pdb) features
    <2039x322 sparse matrix of type '<class 'numpy.float64'>'
            with 10028 stored elements in Compressed Sparse Row format>
    '''


    features = torch.FloatTensor(np.array(features.todense()))
    return adj_cd, adj_dd, features



def my_load_data_p2v(if_wmd='n'):
    '''
    This is the code to load tfidf features.
    Note that the document order is the same with tfidf; concept should starts from 0
    :return:
    '''
    print ('Loading P2V data..')
    # load features and adj matrix
    rel_base_path = '/home/lily/zl379/SP19/AAN/LectureBank/feature_matrix/TF_IDF/'
    feature_base_path = '/home/lily/zl379/SP19/AAN/LectureBank/feature_matrix/phrase2vec/'
    doc_feature_path = feature_base_path + 'avg_doc_50.tsv'
    con_feature_path = feature_base_path + 'concepts_50.tsv'
    adj_cd_path = rel_base_path + 'tf_rel_cd.tsv'
    adj_dd_path = rel_base_path + 'tf_rel_dd.tsv'


    features = []
    tags = []

    # load documents
    with open(doc_feature_path, 'r') as f:
        for ind, line in enumerate(f.readlines()):
            line = line.strip().split(' ')
            features.append([float(x) for x in line[1:]])
            tags.append(line[0])

    # load concepts
    with open(con_feature_path, 'r') as f:
        for ind, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            features.append([float(x) for x in line[1:]])
            tags.append(line[0])

    features = np.asarray(features, dtype=np.float64)
    features = sp.csr_matrix(features)  # shape(1717x322)

    # normalize features (axis 0 by column, 1 by row)
    features = normalize(features, norm='l1', axis=0)
    n_nodes, n_dim = features.get_shape()

    # load together
    # adj= load_relations_all(adj_cd_path, adj_dd_path,n_nodes)

    if if_wmd=='y':
        adj_cd, adj_dd = _load_wmd_adj()
    else:
        # load one by one
        adj_cd = load_relations(adj_cd_path,n_nodes)
        adj_dd = load_relations(adj_dd_path, n_nodes)


    # import pdb;pdb.set_trace()
    features = torch.FloatTensor(np.array(features.todense()))
    return adj_cd, adj_dd, features



def my_load_data_tfidf_semi(if_wmd='n'):
    '''
    This is the toy code,
    we return label for each file also
    :return:
    '''
    # load features and adj matrix
    print ('Loading TF-IDF data with labels...*** semi-supervised ***')
    # load features and adj matrix
    base_path = '/home/lily/zl379/SP19/AAN/LectureBank/feature_matrix/TF_IDF/'
    feature_path = base_path + 'tf_features.tsv'
    adj_cd_path = base_path + 'tf_rel_cd.tsv'
    adj_dd_path = base_path + 'tf_rel_dd.tsv'

    label_path = '/home/lily/zl379/SP19/AAN/LectureBank/labels/semi/semi_labels.tsv'

    'load doc labels'
    doc_labels = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            doc_labels[line[0]] = line[1]


    features = []
    tags = []
    with open(feature_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            'ignoring id col'
            features.append([float(x) for x in line[1:-1]])
            # tags.append(line[-1])

            if line[-1].startswith('1_') or line[-1].startswith('2_'):
                tags.append(int(doc_labels[line[-1]]))

    # add tags from the documents
    tags +=[_ for _ in range(322)]

    features = np.asarray(features, dtype=np.float64)
    features = sp.csr_matrix(features)  # shape(1717x322)
    n_nodes, n_dim = features.get_shape()

    # load together
    # adj= load_relations_all(adj_cd_path, adj_dd_path,n_nodes)

    if if_wmd == 'y':
        adj_cd, adj_dd = _load_wmd_adj()
    else:
        # load one by one
        adj_cd = load_relations(adj_cd_path, n_nodes)
        adj_dd = load_relations(adj_dd_path, n_nodes)
    '''
    (Pdb) adj
    <2039x2039 sparse matrix of type '<class 'numpy.float64'>'
            with 2020 stored elements in LInked List format>
    (Pdb) features
    <2039x322 sparse matrix of type '<class 'numpy.float64'>'
            with 10028 stored elements in Compressed Sparse Row format>
    '''
    # import pdb;pdb.set_trace()
    features = torch.FloatTensor(np.array(features.todense()))

    'tags is a list of int ids, we make one-hot'
    # nb_classes = max(tags)+1
    # targets = np.array(tags).reshape(-1)
    # one_hot_tags = np.eye(nb_classes)[targets]
    # tags_nodes = torch.IntTensor(one_hot_tags)
    tags_nodes = torch.LongTensor(tags)

    return adj_cd, adj_dd, features, tags_nodes

def my_load_data_p2v_semi(if_wmd='n'):
    '''
    This is the code to load tfidf features.
    Note that the document order is the same with tfidf; concept should starts from 0
    :return:
    '''

    print ('Loading P2V data..with labels...*** semi-supervised ***')
    # load features and adj matrix
    rel_base_path = '/home/lily/zl379/SP19/AAN/LectureBank/feature_matrix/TF_IDF/'
    feature_base_path = '/home/lily/zl379/SP19/AAN/LectureBank/feature_matrix/phrase2vec/'
    doc_feature_path = feature_base_path + 'avg_doc_150.tsv'
    con_feature_path = feature_base_path + 'concepts_150.tsv'
    adj_cd_path = rel_base_path + 'tf_rel_cd.tsv'
    adj_dd_path = rel_base_path + 'tf_rel_dd.tsv'


    label_path = '/home/lily/zl379/SP19/AAN/LectureBank/labels/semi/semi_labels.tsv'

    'load doc labels'
    doc_labels = {}
    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            doc_labels[line[0]] = line[1]


    features = []
    tags = []

    # load documents
    with open(doc_feature_path, 'r') as f:
        for ind, line in enumerate(f.readlines()):
            line = line.strip().split(' ')
            features.append([float(x) for x in line[1:]])

            if line[0].startswith('1_') or line[0].startswith('2_'):
                tags.append(int(doc_labels[line[0]]))


    # add tags from the documents
    tags += [_ for _ in range(322)]

    # load concepts
    with open(con_feature_path, 'r') as f:
        for ind, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            features.append([float(x) for x in line[1:]])

    features = np.asarray(features, dtype=np.float64)
    features = sp.csr_matrix(features)  # shape(1717x322)

    # normalize features (axis 0 by column, 1 by row)
    features = normalize(features, norm='l1', axis=0)
    n_nodes, n_dim = features.get_shape()

    # load together
    # adj= load_relations_all(adj_cd_path, adj_dd_path,n_nodes)

    if if_wmd == 'y':
        adj_cd, adj_dd = _load_wmd_adj()
    else:
        # load one by one
        adj_cd = load_relations(adj_cd_path, n_nodes)
        adj_dd = load_relations(adj_dd_path, n_nodes)



    features = torch.FloatTensor(np.array(features.todense()))

    'tags is a list of int ids, we make one-hot'

    tags_nodes = torch.LongTensor(tags)

    # import pdb;pdb.set_trace()
    return adj_cd, adj_dd, features, tags_nodes

def load_test():
    'Load testing dataset'
    adj_cc_path ='/home/lily/zl379/SP19/AAN/LectureBank/labels/annotations/union_annotation.csv'  # This is for testing

    rows = []
    cols = []
    values = []

    'doc-con relations'
    with open(adj_cc_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            if float(line[2]) > THD:
                rows.append(int(line[0])-1+ID_SHIFT)
                cols.append(int(line[1])-1+ID_SHIFT)
                values.append(float(line[2]))

    # add row and col
    adj = sp.coo_matrix((values, (rows, cols)), shape=(2017, 2017))
    adj = sp.lil_matrix(adj)

    return adj


def load_train_90_percent(iter):
    'Load testing dataset'
    # adj_cc_path ='/home/lily/zl379/SP19/AAN/LectureBank/labels/annotations/union_annotation.csv'  # This is for testing

    train_pos_path = '/home/lily/af726/gae-pytorch/gae/edges/train_edges_positive_'+iter
    train_neg_path = '/home/lily/af726/gae-pytorch/gae/edges/train_edges_negative_'+iter

    rows = []
    cols = []
    values = []

    'doc-con relations'
    with open(train_pos_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')

            rows.append(int(line[0])+ID_SHIFT)
            cols.append(int(line[1])+ID_SHIFT)
            values.append(1.)
    with open(train_neg_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')

            rows.append(int(line[0])+ID_SHIFT)
            cols.append(int(line[1])+ID_SHIFT)
            values.append(0.)

    # add row and col
    adj = sp.coo_matrix((values, (rows, cols)), shape=(2017, 2017))
    adj = sp.lil_matrix(adj)

    return adj

def load_test_10_percent(iter):
    'Load testing dataset'
    # adj_cc_path ='/home/lily/zl379/SP19/AAN/LectureBank/labels/annotations/union_annotation.csv'  # This is for testing

    test_pos_path = '/home/lily/af726/gae-pytorch/gae/edges/test_edges_positive_'+iter
    test_neg_path = '/home/lily/af726/gae-pytorch/gae/edges/test_edges_negative_'+iter

    rows = []
    cols = []
    values = []

    'doc-con relations'
    with open(test_pos_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')

            rows.append(int(line[0])+ID_SHIFT)
            cols.append(int(line[1])+ID_SHIFT)
            values.append(1.)
    with open(test_neg_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')

            rows.append(int(line[0])+ID_SHIFT)
            cols.append(int(line[1])+ID_SHIFT)
            values.append(0.)

    # add row and col
    adj = sp.coo_matrix((values, (rows, cols)), shape=(2017, 2017))
    adj = sp.lil_matrix(adj)


    return adj


def mask_train_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    '''
    TFIDF
    (Pdb) type(adj)
    <class 'scipy.sparse.lil.lil_matrix'>
    (Pdb) adj
    <2039x2039 sparse matrix of type '<class 'numpy.float64'>'
            with 2020 stored elements in LInked List format>
    '''



    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0


    adj_tuple = sparse_to_tuple(adj)
    edges = adj_tuple[0]

    edges_all = sparse_to_tuple(adj)[0] # [(a,b)]
    num_val = int(np.floor(edges.shape[0] / 10.))


    all_edge_idx = [_ for _ in range(edges.shape[0])]
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]

    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, val_edge_idx, axis=0)


    def ismember(a, b, tol=5):
        """
        Check if two arrays.

        Args:
            a: (int): write your description
            b: (int): write your description
            tol: (float): write your description
        """
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))


    # test_edges_false = []
    # while len(test_edges_false) < len(test_edges):
    #     idx_i = np.random.randint(0, adj.shape[0])
    #     idx_j = np.random.randint(0, adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if ismember([idx_i, idx_j], edges_all):
    #         continue
    #     if test_edges_false:
    #         if ismember([idx_j, idx_i], np.array(test_edges_false)):
    #             continue
    #         if ismember([idx_i, idx_j], np.array(test_edges_false)):
    #             continue
    #     test_edges_false.append([idx_i, idx_j])

    'negative samples'
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)



    'we want original values'
    rows = train_edges[:, 0] # train row ids
    cols = train_edges[:, 1] # train col ids
    right_data = []
    tmp_adj = adj.todense()

    'extract original values'
    for i,j in zip(rows,cols):
        right_data.append(tmp_adj[i,j])

    'final train adj matrix'
    adj_train = sp.csr_matrix(
        (right_data, (rows, cols)), shape=adj.shape)

    'not symmetric weights'
    # adj_train = adj_train + adj_train.T

    # data = np.ones(train_edges.shape[0])
    # # Re-build adj matrix
    # adj_train = sp.csr_matrix(
    #     (data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # adj_train = adj_train + adj_train.T


    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false


def make_test_edges(adj, adj_test):
    # TODO: Clean up.
    # adj is adj train
    '''
    adj_test:
    <2017x2017 sparse matrix of type '<class 'numpy.float64'>'
        with 1551 stored elements in LInked List format>
    :param adj:
    :param adj_test:
    :return:
    '''


    # Remove diagonal elements
    # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    # adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    'not useful'
    adj_triu = sp.triu(adj) # (2039, 2039)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0] # [(a,b)]



    train_edges = edges
    test_edges = sparse_to_tuple(adj_test)[0]


    def ismember(a, b, tol=5):
        """
        Check if two arrays.

        Args:
            a: (int): write your description
            b: (int): write your description
            tol: (float): write your description
        """
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    'negative samples'
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])


    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(test_edges, train_edges)


    # NOTE: these edge lists only contain single direction of edge!
    return test_edges, test_edges_false


def my_eval(emb, adj_orig, edges_pos, edges_neg):

    def sigmoid(x):
        'each value goes into this func individually'
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    'inner product?  <2017*16, 16*2017> -> <2017,2017>'
    adj_rec = np.dot(emb, emb.T)
    # import pdb;
    # pdb.set_trace()


    'pos'
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        # preds.append(adj_rec[e[0], e[1]])
        # import pdb;pdb.set_trace()
        if adj_orig[0][e[0], e[1]] >= adj_orig[1][e[0], e[1]]:
            pos.append(adj_orig[0][e[0], e[1]])
        else:
            pos.append(adj_orig[1][e[0], e[1]])

    'neg'
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        # preds_neg.append(adj_rec[e[0], e[1]])

        if adj_orig[0][e[0], e[1]] >= adj_orig[1][e[0], e[1]]:
            neg.append(adj_orig[0][e[0], e[1]])
        else:
            neg.append(adj_orig[1][e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    # import pdb;
    # pdb.set_trace()

    'get accuracy, preds_all ndarray'
    acc_score,p,r,f1 = get_accuracy(labels_all, preds_all)

    auc_score = roc_auc_score(labels_all, preds_all)
    map_score = average_precision_score(labels_all, preds_all)


    return acc_score, p,r,f1, map_score, auc_score



def my_eval_test(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        'each value goes into this func individually'
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    'inner product?  <2017*16, 16*2017> -> <2017,2017>'
    adj_rec = np.dot(emb, emb.T)
    # adj_rec = recovered.data.numpy()

    'pos'
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        # import pdb;pdb.set_trace()
        if adj_orig[0][e[0], e[1]] >= adj_orig[1][e[0], e[1]]:
            pos.append(adj_orig[0][e[0], e[1]])
        else:
            pos.append(adj_orig[1][e[0], e[1]])

    'neg'
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

        if adj_orig[0][e[0], e[1]] >= adj_orig[1][e[0], e[1]]:
            neg.append(adj_orig[0][e[0], e[1]])
        else:
            neg.append(adj_orig[1][e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    'get accuracy, preds_all ndarray'
    acc_score, p, r, f1 = get_accuracy(labels_all, preds_all)


    auc_score = roc_auc_score(labels_all, preds_all)
    map_score = average_precision_score(labels_all, preds_all)
    return acc_score, p, r, f1, map_score, auc_score


def get_accuracy(labels,preds):
    """
    Calculate accuracy.

    Args:
        labels: (todo): write your description
        preds: (array): write your description
    """

    # preds = np.expand_dims(preds,axis=1)
    # threshold is 0.5, x = preds_all.copy()
    #
    x = preds.copy()

    test = (x - np.mean(x)) / np.std(x)
    test[test > 0.6] = 1
    test[test <= 0.6] = 0

    acc = accuracy_score(labels,test)

    p,r,f1,_ = precision_recall_fscore_support(labels,test, average='macro')

    return acc,p,r,f1



# test code
# my_load_data_tfidf_semi()
# _load_wmd_adj()