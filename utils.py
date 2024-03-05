import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
from sklearn import metrics
from models import *
from scipy.sparse import csc_matrix
from scipy.linalg import inv
from layers import aggregator_lookup


def top_k_preds(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)
    top_k_array = np.array(predictions, np.int64)

    return top_k_array


def cal_f1_score(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1


def batch_generator(nodes, batch_size, shuffle=True):
    num = nodes.shape[0]
    chunk = num // batch_size
    while True:
        if chunk * batch_size + batch_size > num:
            chunk = 0
            if shuffle:
                idx = np.random.permutation(num)
        b_nodes = nodes[idx[chunk*batch_size:(chunk+1)*batch_size]]
        chunk += 1

        yield b_nodes


def eval_iterate(nodes, batch_size, shuffle=False):
    idx = np.arange(nodes.shape[0])
    if shuffle:
        idx = np.random.permutation(idx)
    n_chunk = idx.shape[0] // batch_size + 1
    for chunk_id, chunk in enumerate(np.array_split(idx, n_chunk)):
        b_nodes = nodes[chunk]

        yield b_nodes


def do_iter(emb_model, cly_model, adj, adj_val, feature, labels, diff_idx, diff_val, idx, cal_f1=False):
    embs_1 = emb_model(idx, adj, adj_val, feature)
    embs_2 = emb_model(idx, diff_idx, diff_val, feature)
    embs = torch.cat([embs_1, embs_2], dim=1)
    preds = cly_model(embs)
    labels_idx = torch.argmax(labels[idx], dim=1)
    cly_loss = F.cross_entropy(preds, labels_idx)
    if not cal_f1:
        return embs, cly_loss
    else:
        targets = labels[idx].cpu().numpy()
        preds = top_k_preds(targets, preds.detach().cpu().numpy())
        return embs, cly_loss, preds, targets


def evaluate(emb_model, cly_model, adj, adj_val, feature, labels, diff_idx, diff_val, idx, batch_size, mode='val'):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, adj_val, feature, labels,
                                                                                         diff_idx, diff_val, b_nodes, cal_f1=True)
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))

    return cly_loss, micro_f1, macro_f1, embs_whole, targets_whole


def get_split(labels, shot, target_flag, seed):
    idx_tot = np.arange(labels.shape[0])
    num_class = labels.shape[1]
    np.random.seed(seed)
    np.random.shuffle(idx_tot)
    if target_flag:
        labels = np.argmax(labels, axis=1)
        idx_train = []
        for class_idx in range(num_class):
            node_idx = np.where(labels == class_idx)[0]
            node_idx = np.random.choice(node_idx, shot, replace=False)
            idx_train.append(node_idx)
        idx_train = np.concatenate(idx_train, axis=0)
        np.random.shuffle(idx_train)
        idx_val = np.array([])
        idx_test = np.array(list(set(list(idx_tot)) - set(list(idx_train))))
        np.random.shuffle(idx_test)
    else:
        partition = [0.7, 0.1, 0.2]
        num_train, num_val = int(labels.shape[0] * partition[0]), int(labels.shape[0] * partition[1])
        idx_train, idx_val, idx_test = idx_tot[:num_train], idx_tot[num_train:num_train+num_val], \
                                       idx_tot[num_train+num_val:]

    return idx_train, idx_val, idx_test, idx_tot


def make_adjacency(G, max_degree, seed):
    all_nodes = np.sort(np.array(G.nodes()))
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes, max_degree)) + (n_nodes - 1)).astype(int)
    neibs_num = np.array([])
    np.random.seed(seed)
    for node in all_nodes:
        neibs = np.array(G.neighbors(node))
        neibs_num = np.append(neibs_num, len(neibs))
        if len(neibs) == 0:
            neibs = np.array(node).repeat(max_degree)
        elif len(neibs) < max_degree:
            neibs = np.random.choice(neibs, max_degree, replace=True)
        else:
            neibs = np.random.choice(neibs, max_degree, replace=False)
        adj[node, :] = neibs

    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def pre_social_net(adj, features, labels):
    features = csc_matrix(features.astype(np.uint8))
    labels = labels.astype(np.int32)

    return adj, features, labels


def load_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', shot=None, target_flag=False, seed=123,
              is_blog=False, alpha_ppr=0.2, diff_k=128):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    diff_val, diff_idx = compute_ppr(adj, alpha=alpha_ppr, diff_k=diff_k)
    adj_dense = np.array(adj.todense())
    edges = np.vstack(np.where(adj_dense)).T
    Graph = nx.from_edgelist(edges)
    adj = make_adjacency(Graph, 32, seed)
    adj_val = np.ones(shape=adj.shape, dtype=diff_val.dtype)
    if shot is not None:
        target_flag = True
    idx_train, idx_val, idx_test, idx_tot = get_split(labels, shot, target_flag, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    adj_val = torch.from_numpy(adj_val).float()
    diff_idx = torch.from_numpy(diff_idx)
    diff_val = torch.from_numpy(diff_val).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), adj_val.to(device), diff_idx.to(device), diff_val.to(device), features.to(device), labels.to(device), \
           idx_train.to(device), idx_val.to(device), idx_test.to(device), idx_tot.to(device)


def compute_ppr(adj, alpha=0.2, diff_k=128, self_loop=True):
    a = adj.todense().A
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.sum(a, axis=1)
    dinv_sqrt = np.power(d, -0.5)
    dinv_sqrt[np.isinf(dinv_sqrt)] = 0.
    dinv = np.diag(dinv_sqrt)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    diff = alpha * inv(np.eye(a.shape[0]) - (1 - alpha) * at)
    diff_val, diff_idx = get_top_k_matrix_row_norm(diff, k=diff_k)

    return diff_val, diff_idx


def get_top_k_matrix_row_norm(A, k=128):
    A = A.T
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    diff_idx = A.argsort(axis=0)[num_nodes - k:]
    diff_val = A[A.argsort(axis=0)[num_nodes - k:], row_idx]
    norm = diff_val.sum(axis=0)
    norm[norm <= 0] = 1
    diff_val = diff_val/norm

    return diff_val.T, diff_idx.T
