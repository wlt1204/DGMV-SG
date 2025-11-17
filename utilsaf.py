
from sklearn.metrics import roc_auc_score

import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataset.dataload_mv100k import BasicDataset
from time import time
# from AFLayer import PairWiseModel
from parse import parse_args
import random
import os

args = parse_args()

try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname

    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(args.seed)
    sample_ext = True
except:
    print("Cpp extension not loaded")
    sample_ext = False


def graph_to_dict(graphs):
    graph_tup = graphs.nonzero()
    graph_dict = dict()
    for key in graph_tup[0]:
        graph_dict[key] = []

    for key, val in zip(graph_tup[0], graph_tup[1]):
        graph_dict[key].append(val)

    return graph_dict


def graph_to_dict_used_pos(graphs):
    graph_dict = dict()
    for key in range(graphs[-1].shape[0]):
        graph_dict[key] = []

    for graph in graphs:
        graph_tup = graph.nonzero()
        for key, val in zip(graph_tup[0], graph_tup[1]):
            if val in graph_dict[key]:
                continue
            else:
                graph_dict[key].append(val)

    return graph_dict


# class BPRLoss:
#     def __init__(self,  recmodel, args: dict):
#         self.model = recmodel
#         self.bpr_loss = recmodel.bpr_loss
#         self.weight_decay = args.weight_decay
#         self.lr = args.learning_rate
#         self.opt = optim.Adam(recmodel.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#     def stageOne(self, users, pos, neg, g_embed):
#
#         loss, reg_loss = self.bpr_loss(users, pos, neg, g_embed)  ## 输出12个Graph的
#         reg_loss = reg_loss * self.weight_decay
#         loss = loss + reg_loss
#
#         self.opt.zero_grad()
#         loss.backward()
#         self.opt.step()
#         # loss.cpu().item()
#
#         return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio=1):
    dataset: BasicDataset
    allPos = dataset.allPos
    start = time()
    if sample_ext:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    end = time()
    # print('UniformSample_original time...', start - end)
    return S

# 正负样本采样方法
def UniformSample_original_python_all_pos(dataset):

    total_start = time()
    dataset: BasicDataset
    # user_num = dataset.trainDataSize
    # users = np.random.randint(0, dataset.n_users, user_num)  # 生成user_num个[0,n_user)的随机数
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    # for i, user in enumerate(users):
    for user, pos_con in enumerate(allPos):
        start = time()
        posForUser = pos_con
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start

        for positem in posForUser:
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
        '''
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        '''
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    # print('total_time:', total)
    return np.array(S)
def UniformSample_original_python(dataset):

    total_start = time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)  # 生成user_num个[0,n_user)的随机数
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


# ===================end samplers==========================
# =====================utils====================================


def set_seed(seed):
    np.random.seed(seed)  # 指定随机数生成时所用算法开始的整数值
    if torch.cuda.is_available():  # cuda是否可用
        torch.cuda.manual_seed(seed)  # 为CPU中设置种子，生成随机数
        torch.cuda.manual_seed_all(seed)  # 为特定GPU设置种子，生成随机数
    torch.manual_seed(seed)



def minibatch(*tensors, **kwargs):  # 返回一个batch数量的user

    batch_size = kwargs.get('batch_size', args.bpr_batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]  # yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后(下一行)开始
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}
    return recall


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def HRatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    hr = np.sum(pred_data, axis=1) / np.sum(test_matrix, axis=1)
    return np.sum(hr)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue,
                        predictTopK))  # 得到了一个只有True或False的列表，groundTrue和predictTopK有几个值相同，就有几个True，剩下的就是False
        pred = np.array(pred).astype("float")  # 将True，False转换成1，0
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
