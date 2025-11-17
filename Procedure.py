
# import world
import numpy as np
import torch
from utils import utilsaf
# import dataloader
# from pprint import pprint
from utils.utilsaf import timer

from time import time
from tqdm import tqdm
from models import model_mv100k
import multiprocessing
from sklearn.metrics import roc_auc_score
from parse import parse_args

CORES = multiprocessing.cpu_count() // 2

args = parse_args()


def BPR_train_original(dataset, recommend_model, bpr, args, g_embeds):
    Recmodel = recommend_model
    Recmodel.train()
    # bpr: bpr_fun
    total_batch = 0
    aver_loss = 0.
    # for dataset_in, g_embeds in zip(dataset, g_embeds):
    with timer(name="Sample"):
        S = utilsaf.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(args.device)
    posItems = posItems.to(args.device)
    negItems = negItems.to(args.device)
    users, posItems, negItems = utilsaf.shuffle(users, posItems, negItems)
    total_batch += len(users) // args.bpr_batch_size + 1

    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utilsaf.minibatch(users,
                                                     posItems,
                                                     negItems,
                                                     batch_size=args.bpr_batch_size)):
        # 不要急于计算损失
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, g_embeds)  # 优化  单个图 单个图进行优化 ？？？
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    # time_info = timer.dict()
    timer.zero()
    # print('time_info:', time_info)
    return aver_loss


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utilsaf.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in args.topks:
        ret = utilsaf.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utilsaf.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}

def Test(dataset, Recmodel, t, testbatch, useritem_used_dict, multicore=0):
    u_batch_size = testbatch
    testDict: dict = dataset
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(args.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(args.topks)),
               'recall': np.zeros(len(args.topks)),
               'ndcg': np.zeros(len(args.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []

        total_batch = len(users) // u_batch_size + 1
        for batch_users in utilsaf.minibatch(users, batch_size=u_batch_size):
            allPos = [useritem_used_dict[u] for u in batch_users] #dataset.getUserPosItems(batch_users) # 历史交互过的商家
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)

            rating = Recmodel.getUsersRating(batch_users_gpu, t)
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):  # 将交互过的商家置为-1024 不需要
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)  ############################
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        # if args.tensorboard:
        #     w.add_scalars(f'Test/Recall@{args.topks}',
        #                   {str(args.topks[i]): results['recall'][i] for i in range(len(args.topks))}, epoch)
        #     w.add_scalars(f'Test/Precision@{args.topks}',
        #                   {str(args.topks[i]): results['precision'][i] for i in range(len(args.topks))}, epoch)
        #     w.add_scalars(f'Test/NDCG@{args.topks}',
        #                   {str(args.topks[i]): results['ndcg'][i] for i in range(len(args.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results
