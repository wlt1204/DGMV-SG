# -*- encoding: utf-8 -*-

import time
from utils.preprocess import load_graphs
from utils import utilsaf
from utils import Procedure
from utils.parse import parse_args
from utils.parse import set_seed
from models.model_mv100k import LMADG

from dataset import dataload_mv100k
import torch

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    # data_path = 'F:/models/多模态/VLM多模态蒸馏感知动态图匹配推荐/dataset/1_doubanmv'
    data_path = 'dataset/2_doubanmv'
    args = parse_args()
    print(args)

    set_seed(args.seed)
    graphs_exchange = load_graphs(args.dataset)
    dataset = [dataload_mv100k.Loader(graph, idd, path=data_path) for idd, graph in enumerate(graphs_exchange)]

    val_dict = utilsaf.graph_to_dict(graphs_exchange[-2])
    test_dict = utilsaf.graph_to_dict(graphs_exchange[-1])
    useritem_used_dict = utilsaf.graph_to_dict_used_pos(graphs_exchange[:-2])

    assert args.time_steps <= len(dataset), "Time steps is illegal"  # 判断 步长 和 图是否 对应

    model = LMADG(args=args,
                  num_features=args.input_dim,
                  time_length=args.time_steps,
                  datasetloder=dataset,
                  graphs=graphs_exchange).to(args.device)

    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    # in training
    best_epoch_recall = 0
    best_epoch_ndcg = 0

    patient = 0
    precision_max = 0
    recall_max = 0
    ndcg_max = 0
    for epoch in range(args.epochs):
        start = time.time()

        model.train()
        loss = model()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        end = time.time()

        if epoch % 1 == 0:
            print("TESTing")
            model.eval()

            t_val, t_test = args.time_steps-2, args.time_steps-1
            results = Procedure.Test(val_dict, model, t_val, args.testbatch, useritem_used_dict, args.multicore)
            precision_max = max(precision_max, max(results['precision']))
            recall_max = max(recall_max, max(results['recall']))
            ndcg_max = max(ndcg_max, max(results['ndcg']))
            print('Val: precision_max:', precision_max, 'recall_max:', recall_max, 'ndcg_max:', ndcg_max)
            ############################################################################################

            results = Procedure.Test(test_dict, model, t_test, args.testbatch, useritem_used_dict, args.multicore)
            precision_max = max(precision_max, max(results['precision']))
            recall_max = max(recall_max, max(results['recall']))
            ndcg_max = max(ndcg_max, max(results['ndcg']))
            print('Test: precision_max:', precision_max, 'recall_max:', recall_max, 'ndcg_max:', ndcg_max)

            if recall_max > best_epoch_recall:
                best_epoch_recall = recall_max
                torch.save(model.state_dict(), "./model_checkpoints/model.pt")
                patient = 0
            else:
                patient += 1
                if patient > args.early_stop:
                    break
            print("Epoch {:<3}, recall_max {:.3f} ndcg_max {:.3f}".format(epoch, recall_max, ndcg_max))

            ############################################################################################

