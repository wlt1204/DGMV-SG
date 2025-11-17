import argparse
import numpy as np
import torch
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, nargs='?', default=12, #  切片为10 # 9
                        help="total time steps used for train, eval and test")
    # Experimental settings.
    parser.add_argument('--dataset', type=str, nargs='?', default='1_doubanmv', # [movielens 100k,movielens 1M]
                        help='1_doubanmv 0_taobao')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--epochs', type=int, nargs='?', default=300,
                        help='# epochs')
    parser.add_argument('--input_dim', type=int, nargs='?', default=64, help='dim of feature')
    parser.add_argument('--val_freq', type=int, nargs='?', default=1,
                        help='Validation frequency (in epochs)')
    parser.add_argument('--test_freq', type=int, nargs='?', default=1,
                        help='Testing frequency (in epochs)')
    parser.add_argument('--batch_size', type=int, nargs='?', default=128, help='Batch size (# nodes)')

    parser.add_argument('--featureless', type=bool, nargs='?', default=True,
                        help='True if one-hot encoding.')
    parser.add_argument("--early_stop", type=int, default=300,
                        help="patient")
    ############################################################################################
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--bpr_batch_size", type=int, default=64, help="bpr_batch_size")
    ############################################################################################
    # 1-hot encoding is input as a sparse matrix - hence no scalability issue for large datasets.
    # Tunable hyper-params
    # TODO: Implementation has not been verified, performance may not be good.
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    # Number of negative samples per positive pair.
    parser.add_argument('--neg_sample_size', type=int, nargs='?', default=1,
                        help='# negative samples per positive')
    # Walk length for random walk sampling.
    parser.add_argument('--walk_len', type=int, nargs='?', default=5,
                        help='Walk length for random walk sampling')
    # Weight for negative samples in the binary cross-entropy loss function.
    parser.add_argument('--neg_weight', type=float, nargs='?', default=1.0,
                        help='Weightage for negative samples')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.002,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--spatial_drop', type=float, nargs='?', default=0.4,
                        help='Spatial (structural) attention Dropout (1 - keep probability).')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    # Architecture params
    parser.add_argument('--structural_layer_num', type=str, nargs='?', default='3',
                        help='Encoder layer num: # attention heads in each GAT layer')
    parser.add_argument('--structural_layer_dim', type=str, nargs='?', default='64',
                        help='Encoder layer dim: # units in each layer')
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',  #####################
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_dim', type=str, nargs='?', default='64',
                        help='Encoder layer dim: # units in each Temporal layer')
    parser.add_argument('--position_ffn', type=str, nargs='?', default='True',help='Position wise feedforward')
    parser.add_argument('--window', type=int, nargs='?', default=-1,help='Window for temporal attention (default : -1 => full)')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent_dim')
    parser.add_argument('--topks', nargs='?', default=[5, 10, 15, 20, 25], help="@k test list")
    parser.add_argument('--layers', type=int, default=3, help="the layer num of DYSAT")
    parser.add_argument('--decay', type=float, default=1e-4, help="the weight decay for l2 normalizaton")
    parser.add_argument('--keep_prob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--A_split', type=bool, default=False, help="--") # 0.6
    parser.add_argument('--Neg_k', type=int, default=1, help="Neg_k")  # 0.6
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--testbatch', type=int, default=100, help="the batch size of users for testing")  # 100
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)  # 指定随机数生成时所用算法开始的整数值
    if torch.cuda.is_available():  # cuda是否可用
        torch.cuda.manual_seed(seed)  # 为CPU中设置种子，生成随机数
        torch.cuda.manual_seed_all(seed)  # 为特定GPU设置种子，生成随机数
    torch.manual_seed(seed)