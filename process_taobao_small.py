
import itertools

from itertools import chain

import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix

import dateutil.parser


def lines_per_n(f, n):
    for line in f:
        yield ''.join(chain([line], itertools.islice(f, n - 1)))


def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d


if __name__ == "__main__":
    # def UserItemNet_snapshot():
    data_id = 1
    snapshot = 12
    # data = '{}_taobao'.format(data_id)
    name1 = 'train'
    name2 = 'test'
    save_path = r'F:\models\多模态\VLM多模态蒸馏感知动态图匹配推荐NEW\dataset\{}_taobao_small\sp_npz\graph.pkl'.format(data_id)
    root = r'F:\models\多模态\VLM多模态蒸馏感知动态图匹配推荐NEW\dataset\{}_taobao_small\\'.format(data_id)
    links_1 = np.loadtxt(r'F:\models\多模态\VLM多模态蒸馏感知动态图匹配推荐NEW\dataset\{}_taobao_small\u_i_t_sorted.txt'.format(data_id),
                                dtype=int, delimiter='\t', encoding='utf-8')

    #
    img_feat_arr = np.load(r'F:\models\多模态\VLM多模态蒸馏感知动态图匹配推荐NEW\dataset\{}_taobao_small\img_feat_arr.npy'.format(data_id))
    img_type_feat_arr = np.load(r'F:\models\多模态\VLM多模态蒸馏感知动态图匹配推荐NEW\dataset\{}_taobao_small\img_type_feat_arr.npy'.format(data_id)) # (11631, 4, 300)


    u_map= max(links_1[:, 0])+1
    p_map= max(links_1[:, 1])+1

    print('u_map', u_map)
    print('p_map', p_map)

    links_3 = links_1
    links_4 = links_3
    links = np.array(sorted(links_4, key=lambda x: x[2]))

    ts = links[:, 2]

    print('min(ts)',min(ts), 'max(ts)',max(ts))
    print("# interactions", links.shape[0])
    links = sorted(links,key=lambda x: x[2])
    links.sort(key=lambda x: x[2])

    START_DATE = min(ts)
    END_DATE = max(ts)

    step_times = (END_DATE-START_DATE)//(snapshot-1)
    Date_DATE = (END_DATE-START_DATE)//step_times
    print("Spliting Time Interval: \n Start Time : {}, End Time : {}".format(START_DATE, END_DATE))

    Date_list = [[] for date in range(Date_DATE+1)]

    time_flog = False
    # False
    if time_flog == True:
        for (a, b, time) in links:
            datetime_object = time
            if datetime_object > END_DATE:
                months_diff = (END_DATE - START_DATE) // step_times
            else:
                months_diff = (datetime_object - START_DATE) // step_times
            Date_list[months_diff].append([a, b])

    else:
        n = len(links)
        n_splits = snapshot
        indices = [i * n // n_splits for i in range(n_splits + 1)]
        Date_list = [links[indices[i]:indices[i + 1]] for i in range(n_splits)]

    all_edge = len(links)
    train_edge = sum([len(Date_list[i]) for i in range(len(Date_list)-2)])
    val_edge = len(Date_list[-2])
    test_edge = len(Date_list[-1])

    if time_flog == True:
        for idx, Date_li in enumerate(Date_list):
            if (idx > 0) and (idx < (len(Date_list)-2)):
                Date_list[idx].extend(Date_list[idx-1])
            elif idx == len(Date_list)-2:
                Date_list[-1].extend(Date_list[idx][int(0.5*len(Date_list[idx])):])
                Date_list[-2] = Date_list[-2][:int(0.5*len(Date_list[idx]))]
            else:
                continue
    else:
        for idx, Date_li in enumerate(Date_list):
            if (idx > 0) and (idx < (len(Date_list)-2)):
                Date_list[idx].extend(Date_list[idx-1])

    print('train_edge:{} %.3'.format(train_edge/all_edge),
          'val_edge:{} %'.format(len(Date_list[-2])/all_edge),
          'test_edge:{} %'.format(len(Date_list[-1])/all_edge))

    train_ = np.concatenate((np.array(Date_list[-3]), np.reshape(np.ones(len(Date_list[-3])),[-1,1])), axis=1)
    val_ = np.concatenate((np.array(Date_list[-2]), np.reshape(np.ones(len(Date_list[-2])),[-1,1])), axis=1)
    test_ = np.concatenate((np.array(Date_list[-1]), np.reshape(np.ones(len(Date_list[-1])),[-1,1])), axis=1)

    np.savetxt(root+'train.txt', train_, fmt="%d", delimiter=',')
    np.savetxt(root+'test.txt', test_, fmt="%d", delimiter=',')
    np.savetxt(root+'val.txt', val_, fmt="%d", delimiter=',')

    print('Time steps:', len(Date_list))

    UserItemNet = []
    for date_graph in Date_list:
        date_graph_ = np.array([[eg[0], eg[1]] for eg in date_graph])
        # print(max(np.array(date_graph_)[:, 0]), max(np.array(date_graph_)[:, 1]))
        UserItemNet.append(csr_matrix((np.ones(len(date_graph_)), (np.array(date_graph_)[:, 0], np.array(date_graph_)[:, 1])),
                                 shape=(u_map, p_map)))


    with open(save_path, "wb") as f:
        pkl.dump(UserItemNet, f)
        print("Processed Data Saved at {}".format(save_path))


