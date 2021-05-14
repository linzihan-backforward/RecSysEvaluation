"""
Put this file into the `RecBole/recbole/utils/` directory.

This file is used to calculate the number of invalid recommended items.
"""


import numpy as np
import torch
from recbole.utils.case_study import full_sort_topk
from recbole.utils.get_timestamp import get_timestamp


def invalid_item_num(train_data, test_data, config, model):
    topk = config['topk']
    eval_batch_size = config['eval_batch_size']
    topk_list = []
    batch_size = eval_batch_size / test_data.item_num
    uid_batch = np.array_split(test_data.uid_list, len(test_data.uid_list) // batch_size, axis=0)

    for uid in uid_batch:
        _, topk_iid_list = full_sort_topk(uid, model, test_data, max(topk))
        topk_list.append(topk_iid_list)
    topk_matrix = torch.cat(topk_list, dim=0)
    user_id = test_data.uid_list
    all_user_time, all_item_time = get_timestamp(train_data, config)
    user_time = all_user_time[user_id]
    item_time = all_item_time[topk_matrix]
    true_index = (item_time < user_time.unsqueeze(dim=1))
    cumsum_item_num = true_index.cumsum(dim=1)
    # max_k = torch.arange(1, topk_matrix.shape[1] + 1, device=config['device'])
    availability = cumsum_item_num.sum(axis=0).cpu().numpy()

    metric_dict = {}
    for k in topk:
        key = '{}@{}'.format('Avai', k)
        metric_dict[key] = round(availability[k - 1], config['metric_decimal_place'])

    return metric_dict
