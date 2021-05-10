"""
Put this file into the `RecBole/recbole/utils/` directory.

"""

import numpy as np
import torch
from recbole.utils.case_study import full_sort_topk
from recbole.utils.get_popularity import  get_popularity


def calculate_nov_and_cov(train_data, test_data, config, model):
    """Calculate novelty and coverage metric.
    """
    topk = config['topk']
    eval_batch_size = config['eval_batch_size']
    topk_list = []
    batch_size = eval_batch_size / test_data.item_num
    uid_batch = np.array_split(test_data.uid_list, len(test_data.uid_list) // batch_size, axis=0)
    for uid in uid_batch:
        _, topk_iid_list = full_sort_topk(uid, model, test_data, max(topk))
        topk_list.append(topk_iid_list)
    topk_matrix = torch.cat(topk_list, dim=0)
    item_popularity = get_popularity(train_data, config).type(torch.float)
    item_popularity = torch.log2(item_popularity)
    user_item_pop = item_popularity[topk_matrix]
    cumsum_pop = user_item_pop.cumsum(axis=1)
    user_list_len = torch.arange(1, topk_matrix.shape[1] + 1, device=config['device'])
    novelty_metric = (cumsum_pop / user_list_len).mean(axis=0).cpu().numpy()
    metric_dict = {}
    for k in topk:
        key = '{}@{}'.format('novelty', k)
        metric_dict[key] = round(novelty_metric[k - 1], config['metric_decimal_place'])

    for k in topk:
        key = '{}@{}'.format('coverage', k)
        coverage_metric = (topk_matrix[:, :k].unique().shape[0] / test_data.item_num)
        metric_dict[key] = round(coverage_metric, config['metric_decimal_place'])

    return metric_dict
