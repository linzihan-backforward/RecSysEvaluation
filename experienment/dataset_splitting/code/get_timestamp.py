"""
Put this file into the `RecBole/recbole/utils/` directory.

This file is used to get the first time an item appears and the time of the last interaction from a user.
"""

import torch
import numpy as np


class TimeCounter(object):
    r"""Pop is an fundamental model that always recommend the most popular item.

    """

    def __init__(self, config, dataset):
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.TIMESTAMP = config['TIME_FIELD']
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_users = dataset.num(self.USER_ID)
        self.device = config['device']
        self.user_time = torch.full((self.n_users,), -np.inf, dtype=torch.float, device=self.device,
                                    requires_grad=False)
        self.item_time = torch.full((self.n_items,), np.inf, dtype=torch.float, device=self.device,
                                    requires_grad=False)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        timestamp = interaction[self.TIMESTAMP]

        self.user_time[user] = torch.max(self.user_time[user], timestamp)
        self.item_time[item] = torch.min(self.item_time[item], timestamp)

        return torch.nn.Parameter(torch.zeros(1))


def get_timestamp(train_data, config):
    counter = TimeCounter(config, train_data)
    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(config['device'])
        loss = counter.calculate_loss(interaction)

    return counter.user_time, counter.item_time
