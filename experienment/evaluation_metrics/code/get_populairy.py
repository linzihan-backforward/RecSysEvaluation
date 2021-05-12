"""
Put this file into the `RecBole/recbole/utils/` directory.

This file is used to count the times each item is consumed.
"""

import torch


class PopCounter(object):
    """Count the times each item is consumed.
    """

    def __init__(self, config, dataset):
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.n_items = dataset.num(self.ITEM_ID)
        self.device = config['device']
        self.item_cnt = torch.ones(self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False)
        self.max_cnt = None
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        item = interaction[self.ITEM_ID]
        self.item_cnt[item, :] = self.item_cnt[item, :] + 1

        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]

        return torch.nn.Parameter(torch.zeros(1))


def get_popularity(train_data, config):
    counter = PopCounter(config, train_data)
    for batch_idx, interaction in enumerate(train_data):
        interaction = interaction.to(config['device'])
        loss = counter.calculate_loss(interaction)

    return counter.item_cnt.squeeze()
