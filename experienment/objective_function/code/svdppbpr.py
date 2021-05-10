# -*- encoding: utf-8 -*-
# @Time    :   2020/11/15
# @Author  :   Zihan Lin
# @email   :   zhlin@ruc.edu.cn

"""
SVDPP
######################################
Reference:
    Yehuda Koren. "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model." in KDD 2008.

"""

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_normal_initialization


class SVDPPBPR(GeneralRecommender):
    """SVDPP is a user history enhanced SVD model which average user's history information to get an more accurate
     user embedding.

    Note:
        For ranking task on the implicit dataset, we modified the loss function from RMSE loss to BCE loss which is
        different from the original paper.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SVDPPBPR, self).__init__(config, dataset)

        # load dataset info

        # get all users' history interaction information.
        # matrix is padding by the maximum number of a user's interactions
        self.history_item_matrix, _, self.history_lens = dataset.history_item_matrix()
        self.history_item_matrix = self.history_item_matrix.to(self.device)
        self.history_lens = self.history_lens.to(self.device).float()
        # load parameters info
        self.embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']

        # define layers and loss
        # construct source and destination item embedding matrix
        self.b_u = nn.Embedding(self.n_users, 1, padding_idx=0)
        self.b_i = nn.Embedding(self.n_items, 1, padding_idx=0)

        self.p_u = nn.Embedding(self.n_users, self.embedding_size, padding_idx=0)
        self.y_i = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.q_i = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        self.miu = nn.Parameter(torch.rand(1, ), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.bpr_loss = BPRLoss()

        self.apply(xavier_normal_initialization)

    def reg_loss(self):
        """calculate the reg loss for embedding layers and mlp layers

        Returns:
            torch.Tensor: reg loss

        """
        l2_reg = self.b_i.weight.norm(2) + self.b_u.weight.norm(2) + \
                 self.p_u.weight.norm(2) + self.q_i.weight.norm(2) + self.y_i.weight.norm(2)
        loss_l2 = self.reg_weight * l2_reg
        return loss_l2

    def user_forward(self, user):
        """forward the model by user

        Args:
            user (torch.Tensor): user id tensor. shape of [batch_size,]

        Returns:
            torch.Tensor: embedding tensor of user shape of [batch_size, embedding_size]

        """
        pu = self.p_u(user)
        user_inter = self.history_item_matrix[user]
        item_num = self.history_lens[user].unsqueeze(1)  # batch_size x 1
        item_num = torch.sqrt(item_num)  # batch_size x 1
        user_history = self.y_i(user_inter)  # batch_size x max_len x embedding_size
        user_history = torch.sum(user_history, dim=1)  # batch_size x embedding_size
        user_history = torch.div(user_history, item_num)  # batch_size x embedding_size
        u_embedding = pu + user_history
        return u_embedding

    def forward(self, user, item):
        u_embedding = self.user_forward(user)
        score = self.miu + self.b_i(item).squeeze() + self.b_u(user).squeeze() \
                + torch.sum(torch.mul(self.q_i(item), u_embedding), dim=1)
        score = self.sigmoid(score)
        return score.squeeze()

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        pos_scores = self.forward(user, pos_item)
        neg_scores = self.forward(user, neg_item)
        loss = self.bpr_loss(pos_scores, neg_scores) + self.reg_loss()
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embedding = self.user_forward(user)
        all_item_q_i = self.q_i.weight
        inter_score = torch.matmul(u_embedding, all_item_q_i.transpose(0, 1))  # batch_size x n_items
        score = self.b_u(user) + inter_score
        score = score + self.b_i.weight.transpose(0, 1)
        score = score + self.miu
        score = self.sigmoid(score)
        return score.view(-1)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        output = self.forward(user, item)
        return output
