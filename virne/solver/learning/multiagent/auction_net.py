# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from ..neural_network import GATConvNet, GCNConvNet, ResNetBlock, MLPNet


class AuctionActorCritic(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim=64, dropout_prob=0., batch_norm=False):
        super(AuctionActorCritic, self).__init__()
        self.actor = Actor(p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.actor_target = Actor(p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic_target = Critic(p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim, dropout_prob=dropout_prob, batch_norm=batch_norm)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def act(self, obs):
        return self.actor(obs)
    
    def act_target(self, obs):
        return self.actor_target(obs)

    def evaluate(self, obs):
        return self.critic(obs)

    def evaluate_actions(self, obs, actions):
        obs['total_bids'] -= actions
        for i in range(actions.shape[0]):
            obs['scalar'][i][2] -= actions[i]
        return self.critic(obs)
    
    def evaluate_actions_target(self, obs, actions):
        obs['total_bids'] -= actions
        for i in range(actions.shape[0]):
            obs['scalar'][i][2] -= actions[i]
        return self.critic_target(obs)


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim=64, dropout_prob=0., batch_norm=False):
        super(Actor, self).__init__()
        self.p_mlp = MLPNet(p_net_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.v_mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.scalar_mlp = MLPNet(3, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.att = nn.MultiheadAttention(embedding_dim, num_heads=1, batch_first=True)
        self.lin_fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs):
        """Return logits of actions"""
        p_node_embeddings = self.p_mlp(obs['p_net_x'])
        v_node_embedding = self.v_mlp(obs['v_net_x'])
        scalars_embedding = self.scalar_mlp(obs['scalar'])
        init_fusion_embeddings = p_node_embeddings + v_node_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1) \
            + scalars_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1)

        # init_fusion_embeddings = v_node_embedding + nn.AdaptiveAvgPool1d(v_node_embedding.shape[1])(p_node_embeddings.transpose(1, 2)).transpose(1, 2)
        att_fusion_embeddings, _ = self.att(init_fusion_embeddings, init_fusion_embeddings, init_fusion_embeddings)
        fusion_embeddings = att_fusion_embeddings + init_fusion_embeddings
        pool_embeddings = nn.AdaptiveAvgPool1d(1)(fusion_embeddings.transpose(1, 2)).squeeze(-1)  
        action_logits = self.lin_fusion(pool_embeddings).squeeze(-1)
        return action_logits * obs['total_bids']


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_node_feature_dim, embedding_dim=64, dropout_prob=0., batch_norm=False):
        super(Critic, self).__init__()
        self.p_mlp = MLPNet(p_net_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.v_mlp = MLPNet(v_node_feature_dim, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        self.att = nn.MultiheadAttention(embedding_dim, num_heads=1, batch_first=True)
        self.scalar_mlp = MLPNet(3, embedding_dim, num_layers=2, embedding_dims=None, batch_norm=batch_norm, dropout_prob=dropout_prob)
        # self.lin_fusion = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 1),
        # )

    def forward(self, obs):
        """Return logits of actions"""
        p_node_embeddings = self.p_mlp(obs['p_net_x'])
        v_node_embedding = self.v_mlp(obs['v_net_x'])
        scalars_embedding = self.scalar_mlp(obs['scalar'])
        init_fusion_embeddings = p_node_embeddings + v_node_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1) \
            + scalars_embedding.unsqueeze(1).repeat(1, p_node_embeddings.shape[1], 1)
        # init_fusion_embeddings = v_node_embedding + nn.AdaptiveAvgPool1d(v_node_embedding.shape[1])(p_node_embeddings.transpose(1, 2)).transpose(1, 2)
        att_fusion_embeddings, _ = self.att(init_fusion_embeddings, init_fusion_embeddings, init_fusion_embeddings)
        fusion_embeddings = att_fusion_embeddings + init_fusion_embeddings
        pool_embeddings = nn.AdaptiveAvgPool1d(1)(fusion_embeddings.transpose(1, 2)).squeeze(-1)  
        # action_logits = self.lin_fusion(fusion_embeddings).squeeze(-1)
        # value = action_logits.mean(dim=1)
        value = fusion_embeddings.mean(dim=1).mean(dim=1).unsqueeze(-1)
        # value = pool_embeddings.mean(dim=1).mean(dim=1).unsqueeze(-1)
        return value
