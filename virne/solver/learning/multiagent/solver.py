# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import os
import torch
import random
from collections import deque
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

from virne.solver import registry
from .instance_env import InstanceRLEnv, AuctionInstanceEnv, InnerInstanceRLEnv
from .net import ActorCritic
from .auction_net import AuctionActorCritic
from virne.solver.learning.rl_base import RLSolver, PPOSolver, A2CSolver, InstanceAgent, A3CSolver, RolloutBuffer
from ..utils import get_pyg_data
from ..obs_handler import POSITIONAL_EMBEDDING_DIM
from virne.base import Solution, SolutionStepEnvironment, AuctionEnviroment

EMBEDDING_INDEX = 0
AUCTION_INDEX = 1

class DdpgAttentionSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnv)
        PPOSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)

def make_policy(agent, **kwargs):
    num_vn_attrs = agent.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.v_sim_setting_num_link_resource_attrs
    policy = ActorCritic(p_net_num_nodes=agent.p_net_setting_num_nodes, 
                        p_net_feature_dim=num_vn_attrs*2 + num_vl_attrs*2 + 1, 
                        v_node_feature_dim=num_vn_attrs+num_vl_attrs+1,
                        embedding_dim=agent.embedding_dim, 
                        dropout_prob=agent.dropout_prob, 
                        batch_norm=agent.batch_norm).to(agent.device)
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.lr_actor},
            {'params': policy.critic.parameters(), 'lr': agent.lr_critic},
        ], weight_decay=agent.weight_decay
    )
    return policy, optimizer

class AutionSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self, AuctionInstanceEnv)
        PPOSolver.__init__(self, controller, recorder, counter, auction_make_policy, auction_obs_as_tensor, **kwargs)
        self.preprocess_encoder_obs = encoder_obs_to_tensor
    
    def select_action(self, observation, sample=True):
        action = self.policy.act(observation).cpu().data.numpy().flatten()
        if sample:
             action += np.random.normal(0, 0.1, size=action.shape)
        return action.clip(0, observation['total_bids'])
    
    def select_action_target(self, observation, sample=True):
        action = self.policy.act_target(observation).cpu().data.numpy().flatten()
        if sample:
             action += np.random.normal(0, 0.1, size=action.shape)
        return action.clip(0, observation['total_bids'])
    
    def update(self, replay_buffer):
        device = torch.device('cpu')

        # if len(replay_buffer) < self.batch_size:
        if len(replay_buffer) < 2:
            return

        # sample
        # batch = random.sample(replay_buffer, self.batch_size)
        batch = random.sample(replay_buffer, 2)
        batch_state = self.preprocess_obs([t[0] for t in batch], device)
        batch_action = torch.FloatTensor(np.array([t[1] for t in batch])).squeeze(1).to(device)
        batch_reward = torch.FloatTensor(np.array([t[2] for t in batch])).squeeze(1).to(device)
        batch_next_state = self.preprocess_obs([t[3] for t in batch], device)
        batch_done = torch.FloatTensor(np.array([t[4] for t in batch])).to(device)

        # update Critic
        with torch.no_grad():
            target_action = self.select_action(batch_next_state)
            target_q_value = self.policy.evaluate_actions(batch_next_state, target_action).squeeze(-1)
            target_q_value = batch_reward + (1 - batch_done) * self.gamma * target_q_value
        current_q_value = self.policy.evaluate_actions(batch_state, batch_action).squeeze(-1)
        critic_loss = F.mse_loss(current_q_value, target_q_value.detach())

        target_action = self.select_action_target(batch_next_state)
        target_q_value = self.policy.evaluate_actions_target(batch_next_state, target_action).squeeze(-1)
        target_q_value = batch_reward + (1 - batch_done) * self.gamma * target_q_value
        current_q_value = self.policy.evaluate_actions(batch_state, batch_action).squeeze(-1)
        critic_loss = F.mse_loss(current_q_value, target_q_value.detach())

        self.policy.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.policy.critic_optimizer.step()

        # update Actor
        actor_loss = -self.policy.evaluate_actions(batch_state, self.select_action(batch_state)).mean()

        self.policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.policy.actor_optimizer.step()

        for param, target_param in zip(self.policy.critic.parameters(), self.policy.critic_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
        for param, target_param in zip(self.policy.actor.parameters(), self.policy.actor_target.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
    


@registry.register(
    solver_name='multiagentAuction', 
    env_cls=AuctionEnviroment,
    solver_type='r_learning')
class MultiAgentAuctionSolver(InstanceAgent, PPOSolver):
    def __init__(self, controller, recorder, counter, **kwargs):
        InstanceAgent.__init__(self, InstanceRLEnv)
        PPOSolver.__init__(self, controller, recorder, counter, make_policy, obs_as_tensor, **kwargs)
        self.preprocess_encoder_obs = encoder_obs_to_tensor
        self.agnets_num = kwargs['p_net_setting']['agents_num']
        self.agent = {
            'auction': AutionSolver(controller, recorder, counter, **kwargs),
            'embedding': DdpgAttentionSolver(controller, recorder, counter, **kwargs)
        }

    def learn_singly(self, env, num_epochs=1, **kwargs):
        for epoch_id in range(num_epochs):
            print(f'Training Epoch: {epoch_id}') if self.verbose > 0 else None
            self.training_epoch_id = epoch_id

            instance = env.reset()
            success_count = 0
            epoch_logprobs = []
            revenue2cost_list = []

            while True:

                ### --- instance-level --- ###

                v_net, p_net = instance['v_net'], instance['p_net']
                auction_instance_buffer = deque(maxlen=100000)
                embedding_instance_buffer = RolloutBuffer()
                instance_env = self.InstanceEnv(p_net, v_net, self.controller, self.recorder, self.counter, **self.basic_config)
                instance_obs = instance_env.reset()

                inner_instances_env = []
                inner_instances_obs = []

                for agent_id in range(self.agnets_num):
                    inner_env = InnerInstanceRLEnv (
                        p_net.get_subgraph_by_domain(agent_id),
                        v_net,
                        self.agent['embedding'].controller,
                        self.agent['embedding'].recorder,
                        self.agent['embedding'].counter,
                        **self.agent['embedding'].basic_config
                    )

                    inner_instances_env.append(inner_env)
                    inner_instances_obs.append(inner_env.reset())

                while True:
                    curr_v_node_id = instance_obs['curr_v_node_id']
                    
                    ### auction ###
                    bids = []
                    for agent_id in range(self.agnets_num):
                        inner_auction_obs = inner_instances_obs[agent_id][AUCTION_INDEX]
                        tensor_auction_obs = auction_obs_as_tensor(inner_instances_obs[agent_id][AUCTION_INDEX], self.device)
                        action = self.agent['auction'].select_action(tensor_auction_obs, sample=True)
                        bids.append(action)
                        value = self.agent['auction'].estimate_value(tensor_auction_obs)

                    winner = bids.index(max(bids))

                    ### embedding ###
                    inner_embedding_obs = inner_instances_obs[winner][EMBEDDING_INDEX]
                    tensor_embedding_obs = self.agent['embedding'].preprocess_obs(inner_embedding_obs, self.device)
                    action, action_logprob = self.agent['embedding'].select_action(tensor_embedding_obs, sample=True)
                    value = self.agent['embedding'].estimate_value(tensor_embedding_obs)

                    ### step ###
                    next_instance_obs, instance_reward, instance_done, instance_info = instance_env.step(action)

                    next_embedding_obs, embedding_reward, embedding_done, _ = inner_instances_env[winner].step(inner_instances_env[winner].p_net.index_to_node[action])

                    for agent_id in range(self.agnets_num):
                        curr_auction_obs = inner_instances_obs[agent_id][AUCTION_INDEX]
                        inner_instances_obs[agent_id], auction_done = inner_instances_env[agent_id].step_bid(bids[agent_id], action)
                        next_auction_obs = inner_instances_obs[agent_id][AUCTION_INDEX]

                        if agent_id == winner:
                            reward = instance_reward * bids[agent_id] / sum(bids)
                        else:
                            reward = ((1 / bids[agent_id]) / sum([1 / bid for bid in bids])) * instance_reward
                        
                        if auction_done:
                            break
                        
                        auction_instance_buffer.append((curr_auction_obs, bids[agent_id], reward, next_auction_obs, instance_done))

                    embedding_instance_buffer.add(next_embedding_obs[EMBEDDING_INDEX], action, instance_reward, instance_done, action_logprob, value=value)
                    # instance_buffer.action_masks.append(mask if isinstance(mask, np.ndarray) else mask.cpu().numpy())

                    if instance_done or auction_done:
                        break
                    
                    #TODO: update instance_obs
                    instance_obs = next_instance_obs

                ### update modle ###
                self.agent['auction'].update(auction_instance_buffer)

                ### embedding ###
                solution = instance_env.solution
                last_value = self.agent['embedding'].estimate_value(self.agent['embedding'].preprocess_obs(next_embedding_obs[EMBEDDING_INDEX], self.device))


                epoch_logprobs += embedding_instance_buffer.logprobs
                self.agent['embedding'].merge_instance_experience(instance, solution, embedding_instance_buffer, last_value)

                if solution.is_feasible():
                    success_count += 1
                    revenue2cost_list.append(solution['v_net_r2c_ratio'])
                # update parameters
                if self.agent['embedding'].buffer.size() >= self.target_steps:
                    loss = self.agent['embedding'].update()

                instance, reward, done, info = env.step(solution)

                if done:
                    break
                
            epoch_logprobs_tensor = np.concatenate(epoch_logprobs, axis=0)
            print(f'\nepoch {epoch_id:4d}, success_count {success_count:5d}, r2c {info["long_term_r2c_ratio"]:1.4f}, mean logprob {epoch_logprobs_tensor.mean():2.4f}') if self.verbose > 0 else None
            if self.rank == 0:
                # save
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.save_interval == 0:
                    self.save_model(f'model-{epoch_id}.pkl')
                # validate
                if (epoch_id + 1) != num_epochs and (epoch_id + 1) % self.eval_interval == 0:
                    self.validate(env)


def make_policy(agent, **kwargs):
    num_vn_attrs = agent.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.v_sim_setting_num_link_resource_attrs
    policy = ActorCritic(p_net_num_nodes=agent.p_net_setting_num_nodes, 
                        p_net_feature_dim=num_vn_attrs*2 + num_vl_attrs*2 + 1, 
                        v_node_feature_dim=num_vn_attrs+num_vl_attrs+1,
                        embedding_dim=agent.embedding_dim, 
                        dropout_prob=agent.dropout_prob, 
                        batch_norm=agent.batch_norm).to(agent.device)
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.lr_actor},
            {'params': policy.critic.parameters(), 'lr': agent.lr_critic},
        ], weight_decay=agent.weight_decay
    )
    return policy, optimizer

def auction_make_policy(agent, **kwargs):
    num_vn_attrs = agent.v_sim_setting_num_node_resource_attrs
    num_vl_attrs = agent.v_sim_setting_num_link_resource_attrs
    policy = AuctionActorCritic(p_net_num_nodes=agent.p_net_setting_num_nodes, 
                        p_net_feature_dim=num_vn_attrs*2 + num_vl_attrs*2 + 1, 
                        v_node_feature_dim=num_vn_attrs+num_vl_attrs+1,
                        embedding_dim=agent.embedding_dim, 
                        dropout_prob=agent.dropout_prob, 
                        batch_norm=agent.batch_norm).to(agent.device)
    optimizer = torch.optim.Adam([
            {'params': policy.actor.parameters(), 'lr': agent.lr_actor},
            {'params': policy.critic.parameters(), 'lr': agent.lr_critic},
        ], weight_decay=agent.weight_decay
    )
    return policy, optimizer


def encoder_obs_to_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        v_net_x = obs['v_net_x']
        obs_v_net_x = torch.FloatTensor(v_net_x).unsqueeze(dim=0).to(device)
        return {'v_net_x': obs_v_net_x}
    elif isinstance(obs, list):
        obs_batch = obs
        v_net_x_list = []
        for observation in obs:
            v_net_x = obs['v_net_x']
            v_net_x_list.append(v_net_x)
        obs_v_net_x = torch.FloatTensor(np.array(v_net_x_list)).to(device)
        return {'v_net_x': obs_v_net_x}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def obs_as_tensor(obs, device):
    # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        tensor_obs_p_net_x = torch.FloatTensor(np.array([obs['p_net_x']])).to(device)
        tensor_obs_v_net_x = torch.FloatTensor(np.array([obs['v_net_x']])).to(device)
        tensor_obs_curr_v_node_id = torch.LongTensor(np.array([obs['curr_v_node_id']])).to(device)
        tensor_obs_action_mask = torch.FloatTensor(np.array([obs['action_mask']])).to(device)
        tensor_obs_v_net_size = torch.FloatTensor(np.array([obs['v_net_size']])).to(device)
        return {'p_net_x': tensor_obs_p_net_x, 'v_net_x': tensor_obs_v_net_x, 'curr_v_node_id': tensor_obs_curr_v_node_id, 'action_mask': tensor_obs_action_mask, 'v_net_size': tensor_obs_v_net_size}
    # batch
    elif isinstance(obs, list):
        p_net_x_list, v_net_x_list, v_net_size_list, curr_v_node_id_list, action_mask_list = [], [], [], [], []
        max_nodes = 0
        for observation in obs:
            p_net_x_list.append(observation['p_net_x'])
            max_nodes = max(max_nodes, observation['p_net_x'].shape[0])
            v_net_x_list.append(observation['v_net_x'])
            v_net_size_list.append(observation['v_net_size'])
            curr_v_node_id_list.append(observation['curr_v_node_id'])
            action_mask_list.append(observation['action_mask'])
        p_net_x_list = [np.pad(p_net_x, ((0, max_nodes - p_net_x.shape[0]), (0, 0)), mode='constant', constant_values=0) for p_net_x in p_net_x_list]
        action_mask_list = [np.pad(action_mask, ((0, max_nodes - action_mask.shape[0])), mode='constant', constant_values=1) for action_mask in action_mask_list]
        tensor_obs_p_net_x = torch.FloatTensor(np.array(p_net_x_list)).to(device)
        tensor_obs_v_net_x = torch.FloatTensor(np.array(v_net_x_list)).to(device)
        tensor_obs_v_net_size = torch.FloatTensor(np.array(v_net_size_list)).to(device)
        tensor_obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
        tensor_obs_action_mask = torch.FloatTensor(np.array(action_mask_list)).to(device)
        return {'p_net_x': tensor_obs_p_net_x, 'v_net_x': tensor_obs_v_net_x, 'v_net_size': tensor_obs_v_net_size, 'curr_v_node_id': tensor_obs_curr_v_node_id, 'action_mask': tensor_obs_action_mask}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
        raise ValueError('obs type error')

def auction_obs_as_tensor(obs, device):
        # one
    if isinstance(obs, dict):
        """Preprocess the observation to adapte to batch mode."""
        tensor_obs_p_net_x = torch.FloatTensor(np.array([obs['p_net_x']])).to(device)
        tensor_obs_v_net_x = torch.FloatTensor(np.array([obs['v_net_x']])).to(device)
        tensor_obs_curr_v_node_id = torch.LongTensor(np.array([obs['curr_v_node_id']])).to(device)
        tensor_obs_total_bids = torch.FloatTensor(np.array([obs['total_bids']])).to(device)
        tensor_obs_v_net_size = torch.FloatTensor(np.array([obs['v_net_size']])).to(device)
        tensor_scalar = torch.FloatTensor(np.array([obs['scalar']])).to(device)
        return {'p_net_x': tensor_obs_p_net_x, 'v_net_x': tensor_obs_v_net_x, 'curr_v_node_id': tensor_obs_curr_v_node_id, 'total_bids': tensor_obs_total_bids, 'v_net_size': tensor_obs_v_net_size, 'scalar': tensor_scalar}
    # batch
    elif isinstance(obs, list):
        p_net_x_list, v_net_x_list, v_net_size_list, curr_v_node_id_list, total_bids_list, scalar_list = [], [], [], [], [], []
        max_nodes = 0
        for observation in obs:
            p_net_x_list.append(observation['p_net_x'])
            max_nodes = max(max_nodes, observation['p_net_x'].shape[0])
            v_net_x_list.append(observation['v_net_x'])
            v_net_size_list.append(observation['v_net_size'])
            curr_v_node_id_list.append(observation['curr_v_node_id'])
            total_bids_list.append(observation['total_bids'])
            scalar_list.append(observation['scalar'])
        p_net_x_list = [np.pad(p_net_x, ((0, max_nodes - p_net_x.shape[0]), (0, 0)), mode='constant', constant_values=0) for p_net_x in p_net_x_list]
        tensor_obs_p_net_x = torch.FloatTensor(np.array(p_net_x_list)).to(device)
        tensor_obs_v_net_x = torch.FloatTensor(np.array(v_net_x_list)).to(device)
        tensor_obs_v_net_size = torch.FloatTensor(np.array(v_net_size_list)).to(device)
        tensor_obs_curr_v_node_id = torch.LongTensor(np.array(curr_v_node_id_list)).to(device)
        tensor_obs_total_bids = torch.FloatTensor(np.array(total_bids_list)).to(device)
        tensor_scalar_list = torch.FloatTensor(np.array(scalar_list)).to(device)
        return {'p_net_x': tensor_obs_p_net_x, 'v_net_x': tensor_obs_v_net_x, 'v_net_size': tensor_obs_v_net_size, 'curr_v_node_id': tensor_obs_curr_v_node_id, 'total_bids': tensor_obs_total_bids, 'scalar': tensor_scalar_list}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")
        raise ValueError('obs type error')

def split_total_costs(values, total_costs):
    """
    根据列表中的值来分割总成本，列表中的值越大，分割的总成本越小。

    参数:
    values (list): 一个包含分割权重的列表。
    total_costs (float): 总成本。

    返回:
    list: 分割后的成本列表。
    """
    # 计算每个值的反比例
    inverse_values = [1 / value for value in values]

    # 计算反比例的总和
    total_inverse = sum(inverse_values)

    # 根据反比例分配总成本
    costs = [(inverse_value / total_inverse) * total_costs for inverse_value in inverse_values]

    return costs