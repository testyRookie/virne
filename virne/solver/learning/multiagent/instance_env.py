# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================

import numpy as np
import networkx as nx
from gym import spaces
from virne.solver.learning.rl_base import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from virne.solver.learning.obs_handler import POSITIONAL_EMBEDDING_DIM, P_NODE_STATUS_DIM, V_NODE_STATUS_DIM


class InstanceRLEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(InstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        # self.calcuate_graph_metrics(degree=True, closeness=False, eigenvector=False, betweenness=False)

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_node_obs = self._get_v_node_obs()
        return {
            'p_net_x': p_net_obs['x'],
            'v_net_x': v_node_obs['x'],
            'curr_v_node_id': self.curr_v_node_id,
            'v_net_size': self.v_net.num_nodes,
            'action_mask': self.generate_action_mask(),
        }

    def _get_p_net_obs(self, ):
        attr_type_list = ['resource', 'extrema']
        v_node_min_link_demend = self.obs_handler.get_v_node_min_link_demend(self.v_net, self.curr_v_node_id)
        # node data
        p_node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        p_nodes_status = np.zeros((self.p_net.num_nodes, 1), dtype=np.float32)
        selected_p_nodes = list(self.solution['node_slots'].values())
        p_nodes_status[selected_p_nodes, 0] = 1.
        v2p_node_link_demand = self.obs_handler.get_v2p_node_link_demand(self.p_net, self.v_net, self.solution['node_slots'], self.curr_v_node_id, self.link_attr_benchmarks)
        node_data = np.concatenate((p_node_data, p_node_link_sum_resource, p_nodes_status), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        # data
        p_net_obs = {
            'x': node_data,
        }
        return p_net_obs

    def _get_v_node_obs(self):
        if self.curr_v_node_id  >= self.v_net.num_nodes:
            return []
        norm_unplaced = (self.v_net.num_nodes - (self.curr_v_node_id + 1)) / self.v_net.num_nodes

        node_demand = []
        for n_attr in self.v_net.get_node_attrs('resource'):
            node_demand.append(self.v_net.nodes[self.curr_v_node_id][n_attr.name] / self.node_attr_benchmarks[n_attr.name])
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        # max_link_demand = []
        # mean_link_demand = []
        sum_link_demand = []
        num_neighbors = len(self.v_net.adj[self.curr_v_node_id]) / self.v_net.num_nodes
        for l_attr in self.v_net.get_link_attrs('resource'):
            link_demand = [self.v_net.links[(n, self.curr_v_node_id)][l_attr.name] for n in self.v_net.adj[self.curr_v_node_id]]
            # max_link_demand.append(max(link_demand) / self.link_attr_benchmarks[l_attr.name])
            # mean_link_demand.append((sum(link_demand) / len(link_demand)) / self.link_attr_benchmarks[l_attr.name])
            sum_link_demand.append((sum(link_demand)) / self.link_sum_attr_benchmarks[l_attr.name])

        node_data = np.concatenate([norm_node_demand, sum_link_demand, [norm_unplaced]], axis=0)
        return {
            'x': node_data
        }

    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        weight = (1 / self.v_net.num_nodes)
        if solution['result']:
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = weight
        else:
            reward = - weight
        self.solution['v_net_reward'] += reward
        return reward


class InnerInstanceRLEnv(JointPRStepInstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        kwargs['if_allow_constraint_violation'] = True
        super(InnerInstanceRLEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        # self.calcuate_graph_metrics(degree=True, closeness=False, eigenvector=False, betweenness=False)
        self.total_bids = 1000.0
    
    def get_observation(self):
        return self.get_embedding_observation(), self.get_auction_observation()

    def get_embedding_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_node_obs = self._get_v_node_obs()
        return {
            'p_net_x': p_net_obs['x'],
            'v_net_x': v_node_obs['x'],
            'curr_v_node_id': self.curr_v_node_id,
            'v_net_size': self.v_net.num_nodes,
            'action_mask': self.generate_action_mask(),
        }

    def get_auction_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_node_obs = self._get_v_node_obs()
        v_net_obs = self._get_v_net_obs()

        return {
            'p_net_x': p_net_obs['x'],
            'v_net_x': v_node_obs['x'],
            'scalar': np.concatenate(([self.curr_v_node_id], [self.v_net.num_nodes], [self.total_bids]), axis=-1),
            'curr_v_node_id': self.curr_v_node_id,
            'v_net_size': self.v_net.num_nodes,
            'total_bids': self.total_bids,
        }

    def _get_p_net_obs(self, ):
        attr_type_list = ['resource', 'extrema']
        v_node_min_link_demend = self.obs_handler.get_v_node_min_link_demend(self.v_net, self.curr_v_node_id)
        # node data
        p_node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        p_nodes_status = np.zeros((self.p_net.num_nodes, 1), dtype=np.float32)
        selected_p_nodes =[]
        for value in self.solution['node_slots'].values():
            if value in self.p_net.node_to_index.keys():
                selected_p_nodes.append(self.p_net.node_to_index[value])
        p_nodes_status[selected_p_nodes, 0] = 1.
        v2p_node_link_demand = self.obs_handler.get_v2p_node_link_demand(self.p_net, self.v_net, self.solution['node_slots'], self.curr_v_node_id, self.link_attr_benchmarks)
        node_data = np.concatenate((p_node_data, p_node_link_sum_resource, p_nodes_status), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        # data
        p_net_obs = {
            'x': node_data,
        }
        return p_net_obs

    def _get_v_node_obs(self):
        if self.curr_v_node_id  >= self.v_net.num_nodes:
            return []
        norm_unplaced = (self.v_net.num_nodes - (self.curr_v_node_id + 1)) / self.v_net.num_nodes

        node_demand = []
        for n_attr in self.v_net.get_node_attrs('resource'):
            node_demand.append(self.v_net.nodes[self.curr_v_node_id][n_attr.name] / self.node_attr_benchmarks[n_attr.name])
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        # max_link_demand = []
        # mean_link_demand = []
        sum_link_demand = []
        num_neighbors = len(self.v_net.adj[self.curr_v_node_id]) / self.v_net.num_nodes
        for l_attr in self.v_net.get_link_attrs('resource'):
            link_demand = [self.v_net.links[(n, self.curr_v_node_id)][l_attr.name] for n in self.v_net.adj[self.curr_v_node_id]]
            # max_link_demand.append(max(link_demand) / self.link_attr_benchmarks[l_attr.name])
            # mean_link_demand.append((sum(link_demand) / len(link_demand)) / self.link_attr_benchmarks[l_attr.name])
            sum_link_demand.append((sum(link_demand)) / self.link_sum_attr_benchmarks[l_attr.name])

        node_data = np.concatenate([norm_node_demand, sum_link_demand, [norm_unplaced]], axis=0)
        return {
            'x': node_data
        }

    def _get_v_net_obs(self):
        attr_type_list = ['resource']
        v_node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        v_nodes_status = np.zeros((self.v_net.num_nodes, 1), dtype=np.float32)
        selected_v_nodes = [vnode for vnode in range(self.curr_v_node_id)]
        v_nodes_status[selected_v_nodes, 0] = 1.
        node_data = np.concatenate((v_node_data, v_node_link_sum_resource, v_nodes_status), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.v_net)
        # data
        v_net_obs = {
            'x': node_data,
        }
        return v_net_obs

    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        weight = (1 / self.v_net.num_nodes)
        if solution['result']:
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = weight
        else:
            reward = - weight
        self.solution['v_net_reward'] += reward
        return reward

    def generate_action_mask(self):
        candidate_nodes = self.controller.find_candidate_nodes(self.v_net, self.p_net, self.curr_v_node_id, filter=self.selected_p_net_nodes)
        # candidate_nodes = self.controller.find_feasible_nodes(self.p_net, self.v_net, self.curr_v_node_id, self.solution['node_slots'])
        mask = np.zeros(self.num_actions, dtype=bool)
        # add special actions
        if self.allow_rejection:
            candidate_nodes.append(self.rejection_action)
        if self.allow_revocable:
            if self.num_placed_v_net_nodes != 0:
                candidate_nodes.append(self.revocable_action)
            revoked_actions = self.revoked_actions_dict[(str(self.solution.node_slots), self.curr_v_node_id)]
            [candidate_nodes.remove(a_id) for a_id in revoked_actions if a_id in revoked_actions]
        for node_id in candidate_nodes:
            mask[self.p_net.node_to_index[node_id]] = True
        # mask[candidate_nodes] = True
        # if mask.sum() == 0: 
            # mask[0] = True
        return mask
    
    def step_bid(self, bid, p_node_id):
        self.total_bids -= bid[0]
        self.solution['node_slots'][self.curr_v_node_id] = p_node_id
        return self.get_observation(), (self.total_bids == 0.0)





class AuctionInstanceEnv(JointPRStepInstanceRLEnv):
    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(AuctionInstanceEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)
        num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource', 'extrema']))
        num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource', 'extrema']))
        num_p_net_features = num_p_net_node_attrs + 1
        self.total_bids = 1000.0

    def compute_reward(self, solution):
        """Calculate deserved reward according to the result of taking action."""
        if solution['result']:
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = 1 / self.v_net.num_nodes
        else:
            reward = -1 / self.v_net.num_nodes
        self.solution['v_net_reward'] += reward
        return reward

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_node_obs = self._get_v_node_obs()
        v_net_obs = self._get_v_net_obs()

        return {
            'p_net_x': p_net_obs['x'],
            'v_net_x': v_node_obs['x'],
            'scalar': np.concatenate(([self.curr_v_node_id], [self.v_net.num_nodes], [self.total_bids]), axis=-1),
            'curr_v_node_id': self.curr_v_node_id,
            'v_net_size': self.v_net.num_nodes,
            'total_bids': self.total_bids,
        }

    def _get_p_net_obs(self, ):
        attr_type_list = ['resource', 'extrema']
        v_node_min_link_demend = self.obs_handler.get_v_node_min_link_demend(self.v_net, self.curr_v_node_id)
        # node data
        p_node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        p_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        p_nodes_status = np.zeros((self.p_net.num_nodes, 1), dtype=np.float32)
        selected_p_nodes = list(self.solution['node_slots'].values())
        p_nodes_status[selected_p_nodes, 0] = 1.
        v2p_node_link_demand = self.obs_handler.get_v2p_node_link_demand(self.p_net, self.v_net, self.solution['node_slots'], self.curr_v_node_id, self.link_attr_benchmarks)
        node_data = np.concatenate((p_node_data, p_node_link_sum_resource, p_nodes_status), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        # data
        p_net_obs = {
            'x': node_data,
        }
        return p_net_obs

    def _get_v_net_obs(self):
        attr_type_list = ['resource']
        v_node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=attr_type_list, node_attr_benchmarks=self.node_attr_benchmarks)
        v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=attr_type_list, aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        v_nodes_status = np.zeros((self.v_net.num_nodes, 1), dtype=np.float32)
        selected_v_nodes = list(self.solution['node_slots'].values())
        v_nodes_status[selected_v_nodes, 0] = 1.
        node_data = np.concatenate((v_node_data, v_node_link_sum_resource, v_nodes_status), axis=-1)
        edge_index = self.obs_handler.get_link_index_obs(self.v_net)
        # data
        v_net_obs = {
            'x': node_data,
        }
        return v_net_obs

    def _get_v_node_obs(self):
        if self.curr_v_node_id  >= self.v_net.num_nodes:
            return []
        norm_unplaced = (self.v_net.num_nodes - (self.curr_v_node_id + 1)) / self.v_net.num_nodes

        node_demand = []
        for n_attr in self.v_net.get_node_attrs('resource'):
            node_demand.append(self.v_net.nodes[self.curr_v_node_id][n_attr.name] / self.node_attr_benchmarks[n_attr.name])
        norm_node_demand = np.array(node_demand, dtype=np.float32)

        # max_link_demand = []
        # mean_link_demand = []
        sum_link_demand = []
        num_neighbors = len(self.v_net.adj[self.curr_v_node_id]) / self.v_net.num_nodes
        for l_attr in self.v_net.get_link_attrs('resource'):
            link_demand = [self.v_net.links[(n, self.curr_v_node_id)][l_attr.name] for n in self.v_net.adj[self.curr_v_node_id]]
            # max_link_demand.append(max(link_demand) / self.link_attr_benchmarks[l_attr.name])
            # mean_link_demand.append((sum(link_demand) / len(link_demand)) / self.link_attr_benchmarks[l_attr.name])
            sum_link_demand.append((sum(link_demand)) / self.link_sum_attr_benchmarks[l_attr.name])

        node_data = np.concatenate([norm_node_demand, sum_link_demand, [norm_unplaced]], axis=0)
        return {
            'x': node_data
        }

    def place_bids(self):
        """
        Place bids for each node in the v_net.
        
        Returns:
            list: A list of float values representing the bids for each node.
        """
        num_v_nodes = self.v_net.num_nodes
        total_bid_amount = 1000.0
        bids = np.random.dirichlet(np.ones(num_v_nodes)) * total_bid_amount
        return bids.tolist()
    
    def step(self, action):
        """
        Joint Place and Route with action p_net node.

        All possible case
            Uncompleted Success: (Node place and Link route successfully)
            Completed Success: (Node Mapping & Link Mapping)
            Falilure: (Node place failed or Link route failed)
        """
        self.solution['num_interactions'] += 1
        p_node_id = int(action)
        self.solution.selected_actions.append(p_node_id)
        if self.solution['num_interactions'] > 10 * self.v_net.num_nodes:
            # self.solution['description'] += 'Too Many Revokable Actions'
            return self.reject()
        # Case: Reject
        if self.if_rejection(action):
            return self.reject()
        # Case: Revoke
        if self.if_revocable(action):
            return self.revoke()
        # Case: reusable = False and place in one same node
        elif not self.reusable and (p_node_id in self.selected_p_net_nodes):
            self.solution['place_result'] = False
            solution_info = self.counter.count_solution(self.v_net, self.solution)
            done = True
            # solution_info = self.solution.to_dict()
        # Case: Try to Place and Route
        else:
            assert p_node_id in list(self.p_net.nodes)
            place_and_route_result, place_and_route_info = self.controller.place_and_route(
                                                                                self.v_net, 
                                                                                self.p_net, 
                                                                                self.curr_v_node_id, 
                                                                                p_node_id,
                                                                                self.solution, 
                                                                                shortest_method=self.shortest_method, 
                                                                                k=self.k_shortest,
                                                                                if_allow_constraint_violation=self.if_allow_constraint_violation)
            # Step Failure
            if not place_and_route_result:
                if self.allow_revocable and self.solution['num_interactions'] <= self.v_net.num_nodes * 10:
                    self.solution['selected_actions'].append(self.revocable_action)
                    return self.revoke()
                else:
                    solution_info = self.counter.count_solution(self.v_net, self.solution)
                    done = True
    
                # solution_info = self.solution.to_dict()
            else:
                # VN Success ?
                if self.num_placed_v_net_nodes == self.v_net.num_nodes:
                    self.solution['result'] = True
                    solution_info = self.counter.count_solution(self.v_net, self.solution)
                    done = True
                # Step Success
                else:
                    done = False
                    solution_info = self.counter.count_partial_solution(self.v_net, self.solution)
                    
        if done:
            pass
        
        # print(f'{t2-t1:.6f}={t3-t1:.6f}+{t2-t3:.6f}')
        return self.get_observation(), self.compute_reward(self.solution), done, self.get_info(solution_info)