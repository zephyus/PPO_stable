"""
ATSC scenario: Kaohsiung traffic network
@author: Yuchen Luo
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from collections import deque
from envs.atsc_env import PhaseMap, PhaseSet, TrafficSimulator
from envs.kaohsiung_data.build_file import gen_rou_file

sns.set_color_codes()

STATE_NAMES = ['wave']

NODES = {'1031441085': ('3.0', ['281697929', '1031441690', '281697913']),
        '1031441690': ('3.1', ['254581855', '1031441085', '256456400']),
        '1031441843': ('4.0', ['256456400', '258772629', '1031441857']),
        '1031441857': ('4.0', ['1031441843', '256456406', '261601090']),
        '1069364854': ('3.2', ['650370170', '2753105687', '254585719']),
        '1069364967': ('2.0', ['1069365093', '1069365412']),
        '1069365087': ('3.2', ['854896775', '854896793', '254581854']),
        '1069365093': ('2.1', ['254581855', '1069365087']),
        '1069365138': ('2.2', ['255026270']),
        '1069365148': ('3.3', ['1069365138', '2753105680', '259975052']),
        '1069365223': ('3.4', ['271829728', '276377337', '2408412536']),
        '1069365270': ('3.4', ['271829737', '1069365093', '1069365532']),
        '1069365412': ('2.3', ['1069365602', '271829737']),
        '1069365532': ('3.5', ['420582576', '1069365606']),
        '1069365602': ('2.3', ['2145146069', '4186892251']),
        '1069365606': ('4.0', ['265599714', '650370172', '854896775', '1069365532']),
        '1069365670': ('2.4', ['260281837', '854896683']),
        '1093480689': ('3.4', ['4186892251', '1069365270']),
        '2145146069': ('3.6', ['2408412534']),
        '2306420285': ('3.7', ['4191744402', '259974960']),
        '2408412534': ('4.1', ['2408412536', '4186892251', '2145146069', '271829728']),
        '2408412536': ('2.3', ['1093480689', '2408412534']),
        '2408497626': ('4.1', ['276377337', '255378183']),
        '2536001831': ('4.1', ['259975052', '650370182', '2536001834']),
        '2536001834': ('4.0', ['2536001831', '278701745', '650370174']),
        '254531033': ('4.1', ['265599709', '2753105686', '650370183', '650370170']),
        '254581852': ('4.1', ['256456406', '278701744', '261601090']),
        '254581854': ('3.8', ['1069365087', '854896793', '254581854']),
        '254581855': ('3.9', ['281697929', '1031441690']),
        '254585719': ('3.8', ['1069364854', '278701747', '256456773']),
        '255026270': ('4.0', ['650370174', '265599714', '255376591']),
        '255376591': ('2.5', ['255026270', '420582567']),
        '255378181': ('3.5', ['255378183', '420582576']),
        '255378183': ('3.7', ['2408497626', '255378181']),
        '256456400': ('3.1', ['281697913', '259974875', '1031441843']),
        '256456406': ('4.1', ['257929386', '278701748', '1031441857', '254581852']),
        '256456715': ('2.5', ['254581852', '259974960']),
        '256456773': ('4.0', ['254585719', '278701748', '259987575', '4191744403']),
        '256725847': ('3.8', ['854896776', '260281837']),
        '257929386': ('4.2', ['254581854', '278943449', '256456406']),
        '258772629': ('4.1', ['1031441843', '261601090', '258772630']),
        '258772630': ('3.1', ['276452387', '258772629']),
        '258845414': ('2.0', ['4191744405', '278942780']),
        '259974875': ('2.6', ['258772630', '281697912']),
        '259974960': ('4.1', ['588223739', '2306420285', '256456715']),
        '259975052': ('4.0', ['1069365148', '265599709', '2536001831']),
        '259987575': ('4.1', ['256456773', '278701744', '588223739']),
        '260281837': ('3.8', ['854896683', '1069365670']),
        '261601090': ('4.0', ['254581852', '1031441857', '258772629']),
        '262996119': ('3.4', ['4191744402']),
        '265599709': ('4.1', ['259975052', '650370182', '2753105682', '254531033']),
        '265599714': ('4.0', ['255026270', '650370173', '1069365606']),
        '271829728': ('2.5', ['1069365223', '2408412534']),
        '271829737': ('4.1', ['1069365412', '4186892251', '1069365270']),
        '2753105680': ('3.4', ['1069365148']),
        '2753105681': ('3.4', ['2753105686', '331258069']),
        '2753105682': ('3.3', ['278942791', '1069365148', '265599709']),
        '2753105686': ('3.6', ['2753105681', '2753105682', '254531033']),
        '2753105687': ('2.0', ['331258069', '2753105686']),
        '276377337': ('4.0', ['420582567', '2408497626', '1069365223']),
        '276452387': ('2.7', ['258772630', '281697912']),
        '278701744': ('4.3', ['278701748', '259987575', '254581852']),
        '278701745': ('4.1', ['2536001834', '650370182', '650370173', '278701746']),
        '278701746': ('4.0', ['278701745', '650370183', '854897004', '650370172']),
        '278701747': ('3.8', ['854896793', '254581854', '278943449']),
        '278701748': ('4.0', ['278943449', '256456773', '256456406', '278701744']),
        '278942780': ('3.4', ['4191744403', '262996119']),
        '278942791': ('2.3', ['2753105682', '2753105681']),
        '278943449': ('4.0', ['278701747', '257929386', '278701748']),
        '281697912': ('3.0', ['281697913', '259974875', '276452387']),
        '281697913': ('2.7', ['256456400', '1031441085']),
        '281697929': ('2.8', ['1069364967', '256725847']),
        '331258069': ('2.1', ['258845414']),
        '4186892251': ('4.1', ['2408412534', '1093480689', '1069365602', '271829737']),
        '4191744402': ('3.3', ['4191744403', '262996119']),
        '4191744403': ('3.6', ['4191744405', '278942780', '256456773']),
        '4191744405': ('2.9', ['2753105687', '254585719']),
        '420582567': ('4.0', ['255376591', '276377337']),
        '420582576': ('3.5', ['255378181', '1093480689', '1069365532']),
        '588223739': ('4.1', ['259987575', '259974960']),
        '650370170': ('3.5', ['254531033', '854897004', '1069364854']),
        '650370171': ('4.3', ['650370172', '854896775', '854897004']),
        '650370172': ('4.1', ['650370173', '278701746', '1069365606', '650370171']),
        '650370173': ('4.0', ['650370174', '278701745', '265599714', '650370172']),
        '650370174': ('4.1', ['2536001834', '255026270', '650370173']),
        '650370182': ('4.1', ['2536001831', '265599709', '650370183', '278701745']),
        '650370183': ('4.1', ['650370182', '254531033', '278701746']),
        '854896683': ('3.2', ['260281837', '854896776']),
        '854896775': ('3.7', ['1069365606', '650370171', '1069365087']),
        '854896776': ('3.2', ['256725847', '1069364967']),
        '854896793': ('3.2', ['854897004', '1069364854', '278701747']),
        '854897004': ('4.0', ['278701746', '650370171', '854896793', '650370170'])}

PHASES = {'3.0': ['GGgrrrrGgg', 'rrGrrrrrGG', 'rrrGGGGrrr'],
        '3.1': ['GggGGgrrrr', 'rGGrrGrrrr', 'rrrrrrGGGG'],
        '4.0': ['GGggrrrrGGggrrrr', 'rrGGrrrrrrGGrrrr', 'rrrrGGggrrrrGGgg', 'rrrrrrGGrrrrrrGG'],
        '3.2': ['rrrrrGggGGg', 'rrrrrrGGrrG', 'GGGGGrrrrrr'],
        '2.0': ['GGGGrrr', 'rrrrGGG'],
        '2.1': ['GGGGrr', 'rrrrGG'],
        '2.2': ['gG', 'rG'],
        '3.3': ['rrrGggGGg', 'rrrrGGrrG', 'GGGrrrrrr'],
        '3.4': ['GggGGgrrr', 'rGGrrGrrr', 'rrrrrrGGG'],
        '2.3': ['GGrr', 'rrGG'],
        '3.5': ['rrrGGgGgg', 'rrrrrGrGG', 'GGgGrrrrr'],
        '2.4': ['GgrrGg', 'rrGGrr'],
        '3.6': ['GGgrrrGgg', 'rrGrrrrGG', 'rrrGGGrrr'],
        '3.7': ['GggrrrGGg', 'rGGrrrrrG', 'rrrGGgGrr'],
        '4.1': ['rrrrGGggrrrrGGgg', 'rrrrrrGGrrrrrrGG', 'GGggrrrrGGggrrrr', 'rrGGrrrrrrGGrrrr'],
        '3.8': ['GGgrrrrrGgg', 'rrGrrrrrrGG', 'rrrGGGGGrrr'],
        '3.9': ['rrrrGGggg', 'rrrrrrGGG', 'GGGGrrrrr'],
        '2.5': ['GgGr', 'GrGg'],
        '4.2': ['GGggrrrrGGgg', 'rrGGrrrrrrGG', 'rrrrGGGgGrrr'],
        '2.6': ['GGrrr', 'rrGGG'],
        '2.7': ['GGGrr', 'rrrGG'],
        '4.3': ['GGgGggrrr', 'rrGrGGrrr', 'GrrrrrGGg'],
        '2.8': ['GGGGrrrrr', 'rrrrGGGGG'],
        '2.9': ['GGrrrr', 'rrGGGG']}

EXTENDED_LANES = {('254581855', '330465827#3_1'): ['330465827#2_1'],
                  ('254581855', '330465827#3_2'): ['330465827#2_2'],
                  ('254581855', '330465827#3_3'): ['330465827#2_3']}

class KaoNetPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)
            
class KaoNetController:
    def __init__(self, node_names, nodes):
        self.name = 'greedy'
        self.node_names = node_names
        self.nodes = nodes
        
    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions
    
    def greedy(self, ob, node_name):
        # get the action space
        phases = PHASES[NODES[node_name][0]]
        flows = []
        node = self.nodes[node_name]
        # get the green waves
        for phase in phases:
            wave = 0
            visited_ilds = set()
            for i, signal in enumerate(phase):
                if signal == 'G':
                    # find controlled lane
                    lane = node.lanes_in[i]
                    # ild = 'ild:' + lane
                    ild = lane
                    # if it has not been counted, add the wave
                    if ild not in visited_ilds:
                        j = node.ilds_in.index(ild)
                        wave += ob[j]
                        visited_ilds.add(ild)
            flows.append(wave)
        return np.argmax(np.array(flows))
    
class KaoNetEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.flow_rate = config.getint('flow_rate')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _bfs(self, i):
        d = 0
        self.distance_mask[i, i] = d
        visited = [False]*self.n_node
        que = deque([i])
        visited[i] = True
        while que:
            d += 1
            for _ in range(len(que)):
                node_name = self.node_names[que.popleft()]
                for nnode in self.neighbor_map[node_name]:
                    ni = self.node_names.index(nnode)
                    if not visited[ni]:
                        self.distance_mask[i, ni] = d
                        visited[ni] = True
                        que.append(ni)
        return d

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    '''
    The nodes are divided by the descending order with the number of neighbors.
    Note that the members in the same group can not connect with each other.
    '''
    def move_groups(self, nodes: dict):
        later_group = []
        group_2 = []
        descending_nodes = dict(sorted(nodes.items(), key=lambda item: len(item[1][1]), reverse=True))

        for n in descending_nodes:
            group = 1
            for m in later_group:
                if n in nodes[m][1] or len(nodes[n][1]) == 0:
                    group = 2
                    break
            if group == 1:
                later_group.append(n)
            else:
                group_2.append(n)
        return later_group

    def _init_neighbor_map(self):
        later_group = self.move_groups(NODES)
        self.later_group = []
        self.neighbor_map = dict([(key, val[1]) for key, val in NODES.items()])
        self.neighbor_mask = np.zeros((self.n_node, self.n_node)).astype(int)
        for i, node_name in enumerate(self.node_names):
            # divide the nodes into first group and later group for nclm
            if node_name in later_group:
                self.later_group.append(i)
            for nnode in self.neighbor_map[node_name]:
                ni = self.node_names.index(nnode)
                self.neighbor_mask[i, ni] = 1
                
        # temp = self.neighbor_mask
        # self.neighbor_mask = np.linalg.matrix_power(self.neighbor_mask, 2)
        # self.neighbor_mask += temp
        # self.neighbor_mask = np.clip(self.neighbor_mask, 0, 1)
        # np.fill_diagonal(self.neighbor_mask, 0)
        logging.info('neighbor mask:\n %r' % self.neighbor_mask)

    def _init_distance_map(self):
        self.distance_mask = -np.ones((self.n_node, self.n_node)).astype(int)
        self.max_distance = 0
        for i in range(self.n_node):
            self.max_distance = max(self.max_distance, self._bfs(i))
        logging.info('distance mask:\n %r' % self.distance_mask)

    def _init_map(self):
        self.node_names = sorted(list(NODES.keys()))
        self.n_node = len(self.node_names)
        self._init_neighbor_map()
        self._init_distance_map()
        self.phase_map = KaoNetPhase()
        self.phase_node_map = dict([(key, val[0]) for key, val in NODES.items()])
        self.state_names = STATE_NAMES
        self.extended_lanes = EXTENDED_LANES

    def _init_sim_config(self, seed):
        # comment out to call build_file.py
        return gen_rou_file(self.data_path,
                            self.flow_rate,
                            seed=seed,
                            thread=self.sim_thread)

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)