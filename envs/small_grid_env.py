import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from collections import deque
from envs.atsc_env import PhaseMap, PhaseSet, TrafficSimulator
from envs.small_grid_data.build_file import gen_rou_file

sns.set_color_codes()

STATE_NAMES = ['wave']
PHASE_NUM = 4

EXTENDED_LANES = {}

class SmallGridPhase(PhaseMap):
    def __init__(self):
        phases = ['GGrrrrGGrrrr', 'rrGrrrrrGrrr', 'rrrGGrrrrGGr',
                  'rrrrrGrrrrrG']
        self.phases = {PHASE_NUM: PhaseSet(phases)}
        
class SmallGridController:
    def __init__(self, node_names):
        self.name = 'greedy'
        self.node_names = node_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))
    
class SmallGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.peak_flow1 = config.getint('peak_flow1')
        self.peak_flow2 = config.getint('peak_flow2')
        self.init_density = config.getfloat('init_density')
        super().__init__(config, output_path, is_record, record_stat, port=port)
        self.later_group = [i for i in range(self.n_node) if i % 2 != 0]
        
    def _get_node_phase_id(self, node_name):
        return PHASE_NUM
    
    def _init_neighbor_map(self):
        nodes = ['1', '2', '5', '6']
        neighbor_map = {}
        # neighbor_map['1'] = ['2', '5']
        # neighbor_map['2'] = ['1', '6']
        # neighbor_map['5'] = ['1', '6']
        # neighbor_map['6'] = ['2', '5']
        
        neighbor_map['1'] = []
        neighbor_map['2'] = []
        neighbor_map['5'] = []
        neighbor_map['6'] = []
        self.neighbor_map = neighbor_map
        self.neighbor_mask = np.zeros((self.n_node, self.n_node))
        # for i in range(self.n_node):
        #     for nnode in neighbor_map['nt%d' % (i+1)]:
        #         ni = self.node_names.index(nnode)
        #         self.neighbor_mask[i, ni] = 1
        for i, n in enumerate(nodes):
            for nnode in neighbor_map[n]:
                ni = self.node_names.index(nnode)
                # self.neighbor_mask[i, ni] = 1
                self.neighbor_mask[i, ni] = 0
        logging.info('neighbor mask:\n %r' % self.neighbor_mask)
        
    def _init_distance_map(self):
        block0 = np.array([[0,1],[1,0]])
        block1 = block0 + 1
        row0 = np.hstack([block0, block1])
        row1 = np.hstack([block1, block0])
        self.distance_mask = np.vstack([row0, row1])
        
    def _init_map(self):
        self.node_names = ['1', '2', '5', '6']
        self.n_node = 4
        self._init_neighbor_map()
        # for spatial discount
        self._init_distance_map()
        self.max_distance = 2
        self.phase_map = SmallGridPhase()
        self.state_names = STATE_NAMES
        self.extended_lanes = EXTENDED_LANES
        
    def _init_sim_config(self, seed):
        return gen_rou_file(self.data_path,
                            self.peak_flow1,
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
        