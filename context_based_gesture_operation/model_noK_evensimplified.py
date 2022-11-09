import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt
import pandas

from pymc3_lib import *

import matplotlib.pyplot as plt
from copy import deepcopy

class MappingBayesNet_EvenSimplified():
    def __init__(self, scene_state):
        self.objects = scene_state['objects'] # Object data
        self.obj_types = scene_state['obj_types'] #types of objects
        self.A = scene_state['A'] # Action names
        self.G = scene_state['G'] # Possible gestures
        self.U = scene_state['User']  # User names
        self.CU = scene_state['User_C'] #current user

        self.O = list(self.objects.keys()) # Object names


        # Conditional prob tables (mapping actions to gestures for individual users and objects)
        self.CM = np.zeros((3,3)) # actions x gestures
        # zero user x cup x (g1 ,g2,g3, ... )
        self.CM[:,:] = [[1  , 0, 0  ],
                        [0  , 1, 0  ],
                        [0  , 0, 1  ]]
        self.policy_history = []

    @property
    def policy(self):
        return self.policy_history[-1]

    def create_observation(self, action=None):
        if action is not None:
            self.TA = self.A.index(action[0])
            self.TO = action[1]
        self.observation = {}
        self.observation['focus_point'] = np.zeros(3)
        self.observation['gesture_vec'] = np.zeros(3)

        #generate focus point - add noise to the target object location
        self.observation['focus_point'] = abs(np.random.normal(loc = self.objects[self.TO]['position'], scale = 0.1,size =(1,3)))
        gesture_probs_intent = self.CM[self.TA,:]
        if (gesture_probs_intent == np.zeros(3)).all():
            print('ERROR: Infeasible action, CPT probs zeros')
            return False
        performed_gesture = np.random.choice(np.arange(len(self.G)),p = gesture_probs_intent,size=1)
        self.observation['gesture_vec'][performed_gesture] = 1
        self.observation['gesture_vec'] = abs(self.observation['gesture_vec'] + np.random.normal(loc = 0, scale = 0.2,size =(len(self.observation['gesture_vec']))))
        self.observation['gesture_vec'] = self.observation['gesture_vec']/np.sum(self.observation['gesture_vec'])
        return self.observation

    def init_TaTo(self, TaTo):
        self.TA = self.A.index(TaTo[0]) #target action
        self.TO = TaTo[1] #target object
        self.obj_idx = self.obj_types.index(self.objects[self.TO]['type'])

    def init_policy_simple(self, policy=None):
        if policy is not None:
            self.policy_history.append(policy)
        else:
            policy = {}
            policy['CM_est'] = np.diag(np.full(3,1))
            policy['i'] = 'init'
            policy['r'] = False
            policy['permutation'] = 0

            self.policy_history.append(deepcopy(policy))
        return policy

    def select_action(self):
        action = {}
        action['target_object'] = 0
        action['target_action'] = 0

        dist_vec = []
        for obj_name in self.O:
            dist_vec.append( np.linalg.norm(self.objects[obj_name]['position'] - self.observation['focus_point']) )
        idx = np.argmin(dist_vec)
        action['target_object'] = self.O[idx]
        action['target_action'] = self.A[np.argmax(np.matmul(self.policy['CM_est'],self.observation['gesture_vec']))]

        self.policy_history[-1]['action'] = action
        return action

    def policy_step(self, reward=0., type='random', out=False):
        if out: print(f"r: {reward}")
        policy = deepcopy(self.policy_history[-1]) # new policy
        policy['r'] = reward
        self.policy_history.append(deepcopy(policy))
        return policy


    def policy_update(self, reward=0., type='random', out=False):
        if len(self.policy_history) > 1:
            reward_dif = reward - self.policy_history[-1]['r']
        else:
            reward_dif = 0.00000001

        policy = deepcopy(self.policy_history[-1]) # new policy
        policy['r'] = reward
        if out: print(f"r: {reward}, diff: {reward_dif}")
        if reward_dif >= 0:
            r1,r2 = np.random.choice(len(self.A), 2, replace=False)
            policy['CM_est'][[r1,r2]] = policy['CM_est'][[r2,r1]]
            policy['i'] = 'forward'
        else: # Rewrites the policy from history, it has its own reward tag saved
            policy = deepcopy(self.policy_history[-2])
            policy['i'] = 'revert'

        self.policy_history.append(deepcopy(policy))
        return policy

    def print_policy(self):
        print(f"total: {len(self.policy_history)}")
        for i in range(len(self.policy_history)):
            print(f"{i}, r: {self.policy_history[i]['r']} \t-> {self.policy_history[i]['i']}, ({self.policy_history[i]['action']['target_action']}, {self.policy_history[i]['action']['target_object']})")
        print(f"d: done")






















#
