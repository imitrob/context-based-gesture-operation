import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt
import pandas

from pymc3_lib import *

import matplotlib.pyplot as plt

class MappingBayesNet():
    ''' vK
    >>> MappingBayesNet(objects)
    '''
    def __init__(self, scene_state):
        '''
        >>> self = bn
        '''
        self.objects = scene_state['objects'] # Object data
        self.obj_types = scene_state['obj_types'] #types of objects
        self.A = scene_state['A'] # Action names
        self.G = scene_state['G'] # Possible gestures
        self.U = scene_state['User']  # User names
        self.TA = self.A.index(scene_state['TA']) #target action
        self.TO = scene_state ['TO'] #target object
        self.CU = scene_state['User_C'] #current user

        self.O = list(self.objects.keys()) # Object names
        self.obj_idx = self.obj_types.index(self.objects[self.TO]['type'])


        # Conditional prob tables (mapping actions to gestures for individual users and objects)
        self.CM = np.zeros((2,2, 6,6)) # users x object types x actions x gestures
        # zero user x cup x (g1 ,g2,g3, ... )
        self.CM[0,0,:,:] = [[0.6, 0, 0.4, 0, 0, 0], #two gestures for moving up the cup
                            [0  , 0, 0  , 0, 0, 0], #target action for cup cannot be open or close (zero probs.)
                            [0  , 1, 0  , 0, 0, 0], #same gesture for move front as for open
                            [0  , 0, 0  , 1, 0, 0],
                            [0  , 0, 0  , 0, 0, 0],
                            [0  , 0, 0  , 0, 0, 1]]
        self.CM[1,0,:,:] = [[0.4, 0, 0.6, 0, 0,  0  ],
                            [0  , 0, 0  , 0, 0,  0  ],
                            [0  , 1, 0  , 0, 0,  0  ],
                            [0  , 0, 0  , 1, 0,  0  ],
                            [0  , 0, 0  , 0, 0,  0  ], #target action for cup cannot be open or close
                            [0  , 0, 0  , 0, 0.2,0.8]]
        self.CM[0, 1, :, :] = [[0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]]
        self.CM[1, 1, :, :] = [[0.4, 0, 0.6, 0, 0  , 0  ],
                               [0  , 1, 0  , 0, 0  , 0  ],
                               [0  , 1, 0  , 0, 0  , 0  ],
                               [0  , 0, 0  , 1, 0  , 0  ],
                               [0  , 0, 0  , 0, 0.8, 0.2], #two gestures for closing drawer
                               [0  , 0, 0  , 0, 0  , 0  ]]
        '''
        [[0.7, 0.3, 0.0, 0, 0, 0],
        [0.2, 0.4, 0.4, 0, 0, 0],
        [0.0, 0.4, 0.6, 0, 0, 0],
        [0.0, 0.0, 0.0, 0.2, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.8],
        [0.7, 0.0, 0.0, 0.3, 0.0, 0.0]]
        '''
        self.policy_history = []
        self.reward_history = []

        self.init_policy_simple()

    def create_observation(self):
        self.observation = {}
        self.observation['focus_point'] = np.zeros(3)
        self.observation['gesture_vec'] = np.zeros(6)

        #generate focus point - add noise to the target object location
        self.observation['focus_point'] = abs(np.random.normal(loc = self.objects[self.TO]['position'], scale = 0.1,size =(1,3)))
        gesture_probs_intent = self.CM[self.CU,self.obj_idx,self.TA,:]
        if (gesture_probs_intent == np.zeros(6)).all():
            print('ERROR: Infeasible action, CPT probs zeros')
            return False
        performed_gesture = np.random.choice(np.arange(len(self.G)),p = gesture_probs_intent,size=1)
        self.observation['gesture_vec'][performed_gesture] = 1
        self.observation['gesture_vec'] = abs(self.observation['gesture_vec'] + np.random.normal(loc = 0, scale = 0.2,size =(len(self.observation['gesture_vec']))))
        self.observation['gesture_vec'] = self.observation['gesture_vec']/np.sum(self.observation['gesture_vec'])
        return self.observation

    def init_policy_complex(self):
        self.policy = {}
        self.policy['CM_est'] = np.zeros((2,2, 6,6)) # users x object types x actions x gestures
        self.policy['CM_est'][0, 0, :, :] = np.diag(np.ones([6]))
        self.policy['CM_est'][0, 1, :, :] = np.diag(np.ones([6]))
        self.policy['CM_est'][1, 0, :, :] = np.diag(np.ones([6]))
        self.policy['CM_est'][1, 1, :, :] = np.diag(np.ones([6]))
        return self.policy

    def init_policy_simple(self):#does not take into account influence of user or object
        self.policy = {}
        self.policy['CM_est'] = np.diag(np.full(6,1))
        self.policy_history.append(self.policy)
        return self.policy

    def select_action(self):
        self.action = {}
        self.action['target_object'] = 0
        self.action['target_action'] = 0

        dist_vec = []
        for obj_name in self.O:
            dist_vec.append( np.linalg.norm(self.objects[obj_name]['position'] - self.observation['focus_point']) )
        idx = np.argmin(dist_vec)
        self.action['target_object'] = self.O[idx]
        self.action['target_action'] = self.A[np.argmax(np.matmul(self.policy['CM_est'],self.observation['gesture_vec']))]

        return self.action

    def policy_update(self, reward, type='random', out=False):
        self.reward_history.append(reward)

        if len(self.reward_history) > 1:
            reward_dif = self.reward_history[-1] - self.reward_history[-2]
        else:
            reward_dif = 0.00000001

        if reward_dif >= 0:
            if out: print("Picking new random")
            if out: print(" ======\n", self.policy)
            if type == 'random':
                r1,r2 = np.random.choice(len(self.A), 2, replace=False)
                self.policy['CM_est'][[r1,r2]] = self.policy['CM_est'][[r2,r1]]
            elif type == 'q-learning':
                raise exception("TODO")
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

            if out: print(" to \n", self.policy, "\n ======\n")
        else:
            if out: print("Reverting")
            if out: print(" ======\n", self.policy)
            self.policy = self.policy_history[-2]
            if out: print(" to \n", self.policy, "\n ======\n")

        self.policy_history.append(self.policy.copy())

    # def model(self):
    #     with pm.Model() as self.m:
    #         intent = pm.Categorical('intent', [1/self.n_actions]*self.n_actions) # push, grab
    #         object_intent = pm.Categorical('object_intent', [1/self.n_objects]*self.n_objects) # obj1, obj2
    #
    #         # can be observed
    #         gesture_id_intent = pm.Categorical('gesture_id_intent', switchn(intent, self.mt_gestures))
    #         misc_features = pm.Categorical('misc_features', self.cpt_misc_features.fn(object_intent, intent)) # fits, not fits
    #         parameter_distance = pm.Categorical('parameter_distance', switchn(intent, self.mt_parameter_distance)) # 5, 10, 20
    #         gesture_object_focus = pm.Categorical('gesture_object_focus', self.cpt_gesture_object_focus.fn(intent, object_intent))
    #
    #         #parameter_distance_gestures = pm.Categorical('parameter_distance_gestures', pm.math.switch(parameter_distance, C(0.9), C(0.1)))
    #         eeff_feature = self.eeff__feature()
    #         eeff = pm.Categorical('eeff', pm.math.switch(object_intent, C(eeff_feature[0]), C(eeff_feature[1])))
    #
    #         self.prior_trace = pm.sample_prior_predictive(10000)
    #


class MappingBayesNet_Testing():
    def __init__(self):
        '''
        >>> MappingBayesNet_Testing.test__eeff()
        '''
    @staticmethod
    def load__default():
        scene_state = {
            'A': ['move up','open', 'move front','pour','close','move back'],
            'G': ['G1','G2' ,'G3','G4','G5','G6'],
            'A_exec':['move_upCup1','move_upCup2','openDrawer','moveFrontCup1','moveFrontCup2','pourCup1','pourCup2','closeDrawer','MoveBackCup1','MoveBackCup2'],
            'obj_types':['cup','drawer'],
            'User':['Jan','Mara'],
            'User_C':0,
            'TA':2,
            'TO':'cup2',
            'objects': {
                'cup1': { 'position': [0,1,0.],
                          'type':'cup',
                        'graspable': True,
                        'pushable': True,
                        'free': True,
                        'full': True,
                        'size': 0 # [m]
                },
                'cup2': { 'position': [1,1,0.],
                          'type': 'cup',
                        'graspable': True,
                        'pushable': True,
                        'free': True,
                        'full':False,
                        'size': 0.5
                },
                'drawer': {'position': [2, 0., 0.],
                           'type': 'drawer',
                         'graspable': True,
                         'pushable': True,
                         'free': True,
                         'full': False,
                          'opened':True,
                         'size': 1
                         },
            },
        }
        return scene_state

    @staticmethod
    def test__K():
        print("Test 1: generate observations")
        scene_state = MappingBayesNet_Testing.load__default()
        bn = MappingBayesNet(scene_state)
        print(bn.create_observation())
        bn.init_policy_simple()
        print(bn.select_action())
        return bn


if __name__ == '__main__':
    bn1 = MappingBayesNet_Testing.test__K()
    input("---------")






















#
