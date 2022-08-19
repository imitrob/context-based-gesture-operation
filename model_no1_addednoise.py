import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt
floatX = theano.config.floatX

from pymc3_lib import *

class MappingBayesNet():
    def __init__(self, scene_observation):
        self.G = scene_observation['G']
        self.A = scene_observation['A']
        self.O = list(scene_observation['objects'].keys())

        self.n_objects = len(scene_observation['objects'])
        self.n_gestures = len(self.G)
        self.n_actions = len(self.A)
        self.object_positions__in = theano.shared(np.asarray([scene_observation['objects'][k]['position'] for k in scene_observation['objects'].keys()]))

        self.gestures__in = theano.shared(np.asarray(scene_observation['gestures']))
        self.gripper_full__in = theano.shared(np.asarray(scene_observation['gripper']['full']))
        self.eef_position__in = theano.shared(np.asarray(scene_observation['eef_position']))

        self.valid_action_object = np.ones([self.n_actions, self.n_objects])
        for n,objk in enumerate(list(scene_observation['objects'].keys())):
            tmp = [1] * self.n_actions
            # action 1
            if scene_observation['objects'][objk]['graspable']:
                tmp[0] = 1
            else:
                tmp[0] = 0
            # action 3
            if scene_observation['objects'][objk]['pushable']:
                tmp[2] = 1
            else:
                tmp[2] = 0
            self.valid_action_object[:,n] = tmp

        self.valid_action_object = theano.shared(self.valid_action_object)

        self.cptm1 = CPTMapping(
            [ # a1,a2,a3,a4
                [1, 0, 0, 0], # g1
                [0, 1, 0, 0], # g2
                [0, 0, 1, 0], # g3
            ],
            input=self.G, output=self.A
        )
    def model(self):
        with pm.Model() as self.m:
            # shape = (n_gestures) [probability], P(g|o)
            gestures = pm.Normal('gestures', mu=self.gestures__in, sigma=0.01, shape=(self.n_gestures))
            # shape = (1) [boolean 'probability'], P(gr|o)
            gripper_full = pm.Normal('gripper_full', mu=self.gripper_full__in, sigma=0.01, shape=(1))

            # shape = (3) [meters]
            eef_position = pm.Normal('eef_position', mu=self.eef_position__in, sigma=0.01, shape = (3))
            # shape = (n_objects, 3) [meters]
            object_positions = pm.Normal('object_positions', mu=self.object_positions__in, sigma=0.01, shape = (self.n_objects, 3))

            # shape = (n_actions) [probability], P(a|g)
            # mapping table application only, unconditioned
            actions = pm.Normal('actions', mu=pm.math.dot(self.cptm1.cpt.T, gestures), sigma=0.01, shape = (self.n_actions))

            # Distances eef to objects [meters], L2_norm(object_positions-eef_position)
            eefobj_distances_ = tt.add(object_positions, tt.dot(-1, eef_position))
            eefobj_distances = tt.sum(tt.pow(eefobj_distances_,2), axis=1)
            # shape = (n_objects) [likelihood], sigmoid(inv(eefobj_distances))
            eeff = pm.Normal('eeff', mu=pm.math.sigmoid(1 / eefobj_distances), sigma=0.01, shape = (self.n_objects))

            # shape = (n_actions, n_objects) [probability], applies action validation
            actions_1 = pm.Normal('actions_1', mu=(self.valid_action_object.T * actions), sigma=0.01, shape = (self.n_objects, self.n_actions))
            # shape = (n_actions, n_objects) [probability], merge with eeff
            actions_2 = pm.Normal('actions_2', mu=(actions_1.T * eeff), sigma=0.01, shape = (self.n_actions, self.n_objects))

            # shape = (1) [id action]
            target_action = pm.Normal('target_action', mu=(tt.argmax(actions_2) // self.n_objects), sigma=0.01)
            # shape = (1) [id object]
            target_object = pm.Normal('target_object', mu=(tt.argmax(actions_2) % self.n_objects), sigma=0.01)

            test = pm.Normal('test', mu=0, sigma=1)
            self.prior_trace = pm.sample_prior_predictive(10000)
            self.trace = pm.sample(10000)
            # adds info to trace
            self.prior_trace['target_action_n'] = self.n_actions
            self.prior_trace['target_object_n'] = self.n_objects

    def predict(self):
        action_probabilities = []
        for i in range(self.n_actions):
            target_action = get_prob_by_conditions_cat(self.prior_trace, target=f'target_action={i}')
            action_probabilities.append(target_action)
        print(f"Action intent: {self.A[np.argmax(action_probabilities)]}")

        object_probabilities = []
        for i in range(self.n_objects):
            target_object = get_prob_by_conditions_cat(bn.prior_trace, target=f'target_object={i}')
            object_probabilities.append(target_object)
        print(f"Object intent: {self.O[np.argmax(object_probabilities)]}")

scene_observation = {
    'A': ['grab','release','push', 'no_action'],
    'G': ['grab','release','push'],
    'gripper': {
        'full': True,
    },
    'objects': {
        'cup1': { 'position': [0.,0.,0.],
                'graspable': True,
                'pushable': True,
                'free': True,
        },
        'cup2': { 'position': [0.,0.,0.],
                'graspable': True,
                'pushable': True,
                'free': True,
        },
    },
    'gestures': [0.9, 0.1, 0.05],
    'eef_position': [0.,0.,0.5],
}
print("Example 1: Two same objects cup1, cup2. cup2 is closer")
scene_observation['objects']['cup1']['position'] = [1.,0.,0.]
scene_observation['objects']['cup2']['position'] = [0.,0.,0.]
bn = MappingBayesNet(scene_observation)
bn.model()
bn.predict()


pm.model_to_graphviz(bn.m)
input("---------")
print("Example 2: Two same objects cup1, cup2. cup1 is closer")

scene_observation['objects']['cup1']['position'] = [0.,0.,0.]
scene_observation['objects']['cup2']['position'] = [1.,0.,0.]
bn = MappingBayesNet(scene_observation)
bn.model()
bn.predict()

input("---------")
print("Example 3: Two different objects cup, drawer. Grab is most probable. Closer to drawer\nInfeasible to grab drawer -> intent to grab cup")

scene_observation['objects'] = {
    'cup': { 'position': [0.,0.,0.],
            'graspable': True,
            'pushable': False,
            'free': True,
    },
    'drawer': { 'position': [0.,0.,0.],
            'graspable': False,
            'pushable': True,
            'free': True,
    },
}
scene_observation['objects']['cup']['position'] = [1.,0.,0.]
scene_observation['objects']['drawer']['position'] = [0.,0.,0.]
scene_observation['gestures'] = [0.9, 0.1, 0.05]
bn = MappingBayesNet(scene_observation)
bn.model()
bn.predict()
input("----------")
print("Example 4: Two different objects cup, drawer. Release most probable. Closer to drawer\nInfeasible to grab drawer -> intent to grab cup")

scene_observation['objects']['cup']['position'] = [1.,0.,0.]
scene_observation['objects']['drawer']['position'] = [0.,0.,0.]
scene_observation['gestures'] = [0.1, 0.9, 0.05]
bn = MappingBayesNet(scene_observation)
bn.model()
bn.predict()


graphvizfig = pm.model_to_graphviz(model)
fig = pm.traceplot(bn.trace)
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
az.plot_forest(trace, r_hat=True);
fig, axarr = plt.subplots(1,1)
trarr = az.plot_trace(bn.trace)
fig = plt.gcf() # to get the current figure...

pm.plot_forest(bn.trace)

#
