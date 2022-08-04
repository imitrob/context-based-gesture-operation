import pymc3 as pm
import numpy as np
import pprint

import theano
import theano.tensor as tt
floatX = theano.config.floatX

from pymc3_lib import *
from matplotlib_lib.circle_gradient import plot_circle_gradient

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
        self.gripper_full__in = theano.shared(np.asarray(int(scene_observation['gripper']['full'])))
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

        self.gripper_range = theano.shared(np.asarray(scene_observation['gripper']['range']))

        object_sizes_ = []
        for n,objk in enumerate(list(scene_observation['objects'].keys())):
            object_sizes_.append(scene_observation['objects'][objk]['size'])
        self.object_sizes = theano.shared(np.asarray(object_sizes_))

    def model(self):
        with pm.Model() as self.m:
            # shape = (n_gestures) [probability], P(g|o)
            gestures = pm.Deterministic('gestures', self.gestures__in)
            # shape = (1) [boolean 'probability'], P(gr|o)
            gripper_full = pm.Deterministic('gripper_full', self.gripper_full__in)

            # shape = (3) [meters]
            eef_position = pm.Deterministic('eef_position', self.eef_position__in)
            # shape = (n_objects, 3) [meters]
            object_positions = pm.Deterministic('object_positions', self.object_positions__in)

            # shape = (n_actions) [probability], P(a|g)
            # mapping table application only, unconditioned
            actions = pm.Deterministic('actions', pm.math.dot(self.cptm1.cpt.T, gestures))

            # Distances eef to objects [meters], L2_norm(object_positions-eef_position)
            eefobj_distances_ = tt.add(object_positions, tt.dot(-1, eef_position))
            eefobj_distances = tt.sum(tt.pow(eefobj_distances_,2), axis=1)
            # shape = (n_objects) [likelihood], gaussian(inv(eefobj_distances))
            eeff = pm.Deterministic('eeff', self.gaussian(eefobj_distances))


            # shape = (n_actions * n_objects) [probability], applies action validation
            actions_1 = pm.Deterministic('actions_1', self.valid_action_object.T * actions)
            # shape = (n_actions * n_objects) [probability], merge with eeff
            actions_2 = pm.Deterministic('actions_2', actions_1.T * eeff)

            # shape = (n_objects) [likelihood], sigmoid( object_sizes - gripper_range )
            graspable_by_size = pm.Deterministic('graspable_by_size', pm.math.sigmoid(tt.add(pm.math.dot(-1,self.object_sizes), self.gripper_range)) )

            self.mask_placeholder = theano.shared(np.ones([self.n_actions-1, self.n_objects]))
            self.intent_misc = pm.Deterministic('intent_misc', pm.math.concatenate([[graspable_by_size], self.mask_placeholder]))
            actions_3 = pm.Deterministic('actions_3', actions_2 * self.intent_misc)

            # shape = (1) [id action]
            target_action = pm.Deterministic('target_action', tt.argmax(actions_3) // self.n_objects)
            # shape = (1) [id object]
            target_object = pm.Deterministic('target_object', tt.argmax(actions_3) % self.n_objects)

            test = pm.Normal('test', mu=0, sigma=1)
            self.prior_trace = pm.sample_prior_predictive(100)
            #self.trace = pm.sample(10000)
            # adds info to trace
            self.prior_trace['target_action_n'] = self.n_actions
            self.prior_trace['target_object_n'] = self.n_objects

    def gaussian(self, x, sig=10):
        return pm.math.exp(-tt.power(x, 2.) / (2 * tt.power(sig, 2.)))

    def predict(self):
        action_probabilities = []
        for i in range(self.n_actions):
            target_action = get_prob_by_conditions_cat(self.prior_trace, target=f'target_action={i}')
            action_probabilities.append(target_action)
        #print(f"Action intent: {self.A[np.argmax(action_probabilities)]}")

        object_probabilities = []
        for i in range(self.n_objects):
            target_object = get_prob_by_conditions_cat(bn.prior_trace, target=f'target_object={i}')
            object_probabilities.append(target_object)
        #print(f"Object intent: {self.O[np.argmax(object_probabilities)]}")
        return self.A[np.argmax(action_probabilities)], self.O[np.argmax(object_probabilities)]

if __name__ == '__main__':
    scene_observation = {
        'A': ['grab','release','push', 'no_action'],
        'G': ['grab','release','push'],
        'gripper': {'range': 0.14, # [m]
                    'full': True,
        },
        'objects': {
            'cup1': { 'position': [0.,0.,0.],
                    'graspable': True,
                    'pushable': True,
                    'free': True,
                    'size': 0.08 # [m]
            },
            'cup2': { 'position': [0.,0.,0.],
                    'graspable': True,
                    'pushable': True,
                    'free': True,
                    'size': 0.06
            },
        },
        'gestures': [0.9, 0.1, 0.05],
        'eef_position': [0.,0.,0.5],
    }
    print("Example 1: Two same objects cup1, cup2. cup2 is closer")
    scene_observation['objects']['cup1']['position'] = [1.,0.,0.]
    scene_observation['objects']['cup2']['position'] = [0.,0.,0.]

    pprint.pprint(scene_observation)
    bn = MappingBayesNet(scene_observation)
    bn.model()
    print(f"Predict (intent, object_intent): {bn.predict()}")

    input("Press enter")

    gv = pm.model_to_graphviz(bn.m)
    gv.render('gv', format='svg')
    help(gv.render)

    print("Example 2: Two same objects cup1, cup2. cup1 is closer")

    scene_observation['objects']['cup1']['position'] = [0.,0.,0.]
    scene_observation['objects']['cup2']['position'] = [1.,0.,0.]
    pprint.pprint(scene_observation)
    bn = MappingBayesNet(scene_observation)
    bn.model()
    print(f"Predict (intent, object_intent): {bn.predict()}")

    bn.prior_trace['eeff']
    bn.prior_trace['intent_misc']
    bn.prior_trace['actions_1']
    bn.prior_trace['actions_2']
    bn.prior_trace['actions_3']

    input("Press enter")
    print("Example 3: Two different objects cup, drawer. Grab is most probable. Closer to drawer\nInfeasible to grab drawer -> intent to grab cup")

    scene_observation['objects'] = {
        'cup': { 'position': [0.,0.,0.],
                'graspable': True,
                'pushable': False,
                'free': True,
                'size': 0.1, # [m]
        },
        'drawer': { 'position': [0.,0.,0.],
                'graspable': False,
                'pushable': True,
                'free': True,
                'size': 0.3, # [m]
        },
    }
    scene_observation['objects']['cup']['position'] = [1.,0.,0.]
    scene_observation['objects']['drawer']['position'] = [0.,0.,0.]
    pprint.pprint(scene_observation)
    bn = MappingBayesNet(scene_observation)
    bn.model()
    print(f"Predict (intent, object_intent): {bn.predict()}")

    input("Press enter")
    print("Example 4: Two different objects cup, drawer. Release most probable. Closer to drawer\nInfeasible to grab drawer -> intent to grab cup")
    scene_observation['objects']['cup']['position'] = [1.,0.,0.]
    scene_observation['objects']['drawer']['position'] = [0.,0.,0.]
    scene_observation['gestures'] = [0.1, 0.9, 0.05]
    pprint.pprint(scene_observation)
    bn = MappingBayesNet(scene_observation)
    bn.model()
    print(f"Predict (intent, object_intent): {bn.predict()}")

    input("Press enter")
    print("Plot scenario")

    plot_circle_gradient(inner_radius = 0.1, outer_radius = 0.3, center = [0., 0.5], color = 'gold')
    a = []
    o = []
    for i in range(0,10):
        scene_observation['objects']['cup']['position'] = [i*0.1,0.,0.]
        scene_observation['objects']['drawer']['position'] = [-0.5,0.,0.]
        scene_observation['gestures'] = [0.1, 0.9, 0.05]
        bn = MappingBayesNet(scene_observation)
        bn.model()
        a_tmp, o_tmp = bn.predict()

        a.append(a_tmp)
        o.append(o_tmp)
        print(o_tmp)
        if o_tmp == 'cup':
            plt.scatter([i*0.1],[0.], color='red', zorder=4)
        elif o_tmp == 'drawer':
            plt.scatter([i*0.1],[0.], color='blue', zorder=4)

    plt.scatter([-0.5],[0.], color='black', zorder=4)
    plt.xlabel("X coordinates")
    plt.ylabel("Z coordinates")
    plt.show()


    input("Press enter")
    print("Plot traceplot")
    scene_observation['objects']['cup']['position'] = [0.,0.,0.]
    scene_observation['objects']['drawer']['position'] = [1.,0.,0.]
    scene_observation['gestures'] = [0.5, 0.5, 0.51]
    bn = MappingBayesNet(scene_observation)
    bn.model()
    print(f"Predict (intent, object_intent): {bn.predict()}")
    pl = pm.traceplot(bn.prior_trace)
    pl.render("sad")
    pm.plot_forest(bn.prior_trace)

#
