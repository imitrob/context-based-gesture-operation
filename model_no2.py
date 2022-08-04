import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt
import pandas

from pymc3_lib import *

import matplotlib.pyplot as plt

class MappingBayesNet():
    ''' v2
    >>> MappingBayesNet(objects)
    '''
    def __init__(self, scene_observation):
        self.objects = scene_observation['objects'] # Object data
        self.G = scene_observation['G'] # Gesture names
        self.A = scene_observation['A'] # Action names
        self.O = list(self.objects.keys()) # Object names

        self.n_objects = len(self.objects)
        self.n_gestures = len(self.G)
        self.n_actions = len(self.A)
        self.object_positions__in = theano.shared(np.asarray([self.objects[k]['position'] for k in self.objects.keys()]))

        # Observations
        self.gestures__in = theano.shared(np.asarray(scene_observation['gestures']))
        self.gripper_full__in = theano.shared(np.asarray(int(scene_observation['gripper']['full'])))
        self.eef_position__in = theano.shared(np.asarray(scene_observation['eef_position']))
        # Scene & robot parameters
        self.gripper_range = theano.shared(np.asarray(scene_observation['gripper']['range']))
        object_sizes_ = []
        for n,objk in enumerate(list(self.objects.keys())):
            object_sizes_.append(self.objects[objk]['size'])
        self.object_sizes = theano.shared(np.asarray(object_sizes_))

        # fills in action dependent features
        self.action_dependent_features = theano.shared(np.ones([self.n_actions, self.n_objects]))
        t0 = []
        for n,a in enumerate(self.A):
            row = theano.shared(np.ones([self.n_objects]))
            for m,action_feature in enumerate(scene_observation['misc_features'][a]):
                row = row * getattr(self, action_feature)()
            t0.append([row])
        self.action_dependent_features = tt.concatenate(t0)

        # Conditional probability table of different features
        self.cpt_misc_features = CPTCat({
        'intent=0': {'object_intent=0': C(self.action_dependent_features[0][0].eval()), 'object_intent=1': C(self.action_dependent_features[0][0].eval())},
        'intent=1': {'object_intent=0': C(self.action_dependent_features[0][1].eval()), 'object_intent=1': C(self.action_dependent_features[0][1].eval())},
        })

        self.cpt_gesture_object_focus = CPTCat({
        'intent=0': {'object_intent=0': C(0.25), 'object_intent=1': C(0.25)},
        'intent=1': {'object_intent=0': C(0.25), 'object_intent=1': C(0.25)},
        })
        self.mt_gestures = theano.shared(np.diag(scene_observation['gestures']))
        self.mt_parameter_distance = theano.shared(np.diag([1/3]*3))

    def model(self):
        with pm.Model() as self.m:
            intent = pm.Categorical('intent', [1/self.n_actions]*self.n_actions) # push, grab
            object_intent = pm.Categorical('object_intent', [1/self.n_objects]*self.n_objects) # obj1, obj2

            # can be observed
            gesture_id_intent = pm.Categorical('gesture_id_intent', switchn(intent, self.mt_gestures))
            misc_features = pm.Categorical('misc_features', self.cpt_misc_features.fn(object_intent, intent)) # fits, not fits
            parameter_distance = pm.Categorical('parameter_distance', switchn(intent, self.mt_parameter_distance)) # 5, 10, 20
            gesture_object_focus = pm.Categorical('gesture_object_focus', self.cpt_gesture_object_focus.fn(intent, object_intent))

            #parameter_distance_gestures = pm.Categorical('parameter_distance_gestures', pm.math.switch(parameter_distance, C(0.9), C(0.1)))
            eeff_feature = self.eeff__feature()
            eeff = pm.Categorical('eeff', pm.math.switch(object_intent, C(eeff_feature[0]), C(eeff_feature[1])))

            self.prior_trace = pm.sample_prior_predictive(10000)

    def eeff__feature(self):
        ''' Deterministically compute eef field based on observation
        '''
        eefobj_distances_ = tt.add(self.object_positions__in, tt.dot(-1, self.eef_position__in))
        eefobj_distances = tt.sum(tt.pow(eefobj_distances_,2), axis=1)

        eeff = self.gaussian(eefobj_distances)
        return eeff

    def feaf__feature(self):
        ''' Deterministically compute action dependent feature
        Returns: shape = (n_objects)
        '''
        feaf = self.sigmoid(self.object_sizes, center=self.gripper_range)
        return feaf

    def predict(self, out=''):
        ''' Returns action intent and object intent
        >>> self = bn
        '''
        ai_0 = get_prob_by_conditions_cat(self.prior_trace, target='intent=0', conditions=['gesture_id_intent=0'])
        ai_1 = get_prob_by_conditions_cat(self.prior_trace, target='intent=1', conditions=['gesture_id_intent=0'])

        ai_0 = get_prob_by_conditions_cat(self.prior_trace, target='intent=0')
        ai_1 = get_prob_by_conditions_cat(self.prior_trace, target='intent=1')
        intent = self.A[np.argmax([ai_0, ai_1])]

        so_0 = get_prob_by_conditions_cat(self.prior_trace, target='object_intent=0', conditions=['eeff=0'])
        so_1 = get_prob_by_conditions_cat(self.prior_trace, target='object_intent=1', conditions=['eeff=0'])

        so_0 = get_prob_by_conditions_cat(self.prior_trace, target='object_intent=0')
        so_1 = get_prob_by_conditions_cat(self.prior_trace, target='object_intent=1')
        selected_object = self.O[np.argmax([so_0, so_1])]

        if out=='data': return [[ai_0, ai_1], [so_0, so_1]]
        return intent, selected_object

    def gaussian(self, x, sigma=0.2):
        return pm.math.exp(-tt.power(x, 2.) / (2 * tt.power(sigma, 2.)))

    def sigmoid(self, x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + tt.exp((center-x)*(-tau)))

    @property
    def intent(self):
        print(self.m.intent)
        print(f"P: {[self.n_actions]*self.n_actions}")
    @property
    def object_intent(self):
        print(self.m.object_intent)
        print(f"P: {[1/self.n_objects]*self.n_objects}")
    @property
    def gesture_id_intent(self):
        print(self.m.gesture_id_intent)
        print(f"CPT: {self.mt_gestures.eval()}")
    @property
    def misc_features(self):
        print(self.m.misc_features)
        print(f"CPT: {self.misc_features.cpt.eval()}")
    @property
    def parameter_distance(self):
        print(self.m.parameter_distance)
        print(f"P switch: {[[0.3,0.3,0.3], [0.3,0.5,0.2]]}")
    @property
    def gesture_object_focus(self):
        print(self.m.gesture_object_focus)
        print(f"CPT: {self.cpt_gesture_object_focus.cpt.eval()}")
    '''
    @property
    def parameter_distance_gestures(self):
        print(self.m.parameter_distance_gestures)
        print(f"P switch: C(0.9), C(0.1))")
    '''
    @property
    def eeff(self):
        print(self.m.eeff)
        print(f"P switch: {self.eeff__feature().eval()}")


class MappingBayesNet_Testing():
    def __init__(self):
        '''
        >>> MappingBayesNet_Testing.test__eeff()
        >>> MappingBayesNet_Testing.test__gaussian()
        >>> MappingBayesNet_Testing.test__feaf()
        >>> MappingBayesNet_Testing.test__sigmoid()
        >>> MappingBayesNet_Testing.test__cup1_x_coordinates()
        >>> MappingBayesNet_Testing.test__cup1_size()
        '''
    @staticmethod
    def load__default():
        scene_observation = {
            'A': ['grab','release'],
            'G': ['grab','release'],
            'gripper': {'range': 0.14, # [m]
                        'full': True,
            },
            'objects': {
                'cup1': { 'position': [-0.5,0.,0.],
                        'graspable': True,
                        'pushable': True,
                        'free': True,
                        'size': 0.08 # [m]
                },
                'cup2': { 'position': [0.5,0.,0.],
                        'graspable': True,
                        'pushable': True,
                        'free': True,
                        'size': 0.06
                },
            },
            'eef_position': [0.,0.,0.5],
            'gestures': [0.9, 0.1],
            'misc_features': {
                'grab': ['eeff__feature'],
                'release': [],
            }
        }
        return scene_observation

    @staticmethod
    def test__gaussian():
        bn = MappingBayesNet(MappingBayesNet_Testing.load__default())
        x = np.linspace(0, 1, 100)
        y = bn.gaussian(x).eval()
        print('Xlabel: distance [meters] Ylabel: probability [-]')
        plt.plot(x,y)
        plt.show()

    @staticmethod
    def test__sigmoid():
        bn = MappingBayesNet(MappingBayesNet_Testing.load__default())
        x = np.linspace(0, 0.3, 100)
        y = bn.sigmoid(x).eval()
        print('Xlabel: distance [meters] Ylabel: probability [-]')
        plt.plot(x,y)
        plt.show()

    @staticmethod
    def test__feaf():
        scene_observation = MappingBayesNet_Testing.load__default()
        object1 = list(scene_observation['objects'].keys())[0]
        bn = MappingBayesNet(scene_observation)
        print(f"{bn.n_objects} objs")

        sizes_test = np.linspace(0, 0.3, 20)
        feaf_plot = []
        for i in sizes_test:
            scene_observation['objects'][object1]['size'] = i

            sizes = [ scene_observation['objects'][obj]['size'] for obj in scene_observation['objects'].keys()]
            bn = MappingBayesNet(scene_observation)
            feaf = bn.feaf__feature().eval()
            feaf_plot.append(feaf)
            print(f"feasib-obj size: {sizes}, feaf: {feaf}")

        plt.plot(sizes_test, feaf_plot)
        plt.show()

    @staticmethod
    def test__eeff():
        scene_observation = MappingBayesNet_Testing.load__default()
        bn = MappingBayesNet(scene_observation)
        print(f"{bn.n_objects} objs")
        for i in range(-10, 0):
            scene_observation['objects']['cup1']['position'][0] = i/10
            scene_observation['objects']['cup2']['position'][0] = 0.5
            bn = MappingBayesNet(scene_observation)
            eefobjdists = np.linalg.norm(np.array(bn.object_positions__in.eval()) - np.array(bn.eef_position__in.eval()), axis=1)
            eeff = bn.eeff__feature().eval()
            print(f"eef-obj dist: {eefobjdists}, gaussian values: {eeff}, argmax {np.argmax(eeff)}")

    @staticmethod
    def test__1():
        print("Test 1: cup1 object is closer")
        scene_observation = MappingBayesNet_Testing.load__default()
        scene_observation['objects']['cup1']['position'][0] = 0.0
        bn = MappingBayesNet(scene_observation)
        bn.model()
        print(bn.predict())
        return bn
    @staticmethod
    def test__2():
        print("Test 2: cup2 object is closer")
        scene_observation = MappingBayesNet_Testing.load__default()
        scene_observation['objects']['cup2']['position'][0] = 0.0
        bn = MappingBayesNet(scene_observation)
        bn.model()
        print(bn.predict())
        return bn
    @staticmethod
    def test__3():
        print("Test 3: cup2 object does not fits")
        scene_observation = MappingBayesNet_Testing.load__default()
        scene_observation['objects']['cup2']['size'] = 0.2
        bn = MappingBayesNet(scene_observation)
        bn.model()
        print(bn.predict())
        return bn
    @staticmethod
    def test__4():
        print("Test 4: cup1 object does not fits")
        scene_observation = MappingBayesNet_Testing.load__default()
        scene_observation['objects']['cup1']['size'] = 0.2
        bn = MappingBayesNet(scene_observation)
        bn.model()
        print(bn.predict())
        return bn
    @staticmethod
    def test__cup1_x_coordinates():
        o1_probs = []
        p_x =np.linspace(-1,0,11)
        for i in p_x:
            scene_observation = MappingBayesNet_Testing.load__default()
            scene_observation['objects']['cup1']['position'][0] = i
            bn = MappingBayesNet(scene_observation)
            bn.model()
            d = bn.predict(out='data')
            o1_probs.append(d[1][0])

        plt.xlabel("cup1 x position [m]")
        plt.ylabel("cup1 object select probability [-]")
        plt.plot(p_x, o1_probs)
        plt.show()
    @staticmethod
    def test__cup1_size():
        o1_probs = []
        p_s =np.linspace(0.04,0.6,11)
        for i in p_s:
            scene_observation = MappingBayesNet_Testing.load__default()
            scene_observation['objects']['cup1']['size'] = i
            bn = MappingBayesNet(scene_observation)
            bn.model()
            d = bn.predict(out='data')
            o1_probs.append(d[1][0])

        plt.xlabel("cup1 size [m]")
        plt.ylabel("cup1 object select probability [-]")
        plt.plot(p_s, o1_probs)
        plt.show()


if __name__ == '__main__':
    MappingBayesNet_Testing.test__eeff()
    input("---------")
    MappingBayesNet_Testing.test__gaussian()
    input("---------")
    MappingBayesNet_Testing.test__feaf()
    input("---------")
    MappingBayesNet_Testing.test__sigmoid()
    input("---------")
    bn1 = MappingBayesNet_Testing.test__1()
    input("---------")
    bn2 = MappingBayesNet_Testing.test__2()
    input("---------")
    bn3 = MappingBayesNet_Testing.test__3()
    input("---------")
    bn4 = MappingBayesNet_Testing.test__4()
    input("---------")
    bn5 = MappingBayesNet_Testing.test__cup1_x_coordinates()
    input("---------")
    bn6 = MappingBayesNet_Testing.test__cup1_size()






















#
