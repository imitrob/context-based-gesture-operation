import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt
import pandas

from pymc3_lib import *

class MappingBayesNet():
    ''' v2.1
    >>> MappingBayesNet(objects)
    '''
    def __init__(self, scene_observation, probabilities=[]):
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
        gripper_full = 1 if scene_observation['robot']['attached'] is not None else 0
        self.gripper_full__in = theano.shared(np.asarray(gripper_full))
        self.eef_position__in = theano.shared(np.asarray(scene_observation['robot']['eef_position']))
        # Scene & robot parameters
        self.gripper_range = theano.shared(np.asarray(scene_observation['robot']['gripper_range']))
        object_sizes_ = []
        for n,objk in enumerate(list(self.objects.keys())):
            object_sizes_.append(self.objects[objk]['size'])
        self.object_sizes = theano.shared(np.asarray(object_sizes_))

        self.mt_gestures = theano.shared(np.diag(scene_observation['gestures']))
        self.mt_parameter_distance = probabilities['mt_parameter_distance']

        self.intent_p = [1/self.n_actions]*self.n_actions
        self.object_intent_p = [1/self.n_objects]*self.n_objects
        self.cpt_gesture_object_focus = probabilities['cpt_gesture_object_focus']

        self.cpt_eeff_feature = self.eeff__feature_cpt()
        self.cpt_feaf_feature = self.feaf__feature_cpt()

    def model(self):
        with pm.Model() as self.m:
            intent = pm.Categorical('intent', self.intent_p) # push, grab
            object_intent = pm.Categorical('object_intent', self.object_intent_p) # obj1, obj2

            # can be observed
            gesture_id_intent = pm.Categorical('gesture_id_intent', switchn(intent, self.mt_gestures))
            parameter_distance = pm.Categorical('parameter_distance', self.mt_parameter_distance.fn(intent)) # 5, 10, 20
            gesture_object_focus = pm.Categorical('gesture_object_focus', self.cpt_gesture_object_focus.fn(intent, object_intent))

            eeff = pm.Categorical('eeff', self.cpt_eeff_feature.fn(intent, object_intent))
            feaf = pm.Categorical('feaf', self.cpt_feaf_feature.fn(intent, object_intent))

            self.prior_trace = pm.sample_prior_predictive(10000)

    def sample(self, n=100):
        with self.m:
            prior_trace = pm.sample_prior_predictive(n)
        return prior_trace

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

    ''' Conversions to CPT '''
    def eeff__feature_cpt(self):
        eeff = self.eeff__feature()
        cpt_dict = {}
        for n,eeff_ in enumerate(eeff.eval()):
            cpt_dict['object='+str(n)] = C(eeff_)
        return CPTCat(cpt_dict)

    def feaf__feature_cpt(self):
        feaf = self.feaf__feature()
        cpt_dict = {}
        for n,feaf_ in enumerate(feaf.eval()):
            cpt_dict['object='+str(n)] = C(feaf_)
        return CPTCat(cpt_dict)

    def predict(self, out=''):
        ''' Returns action intent and object intent
        >>> self = bn
        '''
        #ai_0 = get_prob_by_conditions_cat(self.prior_trace, target='intent=0', conditions=['gesture_id_intent=0'])
        #ai_1 = get_prob_by_conditions_cat(self.prior_trace, target='intent=1', conditions=['gesture_id_intent=0'])

        ai_0 = get_prob_by_conditions_cat(self.prior_trace, target='intent=0')
        ai_1 = get_prob_by_conditions_cat(self.prior_trace, target='intent=1')
        intent = self.A[np.argmax([ai_0, ai_1])]

        #so_0 = get_prob_by_conditions_cat(self.prior_trace, target='object_intent=0', conditions=['eeff=0'])
        #so_1 = get_prob_by_conditions_cat(self.prior_trace, target='object_intent=1', conditions=['eeff=0'])

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
        print(f"P: {self.intent_p}")
    @property
    def object_intent(self):
        print(self.m.object_intent)
        print(f"P: {self.object_intent_p}")
    @property
    def gesture_id_intent(self):
        print(self.m.gesture_id_intent)
        print(f"CPT: {self.mt_gestures.eval()}")
    @property
    def parameter_distance(self):
        print(self.m.parameter_distance)
        print(f"P switch: {self.mt_gestures.eval()}")
    @property
    def gesture_object_focus(self):
        print(self.m.gesture_object_focus)
        print(f"CPT: {self.cpt_gesture_object_focus.cpt.eval()}")
    @property
    def eeff(self):
        print(self.m.eeff)
        print(f"P switch: {self.eeff__feature().eval()}")

    @property
    def info(self):
        pm.model_to_graphviz(self.m)
        self.intent
        self.object_intent
        self.gesture_id_intent
        self.parameter_distance
        self.gesture_object_focus
        self.eeff


if __name__ == '__main__':

    a = probabilities['cpt_gesture_object_focus']

    # %% codecell

    cpt = CPTCat({
    'intent=0': {'object_intent=1': C(0.25)},
    'intent=1': {'object_intent=0': C(0.25), 'object_intent=1': C(0.25)},
    }, vars=['intent','object_intent'], n_vars=[2,2], out_n_vars=2)
