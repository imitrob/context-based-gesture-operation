import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt
import pandas

from pymc3_lib import *

''' This is PyMC3 probabilistic framework tutorial
Values are formed using pm.Categorical discrete distribution

'''

'''
Single node - prior
'''
with pm.Model() as m:
    rain = pm.Categorical('rain', C(0.2))

pm.model_to_graphviz(m)

'''
Two nodes: Second node same as the first one
'''
with pm.Model() as m:
    rain = pm.Categorical('rain', C(0.2))
    wet_grass = pm.Deterministic('wet_grass', rain)

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

p_rain = prior_trace['rain'][:].mean()
p_wet_grass = prior_trace['wet_grass'][:].mean()
print(f'P(rain) is {p_rain}')
print(f'P(wet_grass) is {p_wet_grass}')

'''
Two nodes: Second node related to the first one
'''
with pm.Model() as m:
    rain = pm.Categorical('rain', C(0.2))
    wet_grass = pm.Categorical('wet_grass', pm.math.switch(rain, C(0.9), C(0.1)))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

p_rain = prior_trace['rain'][:].mean()
p_wet_grass = prior_trace['wet_grass'][ (prior_trace['rain'] == 1) ].mean()
print(f'P(rain) is {p_rain}')
print(f'P(wet_grass) is {p_wet_grass}')

# def get_prob_by_conditions

get_prob_by_conditions(prior_trace, target='rain', conditions=['!wet_grass'])
get_prob_by_conditions(prior_trace, target='rain', conditions=['wet_grass'])
get_prob_by_conditions(prior_trace, target='rain', conditions=['wet_grass', 'rain'])
get_prob_by_conditions(prior_trace, target='rain', conditions=['wet_grass', '!rain'])

get_prob_by_conditions(prior_trace, target='wet_grass')
get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['rain'])
get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['!rain'])

'''
Three nodes: Binary vars
'''

# class CPT

cpt = CPT({
'rain':  {'sprinkler': C(0.9), '!sprinkler': C(0.3)},
'!rain': {'sprinkler': C(0.7), '!sprinkler': C(0.00001)},
})
cpt[0,0]
cpt.cpt_dict['!rain']['!sprinkler'][1]

cpt.cpt.eval()
cpt.cpt.eval()[0][0]
cpt.cpt.eval()[0][1]
cpt.cpt.eval()[1][0]
cpt.cpt.eval()[1][1]

with pm.Model() as m:
    rain = pm.Categorical('rain', C(0.2))
    sprinkler = pm.Categorical('sprinkler', C(0.6))

    wet_grass = pm.Categorical('wet_grass', cpt.pymc_model_fn(rain, sprinkler))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

get_prob_by_conditions(prior_trace, target='wet_grass')
get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['rain'])
get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['sprinkler'])

get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['rain', '!sprinkler'])
get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['!rain', 'sprinkler'])

get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['!rain', '!sprinkler'])
get_prob_by_conditions(prior_trace, target='wet_grass', conditions=['rain', 'sprinkler'])

'''
More than 4 nodes and more tables:
'''
cpt_wet_grass = CPT({
'rain':  {'sprinkler': C(0.9), '!sprinkler': C(0.3)},
'!rain': {'sprinkler': C(0.7), '!sprinkler': C(0.00001)},
})
cpt_grass_growth = CPT({
'wet_grass': {'fertilizer': C(0.9), '!fertilizer': C(0.7)},
'!wet_grass': {'fertilizer': C(0.4), '!fertilizer': C(0.1)},
})

with pm.Model() as m:
    rain = pm.Categorical('rain', C(0.2))
    sprinkler = pm.Categorical('sprinkler', C(0.6))
    fertilizer = pm.Categorical('fertilizer', C(0.5))

    wet_grass = pm.Categorical('wet_grass', cpt_wet_grass.pymc_model_fn(rain, sprinkler))
    grass_growth = pm.Categorical('grass_growth', cpt_grass_growth.pymc_model_fn(wet_grass, fertilizer))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

get_prob_by_conditions(prior_trace, target='wet_grass')
get_prob_by_conditions(prior_trace, target='grass_growth')
get_prob_by_conditions(prior_trace, target='grass_growth', conditions=['!rain', '!sprinkler', '!fertilizer'])
get_prob_by_conditions(prior_trace, target='grass_growth', conditions=['rain', '!sprinkler', '!fertilizer'])
get_prob_by_conditions(prior_trace, target='grass_growth', conditions=['rain', 'sprinkler', '!fertilizer'])
get_prob_by_conditions(prior_trace, target='grass_growth', conditions=['rain', 'sprinkler', 'fertilizer'])

'''
==================================
One node: More categorical options
==================================
'''
with pm.Model() as m:
    rain = pm.Categorical('rain', [0.7, 0.2, 0.1])

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

# def get_prob_by_conditions_cat

get_prob_by_conditions_cat(prior_trace, target='rain=0')
get_prob_by_conditions_cat(prior_trace, target='rain=1')
get_prob_by_conditions_cat(prior_trace, target='rain=2')

'''
Two nodes: More categorical options - Deterministic
'''
with pm.Model() as m:
    rain = pm.Categorical('rain', [0.7, 0.2, 0.1])
    wet_grass = pm.Deterministic('wet_grass', rain)

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

get_prob_by_conditions_cat(prior_trace, target='wet_grass=2')

'''
Two nodes: 3 - categorical options
'''

mt1 = theano.shared(np.asarray([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]))

# def switch3

with pm.Model() as m:
    rain = pm.Categorical('rain', [0.7, 0.2, 0.1])
    wet_grass = pm.Categorical('wet_grass', switch3(rain, mt=mt1))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

get_prob_by_conditions_cat(prior_trace, target='wet_grass=0')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=2')
get_prob_by_conditions_cat(prior_trace, target='rain=0')
get_prob_by_conditions_cat(prior_trace, target='rain=1')
get_prob_by_conditions_cat(prior_trace, target='rain=2')



'''
Two nodes: 4 - categorical options
'''

mt2 = theano.shared(np.asarray([[1.,0.,0.,0.], [0.,1.,0.,0.], [0.,0.,1.,0.],[0.,0.,0.,1.]]))

# def switch4

with pm.Model() as m:
    rain = pm.Categorical('rain', [0.7, 0.2, 0.07, 0.03])
    wet_grass = pm.Categorical('wet_grass', switch4(rain, mt=mt2))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

get_prob_by_conditions_cat(prior_trace, target='wet_grass=0')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=2')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=3')

get_prob_by_conditions_cat(prior_trace, target='rain=0')
get_prob_by_conditions_cat(prior_trace, target='rain=1')
get_prob_by_conditions_cat(prior_trace, target='rain=2')
get_prob_by_conditions_cat(prior_trace, target='rain=3')

'''
Transition Mapping Table (2D Matrix) class
'''

# class CPTMapping

cptm1 = CPTMapping(
    [[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]],
    input=['no_rain','small_rain','big_rain'], output=['dried_grass','wet_grass','soak_grass']
            )
cptm1()

with pm.Model() as m:
    rain = pm.Categorical('rain', [0.7, 0.2, 0.1])
    wet_grass = pm.Categorical('wet_grass', switch3(rain, mt=cptm1))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=2')
get_prob_by_conditions_cat(prior_trace, target='rain=0')
get_prob_by_conditions_cat(prior_trace, target='rain=1')
get_prob_by_conditions_cat(prior_trace, target='rain=2')

'''
ConditionalProbabilityTable for multivariate n dimensions
'''

# class CPTCat

cptcat_wet_grass = CPTCat({
'rain=0': {'sprinkler=0': C(0.01), 'sprinkler=1': C(0.1), 'sprinkler=2': C(0.6)},
'rain=1': {'sprinkler=0': C(0.05), 'sprinkler=1': C(0.2), 'sprinkler=2': C(0.75)},
'rain=2': {'sprinkler=0': C(0.3), 'sprinkler=1': C(0.6), 'sprinkler=2': C(0.9)},
})
cptcat_wet_grass.n_vars
cptcat_wet_grass.out_n_vars
cptcat_wet_grass.vars

cptcat_wet_grass.cpt.eval()


with pm.Model() as m:
    rain = pm.Categorical('rain', [0.7, 0.2, 0.1])
    sprinkler = pm.Categorical('sprinkler', [0.01, 0.29, 0.7])

    wet_grass = pm.Categorical('wet_grass', cptcat_wet_grass.pymc_model_fn(rain, sprinkler))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)
get_prob_by_conditions_cat(prior_trace, target='rain=0', conditions=['sprinkler=0'])
get_prob_by_conditions_cat(prior_trace, target='rain=0', conditions=['sprinkler=2'])
get_prob_by_conditions_cat(prior_trace, target='rain=2', conditions=['wet_grass=1', 'sprinkler=2'])
get_prob_by_conditions_cat(prior_trace, target='rain=2', conditions=['sprinkler=0'])

get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=0', 'sprinkler=0'])

get_prob_by_conditions_cat(prior_trace, target='rain=1')
get_prob_by_conditions_cat(prior_trace, target='rain=2')

get_prob_by_conditions_cat(prior_trace, target='sprinkler=0')
get_prob_by_conditions_cat(prior_trace, target='sprinkler=1')
get_prob_by_conditions_cat(prior_trace, target='sprinkler=2')

get_prob_by_conditions_cat(prior_trace, target='wet_grass=0')
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1')

get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=0', 'sprinkler=0'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=0', 'sprinkler=1'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=0', 'sprinkler=2'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=1', 'sprinkler=0'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=1', 'sprinkler=1'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=1', 'sprinkler=2'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=2', 'sprinkler=0'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=2', 'sprinkler=1'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=1', conditions=['rain=2', 'sprinkler=2'])

get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=0', 'sprinkler=0'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=0', 'sprinkler=1'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=0', 'sprinkler=2'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=1', 'sprinkler=0'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=1', 'sprinkler=1'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=1', 'sprinkler=2'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=2', 'sprinkler=0'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=2', 'sprinkler=1'])
get_prob_by_conditions_cat(prior_trace, target='wet_grass=0', conditions=['rain=2', 'sprinkler=2'])

'''
Two nodes: Normal distribution to Categorical one
- two Normal distr.
'''
>>> a = theano.shared(np.asarray([3.,2.,1.]))
>>> b = tt.argmin(a)
>>> b.eval()
>>> c = tt.argmin([5.,6.,7.])
>>> c.eval()

with pm.Model() as m:
    focus_object_0 = pm.Normal('focus_object_0', mu=0.0, sigma=1)
    focus_object_1 = pm.Normal('focus_object_1', mu=1.0, sigma=1)

    focus_object_id = pm.Deterministic('focus_object_id', tt.argmin([focus_object_0, focus_object_1]))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

get_prob_by_conditions_cat(prior_trace, target='focus_object_id=0')
get_prob_by_conditions_cat(prior_trace, target='focus_object_id=1')

'''
- Vector Normal distr.
'''
with pm.Model() as m:
    focus_object = pm.Normal('focus_object', mu=[0.0, 1.0], sigma=1, shape=(2))

    focus_object_id = pm.Deterministic('focus_object_id', tt.argmin(focus_object))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)

get_prob_by_conditions_cat(prior_trace, target='focus_object_id=0')
get_prob_by_conditions_cat(prior_trace, target='focus_object_id=1')

'''
Float input to the model
'''
test_input = theano.shared(np.asarray([0.0,1.0]))
m = pm.Model()
with m:
    focus_object_0 = pm.Normal('focus_object_0', mu=test_input[0], sigma=1)
    focus_object_1 = pm.Normal('focus_object_1', mu=test_input[1], sigma=1)

    focus_object_id = pm.Deterministic('focus_object_id', tt.argmin([focus_object_0, focus_object_1]))

    prior_trace = pm.sample_prior_predictive(100000)

pm.model_to_graphviz(m)


get_prob_by_conditions_cat(prior_trace, target='focus_object_id=0')
get_prob_by_conditions_cat(prior_trace, target='focus_object_id=1')

test_input = theano.shared(np.asarray([1.0,0.0]))
m = pm.Model()
with m:
    focus_object_0 = pm.Normal('focus_object_0', mu=test_input[0], sigma=1)
    focus_object_1 = pm.Normal('focus_object_1', mu=test_input[1], sigma=1)

    focus_object_id = pm.Deterministic('focus_object_id', tt.argmin([focus_object_0, focus_object_1]))

    #prior_trace = pm.sample_prior_predictive(100000)
    trace = pm.sample(10000, initvals={'focus_object_0': 1.0})

pm.traceplot(trace)

trace._straces[0]['samples'][0]

get_prob_by_conditions_cat(prior_trace, target='focus_object_id=0')
get_prob_by_conditions_cat(prior_trace, target='focus_object_id=1')





#
