'''
Risk Assessment and Decision Analysis with Bayesian networks
'''

import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt

names = ['martin_oversleeps', 'train_strike', 'martin_late', 'norman_late']

# P(martin_late|train_strike, martin_oversleeps) -------------------------

martin_late_lookup = theano.shared(np.asarray([
    [[.7, .3], [.4, .6]],
    [[.4, .6], [.2, .8]]]))

def p_martin_late(train_strike, martin_oversleeps):
    return martin_late_lookup[train_strike, martin_oversleeps]

# P(norman_late|train_strike) --------------------------------------------
p_norman_late_lookup = theano.shared(np.asarray([[0.9, .1], [.2, .8]]))

def p_norman_late(train_strike):
    return p_norman_late_lookup[train_strike]

# Build model ------------------------------------------------------------
with pm.Model() as m:
    martin_oversleeps = pm.Categorical('martin_oversleeps', [.6, .4])
    train_strike = pm.Categorical('train_strike', [.9, .1])
    norman_late = pm.Categorical('norman_late',
                                 p_norman_late(train_strike))
    martin_late = pm.Categorical('martin_late',
                                 p_martin_late(train_strike, martin_oversleeps))
    prior = pm.sample_prior_predictive(500_000)
