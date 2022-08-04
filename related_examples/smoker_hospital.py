import pymc3 as pm
import numpy as np

import theano
import theano.tensor as tt

lookup_table = theano.shared(np.asarray([
    # smoker
    [C(.01), C(.9)], # covid
    [C(.1), C(.9)]
    ]))



def C(p):
    return [1-p, p]

def f(smoker, covid):
    return lookup_table[smoker, covid]

lookup_table2 = theano.shared(np.asarray([
    [[.99, .01], [.1, .9]],
    [[.9, .1], [.1, .9]]]))

def f2(smoker, covid):
    return lookup_table2[smoker, covid]

with pm.Model() as m:
    smoker = pm.Categorical('smoker', [.75, .25])
    covid = pm.Categorical('covid', [.9, .1])
    hospital = pm.Categorical('hospital', f(smoker, covid))

    smoker2 = pm.Categorical('smoker2', pm.math.switch(hospital, [1.,0.], [0., 1.]))
    covid2 = pm.Categorical('covid2', [.9, .1])
    hospital2 = pm.Categorical('hospital2', f2(smoker2, covid2))

    prior_trace = pm.sample_prior_predictive(1000)

predict_proba0 = prior_trace['covid'][
    (prior_trace['smoker'] == 0)
  & (prior_trace['hospital'] == 1)].mean()
predict_proba1 = prior_trace['covid'][
    (prior_trace['smoker'] == 1)
  & (prior_trace['hospital'] == 1)].mean()

print(f'P(covid|¬smoking, hospital) is {predict_proba0}')
print(f'P(covid|smoking, hospital) is {predict_proba1}')

pm.model_to_graphviz(m)
