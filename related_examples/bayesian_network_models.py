# https://gist.github.com/tbsexton/1349864212b25cce91dbe5e336d794b4
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

import pymc3 as pm
import theano.tensor as T
from theano.compile.ops import as_op
import theano

d1_prob = np.array([0.3,0.7])  # 2 choices
d2_prob = np.array([0.6,0.3,0.1])  # 3 choices
d3_prob = np.array([[[0.1, 0.9],  # (2x3)x2 choices
                     [0.3, 0.7],
                     [0.4, 0.6]],
                    [[0.6, 0.4],
                     [0.8, 0.2],
                     [0.9, 0.1]]])
d4_prob = np.array([[[0.4, 0.6],  # (2x3)x2 choices
                     [0.6, 0.4],
                     [0.3, 0.7]],
                    [[0.4, 0.6],
                     [0.3, 0.7],
                     [0.1, 0.9]]])

c1_mu, c1_sd = np.array([[10, 14],  # 2 choices inherit
                         [2 , 2 ]])
c2_mu, c2_sd = np.array([[6, 8, 14],  # 3 choices inherit
                         [2, 1, 1 ]])

with pm.Model() as model:

    D1 = pm.Categorical('D1',p=d1_prob)
    D2 = pm.Categorical('D2',p=d2_prob)

    C1 = pm.Normal('C1',mu = 10 + 4*D1, tau = (1./2)**2)
#     p1 = pm.Dirichlet('p1', d1_prob)                     # inexplicable failure
#     C1 = pm.NormalMixture('C1', p1, mu=c1_mu, sd=c1_sd)  # inexplicable failure

    C2 = pm.Normal('C2',mu=6+2*(D2**2), tau=1)
#     p2 = pm.Dirichlet('p2', d2_prob)                     # inexplicable failure
#     C2 = pm.NormalMixture('C2', p2, mu=c2_mu, sd=c2_sd)  # inexplicable failure

    D3_prob = theano.shared(d3_prob)  # make numpy-->theano
    D3_0 = D3_prob[D1, D2]  # select the prob array that "happened" thanks to parents
    D3 = pm.Categorical('D3',p=D3_0)

    C3 = pm.Normal('C3',mu = (0.15*(C2**2)*(1-D3) + 1.5*C2*D3), tau=(1./(2-D3))**2)
    C4 = pm.Normal('C4',mu = 0.1*C2**2 + 0.6*C2+1, tau = 0.25, observed = [7])

#     C3_0 = np.select([T.lt(C3,9), T.gt(C3,9) & T.lt(C3,11), T.gt(C3,11)], [0,1,2])  # doesnt work in Theano
    C3_0 = T.switch(T.lt(C3,9), 0,
                   T.switch(T.gt(C3, 9) & T.lt(C3,11), 1, 2))  # ugly (and hard to generalize)

    D4_prob = theano.shared(d4_prob)  # make numpy-->theano
    D4_0 = D4_prob[D3, C3_0]  # select the prob array that "happened" thanks to parents

    D4 = pm.Categorical('D4', p=D4_0, observed=[0])

# Create MCMC object

with model:
    trace = pm.sample(10000)

print(pm.summary(trace, varnames=['C1', 'C2', 'C3'], start=1000))

pm.traceplot(trace)

pm.plot_forest(trace)
