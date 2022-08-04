import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import theano.tensor as tt
tr = pm.distributions.transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
floatX = theano.config.floatX

warnings.simplefilter(action="ignore", category=FutureWarning)

%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
print(f"Running on PyMC3 v{pm.__version__}")
print(f"Running on ArviZ v{az.__version__}")


with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))

model.basic_RVs
model.free_RVs
model.observed_RVs
model.logp({"mu": 0})

%timeit model.logp({mu: 0.1})
logp = model.logp
%timeit logp({mu: 0.1})

help(pm.Normal)

dir(pm.distributions.mixture)

with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)

x.logp({"x": 0})

np.random.randn(10)
with pm.Model():
    obs = pm.Normal("x", mu=0, sigma=1, observed=[1,1,1,1])
    x = pm.Uniform("xx", lower=0, upper=1, transform=None)
    pm.sample()
obs
print(model.free_RVs)



with pm.Model() as model:
    marry_calls = pm.Normal("marry_calls", mu=0, sigma=1)
    george_calls = pm.Gamma("george_calls", alpha=1, beta=1)

    alarm_rings = pm.Deterministic("alarm_rings", marry_calls*george_calls)

    burglary = pm.Normal("burglary", alarm_rings)
    earthquake = pm.Normal("earthquake", alarm_rings)

    pm.sample()

pm.model_to_graphviz(model)


with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1)
    plus_2 = pm.Deterministic("x plus 2", x + 2)

with pm.Model() as model:
    x = pm.Uniform("x", lower=0, upper=1)



model.free_RVs
model.deterministics

with pm.Model() as model:
    x = pm.Uniform("x", lower=0, upper=1, transform=None)

print(model.free_RVs)










class Exp(tr.ElemwiseTransform):
    name = "exp"

    def backward(self, x):
        return tt.log(x)

    def forward(self, x):
        return tt.exp(x)

    def jacobian_det(self, x):
        return -tt.log(x)



with pm.Model() as model:
    x1 = pm.Normal("x1", 0.0, 1.0, transform=Exp())
    x2 = pm.Lognormal("x2", 0.0, 1.0)

model.named_vars.keys()

lognorm1 = model.named_vars["x1_exp__"]
lognorm2 = model.named_vars["x2"]

_, ax = plt.subplots(1, 1, figsize=(5, 3))
x = np.linspace(0.0, 10.0, 100)
ax.plot(
    x,
    np.exp(lognorm1.distribution.logp(x).eval()),
    "--",
    alpha=0.5,
    label="log(y) ~ Normal(0, 1)",
)
ax.plot(
    x,
    np.exp(lognorm2.distribution.logp(x).eval()),
    alpha=0.5,
    label="y ~ Lognormal(0, 1)",
)
plt.legend();

















Order = tr.Ordered()
Logodd = tr.LogOdds()
chain_tran = tr.Chain([Logodd, Order])

with pm.Model() as m0:
    x = pm.Uniform("x", 0.0, 1.0, shape=2, transform=chain_tran, testval=[0.1, 0.9])
    trace = pm.sample(5000, tune=1000, progressbar=False, return_inferencedata=False)

_, ax = plt.subplots(1, 2, figsize=(10, 5))
for ivar, varname in enumerate(trace.varnames):
    ax[ivar].scatter(trace[varname][:, 0], trace[varname][:, 1], alpha=0.01)
    ax[ivar].set_xlabel(varname + "[0]")
    ax[ivar].set_ylabel(varname + "[1]")
    ax[ivar].set_title(varname)
plt.tight_layout()







'''
Test 1 - Defining probabilistic distributions
'''
with pm.Model() as model:
    # distance from eef to object
    eef_dist = pm.Normal('eef_dist', mu=0, sigma=1)

    # feasibility
    feasibility = pm.Bernoulli('feasibility', p=0.5)
    #feasibility = pm.Uniform('feasibility', lower=0, upper=1)

    #
    common_sense = pm.Beta('common_sense', alpha=0.5, beta=0.5)

    #
    #doability = pm.Sigmoid()


    idata = pm.sample(1000, tune=1500, return_inferencedata=True)

idata.posterior.dims
idata.posterior['eef_dist'].shape
idata.posterior["eef_dist"].sel(chain=1).shape
idata.posterior["eef_dist"].sel(chain=1)[0]

'''
Test plotting & Savefig (.SVG)
'''
fig, axarr = plt.subplots(3,2)
trarr = pm.traceplot(idata)
fig = plt.gcf() # to get the current figure...
fig.savefig("/home/petr/Downloads/disaster.svg") # and save it directly
data = az.summary(idata)
data
data.to_csv('/home/petr/Downloads/disaster.csv')

data = pd.read_csv('/home/petr/Downloads/disaster.csv')
pd.DataFrame(data)
data

az.plot_forest(idata, r_hat=True);
fig, axarr = plt.subplots(1,1)
trarr = az.plot_forest(idata, r_hat=True)
fig = plt.gcf() # to get the current figure...
fig.savefig("/home/petr/Downloads/disaster2.svg") # and save it directly

# az.plot_posterior(idata);
d = pm.Normal.dist(mu=0, sigma=0.01)
d.logp(10).eval()
d.random()

'''
Springler example AIMA
'''

import numpy as np
import pandas as pd
import pymc3 as pm

tune = 5000  # 5000

model = pm.Model()
with model:
    tv = [1]
    rain = pm.Bernoulli('rain', 0.2, shape=1, testval=tv)

    sprinkler_p = pm.Deterministic('sprinkler_p', pm.math.switch(rain, 0.01, 0.40))

    sprinkler = pm.Bernoulli('sprinkler', sprinkler_p, shape=1, testval=tv)

    grass_wet_p = pm.Deterministic('grass_wet_p', pm.math.switch(rain, pm.math.switch(sprinkler, 0.99, 0.80), pm.math.switch(sprinkler, 0.90, 0.0)))

    grass_wet = pm.Bernoulli('grass_wet', grass_wet_p, observed=np.array([1]), shape=1)

    trace = pm.sample(20000, step=[pm.BinaryGibbsMetropolis([rain, sprinkler])], tune=tune, random_seed=124)

# pm.traceplot(trace)
pm.model_to_graphviz(model)

fig, axarr = plt.subplots(1,1)
trarr = pm.traceplot(trace)
fig = plt.gcf() # to get the current figure...
fig.savefig("/home/petr/Downloads/disaster3.svg") # and save it directly

dictionary = {
              'Rain': [1 if ii[0] else 0 for ii in trace['rain'].tolist() ],
              'Sprinkler': [1 if ii[0] else 0 for ii in trace['sprinkler'].tolist() ],
              'Sprinkler Probability': [ii[0] for ii in trace['sprinkler_p'].tolist()],
              'Grass Wet Probability': [ii[0] for ii in trace['grass_wet_p'].tolist()],
              }
df = pd.DataFrame(dictionary)
df

p_rain = df[(df['Rain'] == 1)].shape[0] / df.shape[0]
print(p_rain)

p_sprinkler = df[(df['Sprinkler'] == 1)].shape[0] / df.shape[0]
print(p_sprinkler)





objs = ['box1', 'box2']
gs = ['closed', 'opened']
fts = ['gripper_opened']
obj_fts = ['object_graspable']
acts = ['grab', 'release']

n_objects = len(objs)
n_gestures = len(gs)
n_features = len(fts) + len(obj_fts) * n_objects
n_actions = len(acts)
n_action_params = n_objs

with pm.Model() as model:
    # inputs
    gestures = pm.Normal("gestures", 0, sigma=1, shape=(n_gestures), testval=[1]*n_gestures)
    features = pm.Bernoulli("features", 0, shape=(n_features), testval=[1]*n_features)
    eef_dists = pm.Normal("eef_dists", 0, sigma=1, shape=(n_objects), testval=[1]*n_objects)



    # outputs
    actions = pm.Deterministic("actions", gestures[0]*features[0]) #gestures[1]*(1-features[0])])
    action_parameters = pm.Deterministic("action_parameters", eef_dists*features[1:3])

    #pm.sample()

pm.model_to_graphviz(model)


with pm.Model() as model:
    bb = pm.Bernoulli('bb', 0, shape=(2))
    gestures = pm.Normal("gestures", [0,1,2], sigma=1, shape=(3))#, testval=[1]*n_gestures)
    #pm.Deterministic('probability_g', np.exp(gestures.distribution.logp([0,1,2]).eval()))
    print("logp", gestures.distribution.logp(0).eval())
print("logp", gestures.distribution.logp(0).eval())
print("logp", bb.distribution.logp(0).eval())
a
x = [0,0,0]



np.exp(model.named_vars["gestures"].distribution.logp(x).eval())
model.named_vars['gestures']



def construct_nn(ann_input, ann_output):
    n_hidden = 5

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        # Trick: Turn inputs and outputs into shared variables using the data container pm.Data
        # It's still the same thing, but we can later change the values of the shared variable
        # (to switch in the test-data later) and pymc3 will just use the new data.
        # Kind-of like a pointer we can redirect.
        # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", Y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(X.shape[1], n_hidden), testval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden, n_hidden), testval=init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=1, shape=(n_hidden,), testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
            "out",
            act_out,
            observed=ann_output,
            total_size=Y_train.shape[0],  # IMPORTANT for minibatches
        )
    return neural_network

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import sklearn
import theano
import theano.tensor as T

from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

floatX = theano.config.floatX
X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

neural_network = construct_nn(X_train, Y_train)

pm.model_to_graphviz(neural_network)

























#
