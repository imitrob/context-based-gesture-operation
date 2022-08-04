# Original example in PyMC2:
# https://bugra.github.io/work/notes/2014-05-23/simple-bayesian-network-via-monte-carlo-markov-chain-mcmc-pymc/

import pandas as pd
import pymc3 as pm

model = pm.Model()

with model:
    rain = pm.Bernoulli('rain', 0.2)
    sprinkler_p = pm.Deterministic('sprinkler_p', pm.math.switch(rain, 0.01, 0.40))
    sprinkler = pm.Bernoulli('sprinkler', sprinkler_p)
    grass_wet_p = pm.Deterministic('grass_wet_p', pm.math.switch(rain, pm.math.switch(sprinkler, 0.99, 0.80), pm.math.switch(sprinkler, 0.90, 0.0)))
    grass_wet = pm.Bernoulli('grass_wet', grass_wet_p)

    # start = pm.find_MAP()  # Use MAP estimate (optimization) as the initial state for MCMC
    step = pm.Metropolis()
    trace = pm.sample(100000, step=step, tune=5000, random_seed=123, progressbar=True)  # init=start,

pm.traceplot(trace)

dictionary = {
              'Rain': [1 if ii else 0 for ii in trace['rain'].tolist() ],
              'Sprinkler': [1 if ii else 0 for ii in trace['sprinkler'].tolist() ],
              'Sprinkler Probability': [ii for ii in trace['sprinkler_p'].tolist()],
              'Grass Wet Probability': [ii for ii in trace['grass_wet_p'].tolist()],
              }
df = pd.DataFrame(dictionary)
df.head()

# Given grass is wet, what is the probability that it was rained?
p_rain_wet = float(df[(df['Rain'] == 1) & (df['Grass Wet Probability'] > 0.5)].shape[0]) / df[df['Grass Wet Probability'] > 0.5].shape[0]
print(p_rain_wet)

# Given grass is wet, what is the probability that sprinkler was opened?
p_sprinkler_wet = float(df[(df['Sprinkler'] == 1) & (df['Grass Wet Probability'] > 0.5)].shape[0]) / df[df['Grass Wet Probability'] > 0.5].shape[0]
print(p_sprinkler_wet)

# Given sprinkler is off and it does not rain, what is the probability that grass is wet?
p_not_sprinkler_rain_wet = float(df[(df['Sprinkler'] == 0) & (df['Rain'] == 0) & (df['Grass Wet Probability'] > 0.5)].shape[0]) / df[df['Grass Wet Probability'] > 0.5].shape[0]
print(p_not_sprinkler_rain_wet)
