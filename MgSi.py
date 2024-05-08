import pymc3 as pm
import pandas as pd
import theano.tensor as tt
import numpy as np

data = pd.read_csv('')

X = data[['A', 'Z', 'N', 'NJ', 'E(MeV)', 'T_z', 'p']].values
y = data['Ex(MeV)'].values

n_input = X.shape[1]
n_hidden = 10

with pm.Model() as bayesian_nn
    w_in = pm.Normal('w_in', mu=0, sigma=1, shape=(n_input, n_hidden))
    b_in = pm.Normal('b_in', mu=0, sigma=1, shape=(n_hidden,))

    w_out = pm.Normal('w_out', mu=0, sigma=1, shape=(n_hidden, 1))
    b_out = pm.Normal('b_out', mu=0, sigma=1, shape=(1,))

    act_in = tt.tanh(tt.dot(X, w_in) + b_in.reshape((1, n_hidden)))
    act_out = tt.dot(act_in, w_out) + b_out

    sigma = pm.HalfCauchy('sigma', beta=1)
    y_obs = pm.Normal('y_obs', mu=act_out[:, 0], sigma=sigma, observed=y)

    trace = pm.sample(draws=1000, tune=1000, cores=1)

new_data = np.array([[28, 8, 4, 1, -120.5, 6, +1]])
preds = pm.sampling.sample_posterior_predictive(trace, samples=1000, model=bayesian_nn, var_names=['y_obs'], return_inferencedata=False)['y_obs']

print(f"Predicted Ex(MeV) for A=28, Z=8, N=4, NJ=1, E(MeV)=-120.5, T_z=6, p=+1:")
print(f"Mean: {np.mean(preds):.2f} MeV")
print(f"Standard Deviation: {np.std(preds):.2f} MeV")