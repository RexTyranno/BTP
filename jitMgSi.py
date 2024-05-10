import pymc3 as pm
import pandas as pd
import theano
import theano.tensor as tt
import numpy as np
from sklearn.model_selection import train_test_split
import arviz as az

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'

def create_bayesian_nn(n_input, n_hidden):
    with pm.Model() as bayesian_nn:
        w_in = pm.Normal('w_in', mu=0, sigma=1, shape=(n_input, n_hidden))
        b_in = pm.Normal('b_in', mu=0, sigma=1, shape=(n_hidden,))
        w_out = pm.Normal('w_out', mu=0, sigma=1, shape=(n_hidden, 1))
        b_out = pm.Normal('b_out', mu=0, sigma=1, shape=(1,))
        act_in = tt.tanh(tt.dot(X, w_in) + b_in.reshape((1, n_hidden)))
        act_out = tt.dot(act_in, w_out) + b_out
        sigma = pm.HalfCauchy('sigma', beta=1)
        y_obs = pm.Normal('y_obs', mu=act_out[:, 0], sigma=sigma, observed=y)
    return bayesian_nn

def get_predictions(X, w_in, b_in, w_out, b_out):
    n_hidden = w_in.shape[1]
    act_in = tt.tanh(tt.dot(X, w_in) + b_in.reshape((1, n_hidden)))
    act_out = tt.dot(act_in, w_out) + b_out
    return act_out[:, 0]

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

data = pd.read_csv('/home/agrima/Data/BTP/Combined.csv')

X = data[['A', 'Z', 'N', 'NJ', 'E(MeV)', 'T_z', 'p']].values
y = data['Ex(MeV)'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

n_input = X.shape[1]
n_hidden = 10

bayesian_nn = create_bayesian_nn(n_input, n_hidden)

with bayesian_nn:
    trace = pm.sample(draws=2000, tune=2000, cores=1)
    trace = az.from_pymc3(trace=trace, prior=None, posterior_predictive=None)

new_data = np.array([[28, 8, 4, 1, -120.5, 6, +1]])
preds = pm.sample_posterior_predictive(trace, samples=1000, model=bayesian_nn, var_names=['y_obs'])['y_obs']

train_mae = []
val_mae = []

coords = trace.posterior.coords
nchains = coords['chain'].size
ndraws = coords['draw'].size

for epoch in range(nchains * ndraws):
    chain_idx = epoch // ndraws
    draw_idx = epoch % ndraws

    w_in_val = trace.posterior['w_in'].values[chain_idx, draw_idx]
    b_in_val = trace.posterior['b_in'].values[chain_idx, draw_idx]
    w_out_val = trace.posterior['w_out'].values[chain_idx, draw_idx]
    b_out_val = trace.posterior['b_out'].values[chain_idx, draw_idx]

    train_preds = get_predictions(X_train, w_in_val, b_in_val, w_out_val, b_out_val)
    val_preds = get_predictions(X_val, w_in_val, b_in_val, w_out_val, b_out_val)

    train_mae.append(mean_absolute_error(y_train, train_preds))
    val_mae.append(mean_absolute_error(y_val, val_preds))

print(f"Predicted Ex(MeV) for A=28, Z=8, N=4, NJ=1, E(MeV)=-120.5, T_z=6, p=+1:")
print(f"Mean: {np.mean(preds):.2f} MeV")
print(f"Standard Deviation: {np.std(preds):.2f} MeV")

results_df = pd.DataFrame({'Train_MAE': train_mae, 'Val_MAE': val_mae})

results_df.to_csv('results.csv')
