import os
import itertools

import jax
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import jax.numpy as jnp
from jax import jit, value_and_grad, random
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Dense, Relu, Sigmoid

from parameters.VAE import encoder_fn, fast_sample, decoder_fn, get_params, opt_update, input_dim, latent_dim, \
    learning_rate, data_size

def create_scenarios():
    # Define temperature and rainfall classes
    CT = {'T1': (-float('inf'), 21.2), 'T2': (21.2, 22), 'T3': (22, float('inf'))}
    CR = {'R1': (-float('inf'), 1.8), 'R2': (1.8, 2.2), 'R3': (2.2, float('inf'))}

    # Compute Cartesian product of temperature and rainfall classes
    scenarios = list(itertools.product(CR.keys(), CT.keys()))

    # Create a dictionary to store the scenarios and their corresponding labels
    scenario_labels = {f"x{i + 1}": x for i, x in enumerate(scenarios)}

    return scenario_labels

# Load and preprocess data
def load_and_preprocess_data():
    directory = os.path.abspath(__file__)
    enclosingfolder = os.path.dirname(directory)
    enclosingfolder = os.path.dirname(enclosingfolder)
    noise_data_path_elhcs = os.path.join(enclosingfolder, 'data', 'noise.npy')

    # Load data from CSV files
    station_40_path = os.path.join(enclosingfolder, 'data', 'station_40.csv')
    station_49_path = os.path.join(enclosingfolder, 'data', 'station_49.csv')
    station_63_path = os.path.join(enclosingfolder, 'data', 'station_63.csv')
    station_80_path = os.path.join(enclosingfolder, 'data', 'station_80.csv')

    station_40 = pd.read_csv(station_40_path)
    station_49 = pd.read_csv(station_49_path)
    station_63 = pd.read_csv(station_63_path)
    station_80 = pd.read_csv(station_80_path)

    # Combine, preprocess, and filter data
    station_40_features = station_40.iloc[:, :-1]
    station_40_yield = station_40.iloc[:, -1]
    station_49_features = station_49.iloc[:, :-1]
    station_49_yield = station_49.iloc[:, -1]
    station_63_features = station_63.iloc[:, :-1]
    station_63_yield = station_63.iloc[:, -1]
    station_80_features = station_80.iloc[:, :-1]
    station_80_yield = station_80.iloc[:, -1]

    # Combine data from all stations
    combined_data_features = pd.concat(
        [station_40_features, station_49_features, station_63_features, station_80_features], axis=1)
    combined_data_yield = pd.concat([station_40_yield, station_49_yield, station_63_yield, station_80_yield], axis=1)

    return pd.concat([combined_data_features, combined_data_yield], axis=1)

# Remove highly correlated features
def remove_highly_correlated_features(features):
    correlation_threshold = 0.8
    correlation_matrix = np.corrcoef(features, rowvar=False)
    correlated_features = np.where(np.abs(correlation_matrix) > correlation_threshold)
    unique_correlated_features = {tuple(sorted([i, j])) for i, j in zip(*correlated_features)}
    to_delete = [item for sublist in unique_correlated_features for item in sublist]
    return features.drop(features.columns[to_delete], axis=1)

# Standardize data using StandardScaler
def standardize_data(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Split data into training and testing sets
def split_data(features, targets):
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

# Visualize correlation among features
def visualize_correlation(features):
    correlation_matrix = np.corrcoef(features, rowvar=False)
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=features.columns, yticklabels=features.columns)
    plt.title('Correlation Heatmap')
    plt.show()

# Helper function for inverse sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Inverse sigmoid using fsolve
def inverse_sigmoid(y):
    equation = lambda x: sigmoid(x) - y
    x_solution = fsolve(equation, 0.0)[0]
    return x_solution

# Build VAE model using stax
def build_vae_model(latent_dim, input_dim, hidden_dim):
    encoder_init, encoder_fn = stax.serial(Dense(hidden_dim), Relu, Dense(latent_dim * 2))
    decoder_init, decoder_fn = stax.serial(Dense(hidden_dim), Relu, Dense(input_dim), Sigmoid)
    return encoder_init, encoder_fn, decoder_init, decoder_fn

# Calculate VAE loss
@jit
def vae_loss(rand_key, params, x):
    latent = encoder_fn(params[0:3], x)
    d = latent.shape[-1] // 2
    z_mean, z_log_var = latent[:, :d], latent[:, d:]
    z_sample = fast_sample(rand_key, z_mean, z_log_var)
    x_rec = decoder_fn(params[3:], z_sample)
    x_rec = x_rec[:, :x.shape[1]]
    EPSILON = 1e-6
    negative_xent = - jnp.sum(x * jnp.log(x_rec + EPSILON) + (1 - x) * jnp.log(1 - x_rec + EPSILON), axis=-1)
    negative_kl = - 0.5 * jnp.sum(1 + z_log_var - z_mean ** 2 - jnp.exp(z_log_var), axis=-1)
    negative_elbo = jnp.mean(negative_xent) + jnp.mean(negative_kl)
    return negative_elbo

# Update VAE model parameters
@jit
def update(key, batch, opt_state):
    params = get_params(opt_state)
    loss, grads = value_and_grad(lambda params, x: vae_loss(key, params, x))(params, batch)
    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    opt_state = opt_update(0, grads, opt_state)
    return opt_state, loss

# Train VAE model
def train_vae_model(x_train_standard, encoder_init, encoder_fn, decoder_init, decoder_fn, EPOCHS=200):
    rand_key = jax.random.PRNGKey(42)
    _, params_enc = encoder_init(rand_key, (-1, input_dim))
    _, params_dec = decoder_init(rand_key, (-1, latent_dim))
    params = params_enc + params_dec
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)
    losses = []

    for epoch in range(EPOCHS):
        rand_key, key = random.split(rand_key)
        permutation = random.permutation(key, data_size)

        for i in range(data_size // 64 - 1):
            batch = x_train_standard[permutation[i * 32:(i + 1) * 32]]
            rand_key, key = random.split(rand_key)
            opt_state, loss = update(key, batch, opt_state)
            losses.append(loss)

    return opt_state, get_params(opt_state), decoder_fn

# Generator function for VAE model
def generator(params_dec, noise_and_label):
    noise, scenario_label = jnp.split(noise_and_label, [-1])
    input_vector = jnp.concatenate([noise, scenario_label], axis=-1)
    generated_samples = decoder_fn(params_dec, input_vector)
    return generated_samples

# Generate samples using trained VAE model
def generate_samples(opt_state, params_dec, noise, scenario_label, num_samples=50):
    input_vector = jnp.concatenate([noise, scenario_label], axis=-1)
    generated_samples = generator(params_dec, input_vector)
    return generated_samples

# Generative model
def generative_model(noise, scenario):
    # Load and preprocess data
    features = load_and_preprocess_data()
    visualize_correlation(features)
    features = remove_highly_correlated_features(features)
    features = standardize_data(features)

    # Create scenarios and labels
    scenarios_and_labels = create_scenarios()

    # Display scenarios and labels
    for label, scenario in scenarios_and_labels.items():
        print(f"{label}: {scenario}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(features, features['YIELD'])

    # Train VAE model
    latent_dim = 50
    input_dim = X_train.shape[1]
    hidden_dim = 51
    encoder_init, encoder_fn, decoder_init, decoder_fn = build_vae_model(latent_dim, input_dim, hidden_dim)
    opt_state, params_enc, params_dec = train_vae_model(X_train, encoder_init, encoder_fn, decoder_init, decoder_fn)

    # Generate samples for given noise and scenario
    generated_samples = generate_samples(opt_state, params_dec, noise, scenario)

    return generated_samples
