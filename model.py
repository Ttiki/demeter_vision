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


def create_scenarios():
    """
    Create scenarios based on temperature and rainfall classes.

    Returns:
    - Dictionary of scenarios and their corresponding labels.
    """
    CT = {'T1': (-float('inf'), 21.2), 'T2': (21.2, 22), 'T3': (22, float('inf'))}
    CR = {'R1': (-float('inf'), 1.8), 'R2': (1.8, 2.2), 'R3': (2.2, float('inf'))}

    # Compute Cartesian product of temperature and rainfall classes
    scenarios = list(itertools.product(CR.keys(), CT.keys()))

    # Create a dictionary to store the scenarios and their corresponding labels
    scenario_labels = {f"x{i + 1}": x for i, x in enumerate(scenarios)}

    return scenario_labels


def load_and_preprocess_data():
    """
    Load and preprocess data from CSV files.

    Returns:
    - Combined and preprocessed data.
    """
    station_40 = pd.read_csv("data/station_40.csv")
    station_49 = pd.read_csv("data/station_49.csv")
    station_63 = pd.read_csv("data/station_63.csv")
    station_80 = pd.read_csv("data/station_80.csv")

    # Combine, preprocess, and filter data
    combined_data_features = preprocess_and_combine_data(
        station_40, station_49, station_63, station_80
    )

    return combined_data_features


def preprocess_and_combine_data(*stations):
    """
    Combine, preprocess, and filter data from multiple stations.

    Args:
    - stations: DataFrames representing different stations.

    Returns:
    - Combined and preprocessed data.
    """
    features_list = [station.iloc[:, :-1] for station in stations]
    yield_list = [station.iloc[:, -1] for station in stations]

    # Combine data from all stations
    combined_data_features = pd.concat(features_list, axis=1)
    combined_data_yield = pd.concat(yield_list, axis=1)

    # Combine features and yield
    return pd.concat([combined_data_features, combined_data_yield], axis=1)


def remove_highly_correlated_features(features):
    """
    Remove highly correlated features from the input DataFrame.

    Args:
    - features: Input features DataFrame.

    Returns:
    - DataFrame with highly correlated features removed.
    """
    correlation_threshold = 0.8
    correlation_matrix = np.corrcoef(features, rowvar=False)
    correlated_features = np.where(np.abs(correlation_matrix) > correlation_threshold)
    unique_correlated_features = {tuple(sorted([i, j])) for i, j in zip(*correlated_features)}
    to_delete = [item for sublist in unique_correlated_features for item in sublist]
    return features.drop(features.columns[to_delete], axis=1)


def standardize_data(features):
    """
    Standardize data using StandardScaler.

    Args:
    - features: Input features DataFrame or array-like.

    Returns:
    - Standardized features.
    """
    # If features is a DataFrame, check columns and convert to NumPy array
    if isinstance(features, pd.DataFrame):
        if features.empty or not features.columns.any():
            raise ValueError("Input DataFrame 'features' is empty or does not have columns.")

        features = features.values

    # Check if features is a 2D array-like object
    if not isinstance(features, (np.ndarray, np.generic)) or features.ndim != 2:
        raise ValueError("Input features must be a 2D array-like object.")

    # Check if there is at least one feature
    if features.shape[1] == 0:
        raise ValueError("Input features must have at least one feature.")

    scaler = StandardScaler()
    return scaler.fit_transform(features)


def split_data(features, targets):
    """
    Split data into training and testing sets.

    Args:
    - features: Input features.
    - targets: Target values.

    Returns:
    - Split data: x_train, x_test, y_train, y_test.
    """
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def visualize_correlation(features):
    """
    Visualize correlation among features using a heatmap.

    Args:
    - features: Input features DataFrame.
    """
    correlation_matrix = np.corrcoef(features, rowvar=False)
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", xticklabels=features.columns,
                yticklabels=features.columns)
    plt.title('Correlation Heatmap')
    plt.show()


def sigmoid(x):
    """
    Sigmoid function.

    Args:
    - x: Input value.

    Returns:
    - Sigmoid value.
    """
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(y):
    """
    Inverse sigmoid using fsolve.

    Args:
    - y: Input value.

    Returns:
    - Inverse sigmoid value.
    """
    equation = lambda x: sigmoid(x) - y
    x_solution = fsolve(equation, 0.0)[0]
    return x_solution


def build_vae_model(latent_dim, input_dim, hidden_dim):
    """
    Build VAE model using stax.

    Args:
    - latent_dim: Dimensionality of the latent space.
    - input_dim: Dimensionality of the input.
    - hidden_dim: Dimensionality of the hidden layer.

    Returns:
    - Tuple of encoder and decoder functions.
    """
    encoder_init, encoder_fn = stax.serial(Dense(hidden_dim), Relu, Dense(latent_dim * 2))
    decoder_init, decoder_fn = stax.serial(Dense(hidden_dim), Relu, Dense(input_dim), Sigmoid)
    return encoder_init, encoder_fn, decoder_init, decoder_fn


@jit
def encoder_fn(params, x):
    """
    Encoder function.

    Args:
    - params: Tuple of parameters for the encoder.
    - x: Input data.

    Returns:
    - Output of the encoder.
    """
    hidden_layer = jnp.dot(x, params[0]) + params[1]
    activation = jnp.tanh(hidden_layer)
    output = jnp.dot(activation, params[2]) + params[3]

    return output


@jit
def fast_sample(rand_key, z_mean, z_log_var):
    """
    Fast sampling from the latent space using the reparameterization trick.

    Args:
    - rand_key: Random key for JAX's random module.
    - z_mean: Mean of the latent space.
    - z_log_var: Logarithm of the variance of the latent space.

    Returns:
    - Sampled latent vector.
    """
    epsilon = random.normal(rand_key, shape=z_mean.shape)
    z_sample = z_mean + jnp.exp(0.5 * z_log_var) * epsilon

    return z_sample


@jit
def decoder_fn(params, z_sample):
    """
    Decoder function to generate samples from the latent space.

    Args:
    - params: Decoder parameters.
    - z_sample: Sampled latent vector.

    Returns:
    - Generated samples.
    """
    for w, b in params:
        z_sample = jnp.dot(z_sample, w) + b
        z_sample = jnp.relu(z_sample)

    output = jnp.sigmoid(z_sample)

    return output


@jit
def vae_loss(rand_key, params, x):
    """
    Calculate VAE loss.

    Args:
    - rand_key: Random key for JAX's random module.
    - params: Tuple of encoder and decoder parameters.
    - x: Input data.

    Returns:
    - Negative ELBO (Evidence Lower Bound).
    """
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


def get_params(opt_state):
    """
    Get model parameters from the optimization state.

    Args:
    - opt_state: Optimization state.

    Returns:
    - Model parameters.
    """
    return opt_state[1]


@jit
def update(key, batch, opt_state, opt_update):
    """
    Update VAE model parameters.

    Args:
    - key: Random key for JAX's random module.
    - batch: Input batch.
    - opt_state: Optimization state.
    - opt_update: Update function for the optimizer.

    Returns:
    - Updated optimization state and loss.
    """
    params = get_params(opt_state)
    loss, grads = value_and_grad(lambda params, x: vae_loss(key, params, x))(params, batch)
    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    opt_state = opt_update(0, grads, opt_state)
    return opt_state, loss


def train_vae_model(x_train_standard, encoder_init, encoder_fn, decoder_init, decoder_fn, input_dim, latent_dim,
                    learning_rate, data_size, EPOCHS=200):
    """
    Train VAE model.

    Args:
    - x_train_standard: Standardized training data.
    - encoder_init: Encoder initialization function.
    - encoder_fn: Encoder function.
    - decoder_init: Decoder initialization function.
    - decoder_fn: Decoder function.
    - input_dim: Dimensionality of the input.
    - latent_dim: Dimensionality of the latent space.
    - learning_rate: Learning rate for optimization.
    - data_size: Size of the training data.
    - EPOCHS: Number of training epochs.

    Returns:
    - Trained VAE model parameters.
    """
    rand_key = jax.random.PRNGKey(42)
    params_enc = encoder_init(rand_key, (-1, input_dim))
    params_dec = decoder_init(rand_key, (-1, latent_dim))
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
            opt_state, loss = update(key, batch, opt_state, opt_update)
            losses.append(loss)

    return opt_state, get_params(opt_state), decoder_fn


def generator(params_dec, noise_and_label):
    """
    Generator function for VAE model.

    Args:
    - params_dec: Decoder parameters.
    - noise_and_label: Input noise and scenario label.

    Returns:
    - Generated samples.
    """
    noise, scenario_label = jnp.split(noise_and_label, [-1])
    input_vector = jnp.concatenate([noise, scenario_label], axis=-1)
    generated_samples = decoder_fn(params_dec, input_vector)
    return generated_samples


def generate_samples(opt_state, params_dec, noise, scenario_label, num_samples=50):
    """
    Generate samples using trained VAE model.

    Args:
    - opt_state: Optimization state.
    - params_dec: Decoder parameters.
    - noise: Noise input.
    - scenario_label: Scenario label input.
    - num_samples: Number of samples to generate.

    Returns:
    - Generated samples.
    """
    input_vector = jnp.concatenate([noise, scenario_label], axis=-1)
    generated_samples = generator(params_dec, input_vector)
    return generated_samples


def generative_model(noise, scenario):
    """
    Generative model using VAE.

    Args:
    - noise: Noise input.
    - scenario: Scenario label input.

    Returns:
    - Generated samples.
    """
    # Load and preprocess data
    features = load_and_preprocess_data()

    # Check if 'features' is empty or does not have columns
    if features is None:
        raise ValueError("Loaded data is empty. Check your data loading process.")
    elif isinstance(features, pd.DataFrame) and (features.empty or not features.columns.any()):
        raise ValueError(
            "Loaded DataFrame 'features' is empty or does not have columns. Check your data preprocessing.")

    #visualize_correlation(features)
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
