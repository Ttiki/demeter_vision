#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# Z,x |-> G_\theta(Z,x)
############################################################################
import jax.numpy as jnp
import jax
import numpy as np

from jax import jit, value_and_grad, random
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Dense, Relu, Sigmoid
from scipy.optimize import fsolve

from genhack_data_manager import load_and_preprocess_data, remove_highly_correlated_features, standardize_data, \
    split_data
from dataviz import visualize_correlation
from parameters.VAE import get_params, latent_dim, decoder_fn, data_size, learning_rate, input_dim, opt_update, \
    fast_sample, encoder_fn
from scenarios import create_scenarios

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------
# We notice that among the (few) correlated features (corelation>o.2) the correlation is very high!!
# The observation holds even when we include rows excluded from subset E
# Among the very correlated ones we will keep only one feature(correlation>.8)
# ----------------------------
def inverse_sigmoid(y):
    equation = lambda x: sigmoid(x) - y
    # Use fsolve to find the root of the equation
    x_solution = fsolve(equation, 0.0)[0]
    return x_solution


def build_vae_model(latent_dim, input_dim, hidden_dim):
    # Build VAE model using stax
    encoder_init, encoder_fn = stax.serial(
        Dense(hidden_dim), Relu, Dense(latent_dim * 2))
    decoder_init, decoder_fn = stax.serial(
        Dense(hidden_dim), Relu, Dense(input_dim), Sigmoid)

    return encoder_init, encoder_fn, decoder_init, decoder_fn


@jit
def vae_loss(rand_key, params, x):
    # Encoder
    latent = encoder_fn(params[0:3], x)
    d = latent.shape[-1] // 2
    z_mean, z_log_var = latent[:, :d], latent[:, d:]

    # Sample
    z_sample = fast_sample(rand_key, z_mean, z_log_var)

    # Decoder
    x_rec = decoder_fn(params[3:], z_sample)

    # Ensure x_rec and x have compatible shapes for element-wise operations
    x_rec = x_rec[:, :x.shape[1]]  # Adjust the shape of x_rec if necessary

    # Define reconstruction loss (negative cross-entropy)
    EPSILON = 1e-6
    negative_xent = - jnp.sum(x * jnp.log(x_rec + EPSILON) + (1 - x) * jnp.log(1 - x_rec + EPSILON), axis=-1)

    # Define KL divergence loss
    negative_kl = - 0.5 * jnp.sum(1 + z_log_var - z_mean ** 2 - jnp.exp(z_log_var), axis=-1)

    # Average over the batch, and sum KL / xent
    negative_elbo = jnp.mean(negative_xent) + jnp.mean(negative_kl)

    return negative_elbo


@jit
def update(key, batch, opt_state):
    params = get_params(opt_state)

    # Use the `value_and_grad` function to compute both loss and gradients
    loss, grads = value_and_grad(lambda params, x: vae_loss(key, params, x))(params, batch)

    # Clip gradients to avoid exploding gradients
    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

    # Update the model parameters using the optimizer
    opt_state = opt_update(0, grads, opt_state)

    return opt_state, loss


def train_vae_model(x_train_standard, encoder_init, encoder_fn, decoder_init, decoder_fn, EPOCHS=200, NEURONS_G=100,
                    NEURONS_D=100):
    rand_key = jax.random.PRNGKey(42)
    # Initialize the encoder and decoder
    _, params_enc = encoder_init(rand_key, (-1, input_dim))
    _, params_dec = decoder_init(rand_key, (-1, latent_dim))
    params = params_enc + params_dec

    # Initialize the optimizer
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)

    losses = []

    for epoch in range(EPOCHS):
        # Shuffle the dataset
        rand_key, key = random.split(rand_key)
        permutation = random.permutation(key, data_size)

        for i in range(data_size // 64 - 1):
            batch = x_train_standard[permutation[i * 32:(i + 1) * 32]]
            rand_key, key = random.split(rand_key)
            opt_state, loss = update(key, batch, opt_state)
            losses.append(loss)

    # Return the relevant values
    return opt_state, get_params(opt_state), decoder_fn


def generate_samples(opt_state, params_dec, noise, scenario_label, num_samples=50):
    """
    Generate samples using the trained VAE model

    Parameters
    ----------
    opt_state : tuple
        Optimizer state
    params_dec : tuple
        Decoder parameters
    noise : ndarray with shape (num_samples, latent_dim)
        Random noise vector
    scenario_label : ndarray with shape (num_samples, scenario_dim)
        Scenario label
    num_samples : int, optional
        Number of samples to generate, by default 50

    Returns
    -------
    generated_samples : ndarray
        Generated samples
    """
    # Concatenate noise vector with the scenario label
    input_vector = jnp.concatenate([noise, scenario_label], axis=-1)

    # Use the generator to produce samples
    generated_samples = generator(params_dec, input_vector)

    return generated_samples


def generator(params_dec, noise_and_label):
    """
    Generator model

    Parameters
    ----------
    params_dec : tuple
        Decoder parameters
    noise_and_label: ndarray with shape (n_samples, n_dim=54)
        Input noise and scenario label of the conditional generative model
    """
    # Extract noise and scenario label
    noise, scenario_label = jnp.split(noise_and_label, [-1])

    # Concatenate noise vector with the scenario label
    input_vector = jnp.concatenate([noise, scenario_label], axis=-1)

    # Use the decoder to produce generated samples
    generated_samples = decoder_fn(params_dec, input_vector)

    return generated_samples


# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise, scenario):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (num_samples, latent_dim)
        Random noise vector
    scenario : ndarray with shape (num_samples, scenario_dim)
        Scenario label

    Returns
    -------
    generated_samples : ndarray
        Generated samples
    """
    global features, X_train, X_test, y_train, y_test, opt_state, params_dec

    # Load and preprocess data
    features = load_and_preprocess_data()

    # Visualize correlation among features
    visualize_correlation(features)

    # Remove highly correlated features
    features = remove_highly_correlated_features(features)

    # Standardize data using StandardScaler
    features = standardize_data(features)

    # Call the function to get the scenarios and labels
    scenarios_and_labels = create_scenarios()

    # Display the scenarios and their labels
    for label, scenario in scenarios_and_labels.items():
        print(f"{label}: {scenario}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(features, features['YIELD'])

    # Train the VAE model
    latent_dim = 50
    input_dim = X_train.shape[1]
    hidden_dim = 51

    # Build the VAE model
    encoder_init, encoder_fn, decoder_init, decoder_fn = build_vae_model(latent_dim, input_dim, hidden_dim)

    # Train the VAE model
    opt_state, params_enc, params_dec = train_vae_model(X_train, encoder_init, encoder_fn, decoder_init, decoder_fn)

    # Concatenate noise vector with the scenario label
    input_vector = jnp.concatenate([noise, scenario], axis=-1)

    # Generate samples for the given noise and scenario
    generated_samples = generate_samples(opt_state, params_dec, noise, scenario)

    return generated_samples
