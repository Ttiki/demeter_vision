##############################################
# Goal : return training parameters
# For that we will have to define our loss function, specify number of epochs, batch size,
# latent dimension although this could be included in model.py as implied in the indication provided
# -in the model.py provided code, and lastly the numnber of neurons for the generator and the number of neurons
# -for the discriminator
################################################

import jax
from jax import jit, numpy as jnp, random
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Dense, Relu, Sigmoid

# VAE JAX
# ---------------------------
# Building the encoder :


rand_key = random.PRNGKey(5)
k1, k2 = random.split(rand_key)

# you can update the rand_key
rand_key = k1

# use k2
random.normal(k2, shape=(2, 3))
hidden_dim = 51
latent_dim = 50

print("Parameters: (W,b) of first Dense, Relu, (W,b) of second Dense: ")
print([[p.shape for p in param] for param in params_enc])


# Reparanetrization trick
def sample(rand_key, z_mean, z_log_var):
    epsilon = random.normal(rand_key, shape=z_mean.shape)
    return z_mean + jnp.exp(z_log_var / 2) * epsilon


fast_sample = jit(sample)

# Building the Decoder:

# Reshape the arrays

# Use the modified decoder initialization and function
decoder_init, decoder_fn = stax.serial(
    Dense(hidden_dim), Relu, Dense(input_dim), Sigmoid)
print(decoder_fn)

# initialize the parameterdecoders
rand_key, key = random.split(rand_key)
out_shape, params_dec = decoder_init(rand_key, (-1, latent_dim))

params = params_enc + [params_dec[0], params_dec[1]]

EPSILON = 1e-6
negative_xent = jit(lambda x, y: - jnp.sum(y * jnp.log(x + EPSILON) +
                                           (1 - y) * jnp.log(1 - x + EPSILON), axis=-1))

negative_kl = jit(lambda z_mean, z_log_var: - 0.5 *
                                            jnp.sum(1 + z_log_var - z_mean ** 2 - jnp.exp(z_log_var), axis=-1))


# VAE
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
    assert x_rec.shape == x.shape, f"Shapes mismatch: x_rec ({x_rec.shape}), x ({x.shape})"
    xent_loss = negative_xent(x_rec, x)
    kl_loss = negative_kl(z_mean, z_log_var)

    # average over the batch, and sum kl / xent
    negative_elbo = jnp.mean(xent_loss) + jnp.mean(kl_loss)
    return negative_elbo


print("hereee")
print(len(params[0:3]), len(params[3:]))

from functools import partial

rand_key, key = random.split(rand_key)
vae_loss_rand = partial(vae_loss, key)  # this creates a function val_loss without the rand_key argument

# Training parameters:
# -------------------
EPOCHS = 200  # one forward/backward pass in all training samples

# LATENT_DIM = 10
NEURONS_G = 100
NEURONS_D = 100
# -------------------

# You may run this cell to reinit parameters if needed
_, params_enc = encoder_init(rand_key, (-1, input_dim))
_, params_dec = decoder_init(rand_key, (-1, latent_dim))
params = params_enc + params_dec
print(len(params_enc), len(params_dec), "hereee")

print(x_train_standard.shape)
data_size = x_train_standard.shape[0]
batch_size = 64
learning_rate = 0.00001  # Adjust learning rate

opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)
value_and_grad_fun = jit(jax.value_and_grad(lambda params, x: vae_loss(key, params, x)))
print(value_and_grad_fun(params, x_train_standard)[0])

losses = []


@jit
def update(key, batch, opt_state):
    params = get_params(opt_state)
    value_and_grad_fun = jit(jax.value_and_grad(lambda params, x: vae_loss(key, params, x)))
    loss, grads = value_and_grad_fun(params, batch)
    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    opt_state = opt_update(0, grads, opt_state)
    print(value_and_grad_fun(params, x_train_standard)[0])
    return opt_state, loss
