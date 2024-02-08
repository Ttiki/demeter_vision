##############################################
# Goal : return training parameters
# For that we will have to define our loss function, specify number of epochs, batch size, 
#latent dimension although this could be included in model.py as implied in the indication provided
#-in the model.py provided code, and lastly the numnber of neurons for the generator and the number of neurons
#-for the discriminator
################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.model_selection import train_test_split
from scipy.optimize import fsolve
from scipy.special import expit
from ot.sliced import sliced_wasserstein_distance
from jax import jit, grad, numpy as jnp, random
from jax.example_libraries import stax, optimizers
from jax.example_libraries.stax import Dense, Relu, Sigmoid, Elu, Tanh, LeakyRelu, Softplus, Softmax












correlation_matrix = np.corrcoef(filtered_features, rowvar=False)

# correlation among correlated features after dropping very correlated features
correlation_graph=[]
for i in range(2,10) :
    correlation_graph.append((np.sum(~np.array([(correlation_matrix <= i/10)]))))
bars=plt.bar(x=np.array(range(2,10))/10,height=correlation_graph,width=.05)
for bar, label in zip(bars, correlation_graph):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{label:.2f}', ha='center', va='bottom')

plt.title('Bar Plot with Pods and Labels')
plt.xlabel('X-axis')
plt.ylabel('Correlation Values')
plt.show()


scaler = StandardScaler()
scaled_filtered_features = scaler.fit_transform(filtered_features)

#VAE JAX
#---------------------------
#Building the encoder :




rand_key = random.PRNGKey(5)
k1, k2 = random.split(rand_key)

#you can update the rand_key
rand_key = k1

#use k2
random.normal(k2, shape=(2,3))
# Assuming filtered_features and targets are your original data
x_train, x_test= train_test_split(scaled_filtered_features, test_size=0.2, random_state=42)
plt.plot(x_test)
plt.show()
# Reshape the arrays
x_train_standard = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test_standard = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))




input_dim = x_train_standard.shape[1]
hidden_dim = 51
latent_dim = 50

encoder_init, encoder_fn = stax.serial(
    Dense(hidden_dim), Relu, Dense(latent_dim * 2))

#initialize the parameters
rand_key, key = random.split(rand_key)
out_shape, params_enc = encoder_init(rand_key, (-1, input_dim))

print("Parameters: (W,b) of first Dense, Relu, (W,b) of second Dense: ")
print([[p.shape for p in param] for param in params_enc])
      
#Reparanetrization trick
def sample(rand_key, z_mean, z_log_var):
    epsilon = random.normal(rand_key, shape=z_mean.shape)
    return z_mean + jnp.exp(z_log_var / 2) * epsilon

fast_sample = jit(sample)

#Building the Decoder:

# Reshape the arrays

# Use the modified decoder initialization and function
decoder_init, decoder_fn = stax.serial(
    Dense(hidden_dim), Relu, Dense(input_dim), Sigmoid)
print(decoder_fn)

#initialize the parameterdecoders
rand_key, key = random.split(rand_key)
out_shape, params_dec = decoder_init(rand_key, (-1, latent_dim))

params = params_enc + [params_dec[0], params_dec[1]]

EPSILON = 1e-6
negative_xent = jit(lambda x, y: - jnp.sum(y * jnp.log(x + EPSILON) +
                                           (1 - y) * jnp.log(1 - x + EPSILON), axis=-1))

negative_kl = jit(lambda z_mean, z_log_var: - 0.5 *
                  jnp.sum(1 + z_log_var - z_mean ** 2 - jnp.exp(z_log_var), axis=-1))

#VAE
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
print(len(params[0:3]),len(params[3:]))


from functools import partial
rand_key, key = random.split(rand_key)
vae_loss_rand = partial(vae_loss, key) # this creates a function val_loss without the rand_key argument

# Training parameters:
# -------------------
EPOCHS =  200 # one forward/backward pass in all training samples

#LATENT_DIM = 10
NEURONS_G = 100
NEURONS_D = 100
# -------------------

# You may run this cell to reinit parameters if needed
_, params_enc = encoder_init(rand_key, (-1, input_dim))
_, params_dec = decoder_init(rand_key, (-1, latent_dim))
params = params_enc + params_dec
print(len(params_enc),len(params_dec),"hereee")

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

for epochs in range(EPOCHS):
    # Shuffle the dataset
    rand_key, key = random.split(rand_key)
    permutation = random.permutation(key, data_size)
    for i in range(data_size // 64 - 1):
        
        batch = x_train_standard[permutation[i * 32:(i+1)*32]]
        rand_key, key = random.split(rand_key)
        opt_state, loss = update(key, batch, opt_state)
        losses.append(loss)


import matplotlib.pyplot as plt
plt.plot(losses);
plt.show()


#Generate try
rand_key, key = random.split(key)
params = get_params(opt_state)
params_dec = params[3:]
z = random.normal(key, shape=(1,latent_dim))
jit_decoder_fn = jit(decoder_fn)  # Apply jit to decoder_fn
generated = jit_decoder_fn(params_dec, z)  # Call the jitted function
print(generated[0,-4:])
print(x_train_standard[0,-4:])

generated_yields = generated
training_yields = x_train_standard

print(generated_yields[0,-4:])
print(training_yields[0,-4:])
for i in range(50):    
    if features.iloc[i,-4] ==9.62 :
         print(features.iloc[i,-4:])



#Generate 
rand_key, key = random.split(key)
params = get_params(opt_state)
params_dec = params[3:]
z = random.normal(key, shape=(x_test_standard.shape[0],latent_dim))
z=noise_elhcs[:x_test_standard.shape[0],:latent_dim]
jit_decoder_fn = jit(decoder_fn)  # Apply jit to decoder_fn
generated = jit_decoder_fn(params_dec, z)  # Call the jitted function

generated_yields = generated
testing_data=x_test_standard
testing_data=1 / (1 + jnp.exp(-testing_data))

testing_data=np.array(testing_data)
generated_yields_numpy = np.array(generated_yields)
print(sliced_wasserstein_distance(generated_yields_numpy,testing_data))

unscaled_testing_data=scaler.inverse_transform(testing_data)
unscaled_generated_yields=scaler.inverse_transform(generated_yields)
print(sliced_wasserstein_distance(unscaled_generated_yields,unscaled_testing_data))
unscaled_testing_data= expit(unscaled_testing_data)
unscaled_generated_yields= expit(unscaled_generated_yields)
print(sliced_wasserstein_distance(unscaled_generated_yields,unscaled_testing_data))
# Create subplots
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(10, 8))

# Plot on the first subplot
ax1.plot(generated_yields[:, -4:])
ax1.set_title('Generated Yields')

# Plot on the second subplot

ax2.plot(testing_data[:, -4:])
ax2.set_title('Testing Data')
# Plot on the first subplot
ax3.plot(generated_yields[:, :-4])
ax3.set_title('Generated Yields')

# Plot on the second subplot
ax4.plot(testing_data[:, :-4])
ax4.set_title('Testing Data')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()

print("Any NaN values in x_train_standard:", np.any(np.isnan(x_train_standard)))
print("Any infinite values in x_train_standard:", np.any(np.isinf(x_train_standard)))
print("Any NaN values in x_train_standard:", np.any(np.isnan(x_train)))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot on the first subplot
ax1.plot(unscaled_generated_yields[:, -4:])
ax1.set_title('Generated Yields')

# Plot on the second subplot

ax2.plot(unscaled_testing_data[:, -4:])
ax2.set_title('Testing Data')
plt.show()
