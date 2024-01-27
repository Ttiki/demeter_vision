#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DIRECTORY* <!>
#
# See below an example of a generative model
# Z |-> G_\theta(Z)
############################################################################

import numpy as np
import tensorflow as tf
import viz
import evaluation as  eval
import ds_loading as ds

def load_parameters():
    """
    Load your trained model parameters or weights here.
    Modify this function based on your model architecture and how you saved its parameters.
    """
    parameters = np.load("parameters/param.npy")
    return parameters

def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim=4)
        input noise of the generative model
    """
    # Load parameters
    #parameters = load_parameters()

    # Build a simple feedforward neural network (example structure)
    model = tf.keras.Sequential([
        # First layer: Dense layer with 64 units, ReLU activation function, and input shape of (4,)
        tf.keras.layers.Dense(64, activation='relu', input_shape=(noise.shape[1],)),

        # Second layer: Dense layer with 32 units and ReLU activation function
        tf.keras.layers.Dense(32, activation='relu'),

        # Output layer: Dense layer with 4 units (assuming 4 stations), no activation function
        tf.keras.layers.Dense(4)  # Output layer with 4 units (assuming 4 stations)
    ])

    # Set the weights of the model
    #model.set_weights([parameters])

    # Generate samples
    generated_samples = model.predict(noise)

    # Visualize generated data
    viz.visualize_generated_data(generated_samples)

    # Evaluate generated data
    #eval.evaluate_generated_samples(generated_samples, ds.dataset_station40)

    return generated_samples
#%%
