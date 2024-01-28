import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import ds_loading as ds

def build_generator(input_dim):
    """
    Build the generator model.

    Parameters
    ----------
    input_dim : int
        Dimension of the input noise vector.

    Returns
    -------
    model : keras.models.Sequential
        Generator model.
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(20, activation='linear'))  # Adjust the output size based on your requirements
    return model

def build_discriminator(input_dim):
    """
    Build the discriminator model.

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector.

    Returns
    -------
    model : keras.models.Sequential
        Discriminator model.
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def build_gan(generator, discriminator):
    """
    Build the GAN model.

    Parameters
    ----------
    generator : keras.models.Sequential
        Generator model.
    discriminator : keras.models.Sequential
        Discriminator model.

    Returns
    -------
    model : keras.models.Sequential
        GAN model.
    """
    discriminator.trainable = False  # Set the discriminator to non-trainable during GAN training
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def train_gan(generator, discriminator, gan_model, noise, real_samples):
    """
    Train the GAN model.

    Parameters
    ----------
    generator : keras.models.Sequential
        Generator model.
    discriminator : keras.models.Sequential
        Discriminator model.
    gan_model : keras.models.Sequential
        GAN model.
    noise : numpy.ndarray
        Input noise for generator training.
    real_samples : numpy.ndarray
        Real samples for discriminator training.

    Returns
    -------
    gan_loss : float
        Loss of the GAN model.
    """
    batch_size = len(noise)

    # Generate fake samples using the generator
    generated_samples = generator.predict(noise)

    # Labels for real and fake samples
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    # Train the discriminator on real samples
    discriminator_loss_real = discriminator.train_on_batch(real_samples[:batch_size], real_labels)

    # Train the discriminator on generated (fake) samples
    discriminator_loss_fake = discriminator.train_on_batch(generated_samples, fake_labels)

    # Calculate the average discriminator loss
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)

    # Train the GAN (generator) to fool the discriminator
    gan_loss = gan_model.train_on_batch(noise, real_labels)

    return gan_loss

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generate samples using the trained generator.

    Parameters
    ----------
    generator : keras.models.Sequential
        Trained generator model.
    noise : numpy.ndarray
        Input noise for generating samples.

    Returns
    -------
    generated_samples : numpy.ndarray
        Generated samples.
    """
    # Generating random noise for the generator
    # Defining parameters
    input_dim = 20  # Dimension of the input noise vector
    num_epochs = 1000  # Number of epochs for training
    batch_size = 64  # Batch size for training

    # Defining models
    generator = build_generator(input_dim)
    discriminator = build_discriminator(input_dim)
    gan_model = build_gan(generator, discriminator)

    # Train GAN using real datasets
    generated_samples = train_gan_with_datasets(generator, discriminator, gan_model, input_dim, num_epochs, batch_size)

    return generated_samples

def train_gan_with_datasets(generator, discriminator, gan_model, input_dim, num_epochs, batch_size):
    """
    Train GAN using real datasets.

    Parameters
    ----------
    generator : keras.models.Sequential
        Generator model.
    discriminator : keras.models.Sequential
        Discriminator model.
    gan_model : keras.models.Sequential
        GAN model.
    input_dim : int
        Dimension of the input noise vector.
    num_epochs : int
        Number of epochs for training.
    batch_size : int
        Batch size for training.

    Returns
    -------
    None
    """
    # Generating random noise for the generator
    noise = np.random.normal(0, 1, (batch_size, input_dim))

    # Name of cols
    columns_to_keep = ['YEAR',
                       'W_1', 'W_2', 'W_3', 'W_4', 'W_5', 'W_6', 'W_7', 'W_8', 'W_9',
                       'W_10', 'W_11', 'W_12', 'W_13', 'W_14', 'W_15', 'W_16', 'W_17', 'W_18',
                       'YIELD']

    # Keep only the necessary columns
    dataset1 = ds.dataset_station40[columns_to_keep]
    dataset2 = ds.dataset_station49[columns_to_keep]
    dataset3 = ds.dataset_station63[columns_to_keep]
    dataset4 = ds.dataset_station80[columns_to_keep]

    # Concatenate along the appropriate axis (axis=0 if datasets are vertically stacked, axis=1 if horizontally stacked)
    real_samples = np.concatenate((dataset1, dataset2, dataset3, dataset4), axis=0)

    # Shuffle the data to ensure randomness in each batch
    np.random.shuffle(real_samples)

    # Train your GAN
    for epoch in range(num_epochs):
        # Train GAN
        gan_loss = train_gan(generator, discriminator, gan_model, noise, real_samples)
        print(f"Epoch: {epoch}, GAN Loss: {gan_loss}")

    # Generate samples using the trained generator
    generated_samples = generator.predict(noise)

    return generated_samples
