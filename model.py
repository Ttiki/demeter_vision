# model.py
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import ds_loading as ds
import evaluation
import viz

def build_generator(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(20, activation='linear'))
    return model

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(generator.input_shape[1],))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan_model = Model(inputs=gan_input, outputs=gan_output)
    gan_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))
    return gan_model

def train_gan(generator, discriminator, gan_model, noise, epsilon_samples, real_labels):
    batch_size = len(noise)
    generated_samples = generator.predict(noise)
    real_labels = np.ones((epsilon_samples.shape[0], 1))
    discriminator_loss_real = discriminator.train_on_batch(epsilon_samples, real_labels)
    discriminator_loss_fake = discriminator.train_on_batch(generated_samples, np.zeros((batch_size, 1)))
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
    gan_loss = gan_model.train_on_batch(noise, real_labels)
    return gan_loss

def train_gan_with_datasets(generator, discriminator, gan_model, input_dim, num_epochs, batch_size):
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    columns_to_keep = ['W_13', 'W_14', 'W_15']
    dataset1 = ds.dataset_station40[columns_to_keep]
    dataset2 = ds.dataset_station49[columns_to_keep]
    dataset3 = ds.dataset_station63[columns_to_keep]
    dataset4 = ds.dataset_station80[columns_to_keep]
    real_samples = np.concatenate((dataset1, dataset2, dataset3, dataset4), axis=0)
    np.random.shuffle(real_samples)
    Q = np.array([3.3241, 5.1292, 6.4897, 7.1301])
    epsilon_condition = (
            (dataset1.sum(axis=1) <= Q[0]) &
            (dataset2.sum(axis=1) <= Q[1]) &
            (dataset3.sum(axis=1) <= Q[2]) &
            (dataset4.sum(axis=1) <= Q[3])
    )
    epsilon_samples = np.concatenate((dataset1[epsilon_condition], dataset2[epsilon_condition], dataset3[epsilon_condition], dataset4[epsilon_condition]), axis=0)
    real_labels = np.ones((epsilon_samples.shape[0], 1))
    for epoch in range(num_epochs):
        gan_loss = train_gan(generator, discriminator, gan_model, noise, epsilon_samples, real_labels)
        print(f"Epoch: {epoch}, GAN Loss: {gan_loss}")
    generated_samples = generative_model(noise)
    return generated_samples

def generative_model(noise):
    input_dim = 12
    num_epochs = 1000
    batch_size = 64
    generator = build_generator(input_dim)
    discriminator = build_discriminator(input_dim)
    gan_model = build_gan(generator, discriminator)
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    generated_samples = train_gan_with_datasets(generator, discriminator, gan_model, input_dim, num_epochs, batch_size)
    viz.visualize_generated_data(generated_samples)
    return generated_samples