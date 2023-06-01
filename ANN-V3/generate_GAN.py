from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU
import numpy as np
from sklearn.model_selection import train_test_split
from generate import generate_states
from qutip import Qobj
from qutip import rand_dm, partial_transpose
def is_ppt(state):
    """Return True if the given state has a positive partial transpose, and False otherwise."""
    pt = partial_transpose(state, [1,0]) # Perform a partial transpose on the second subsystem
    eigenvalues = pt.eigenenergies() # Get the eigenvalues of the partial transpose
    return np.all(eigenvalues >= 0) # Check if all eigenvalues are non-negative

# Define the generator model
def create_generator():
    generator_input = Input(shape=(100,))
    x = Dense(128)(generator_input)
    x = LeakyReLU()(x)
    x = Dense(16)(x)
    generator = Model(generator_input, x)
    return generator

# Define the discriminator model
def create_discriminator():
    discriminator_input = Input(shape=(16,))
    x = Dense(128)(discriminator_input)
    x = LeakyReLU()(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, x)
    return discriminator

# Create the GAN
def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan

# Create the generator, discriminator, and GAN
generator = create_generator()
discriminator = create_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())
gan = create_gan(discriminator, generator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

# Load the misidentified states
misidentified_states = np.load('misidentified_states.npy')
# Define the number of epochs
epochs = 100  # Or any other number based on your needs
# Define the number of batch_size
batch_size = 32  # Or any other number based on your needs
# Define the number of new states to be generatedo
n_new_states = 1000  # Or any other number based on your needs
# Train the GAN on the misidentified states
for epoch in range(epochs):
    # Select a random batch of misidentified states
    idx = np.random.randint(0, misidentified_states.shape[0], batch_size)
    real_states = misidentified_states[idx]
    
    # Generate a batch of new states
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_states = generator.predict(noise)
    
    # Train the discriminator
    real_states = real_states.reshape(batch_size, -1)
    d_loss_real = discriminator.train_on_batch(real_states, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_states, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

# Generate new states with the trained generator
new_states = generator.predict(np.random.normal(0, 1, (n_new_states, 100)))
# Generate the data
states, labels = generate_states(10000)
# Initialize an array for new labels
new_labels = np.empty(new_states.shape[0], dtype=int)

# Assign labels to new states
for i in range(new_states.shape[0]):
    # Convert the state back to a density matrix
    rho = Qobj(new_states[i].reshape((4,4)), dims=[[2,2],[2,2]])
    
    # Assign the label based on the PPT criterion
    new_labels[i] = 0 if is_ppt(rho) else 1
    
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(states, labels, test_size=0.2, random_state=42)
misidentified_labels = [0 if is_ppt(Qobj(state.reshape((4,4)), dims=[[2,2],[2,2]])) else 1 for state in misidentified_states]
# Add the new states and their labels to the training set
# Squeeze the misidentified_states and new_states arrays to remove the extra dimension
misidentified_states_squeezed = np.squeeze(misidentified_states, axis=1)
new_states_squeezed = np.squeeze(new_states, axis=1)
X_train_augmented = np.concatenate((X_train, misidentified_states_squeezed, new_states_squeezed))
y_train_augmented = np.concatenate((y_train, misidentified_labels, new_labels))  # You'll need to determine new_labels
