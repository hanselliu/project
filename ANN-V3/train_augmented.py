import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from generate import generate_states
from qutip import Qobj
from qutip import rand_dm, partial_transpose

# Function to check if a state is PPT
def is_ppt(state):
    """Return True if the given state has a positive partial transpose, and False otherwise."""
    pt = partial_transpose(state, [1,0]) # Perform a partial transpose on the second subsystem
    eigenvalues = pt.eigenenergies() # Get the eigenvalues of the partial transpose
    return np.all(eigenvalues >= 0) # Check if all eigenvalues are non-negative

# Generate the data
states, labels = generate_states(10000)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(states, labels, test_size=0.2, random_state=42)

# Load the misidentified states
misidentified_states = np.load('misidentified_states.npy')

# Remove the extra dimension from misidentified_states
misidentified_states = np.squeeze(misidentified_states)

# Now you can concatenate misidentified_states with X_train
X_train_augmented = np.concatenate((X_train, misidentified_states))


# Get the correct labels for the misidentified states
misidentified_labels = [0 if is_ppt(Qobj(state.reshape((4,4)), dims=[[2,2],[2,2]])) else 1 for state in misidentified_states]

# Add the misidentified states and their labels to the training set
X_train_augmented = np.concatenate((X_train, misidentified_states))
y_train_augmented = np.concatenate((y_train, misidentified_labels))

# Define the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(16,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=100)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(X_train_augmented, y_train_augmented, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

# Save the model
model.save('entanglement_model_augmented.h5')
# Save the training history
np.save('train_history_augmented.npy', history.history)