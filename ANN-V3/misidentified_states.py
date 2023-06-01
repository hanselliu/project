import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from generate import generate_states

# Generate the data
states, labels = generate_states(10000)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(states, labels, test_size=0.2, random_state=42)

# Load the trained model
model = load_model('entanglement_model.h5')

# Initialize a list to store the misidentified states
misidentified_states = []

# Iterate over the test set
for state, true_label in zip(X_test, y_test):
    # Reshape the state to match the input shape of the model
    state = state.reshape(1, -1)
    
    # Predict the label for the state
    predicted_label = np.round(model.predict(state)[0][0])

    # If the predicted label doesn't match the true label, save the state
    if predicted_label != true_label:
        misidentified_states.append(state)

# Convert the list of misidentified states into an array
misidentified_states = np.array(misidentified_states)

# Save the misidentified states
np.save('misidentified_states.npy', misidentified_states)
