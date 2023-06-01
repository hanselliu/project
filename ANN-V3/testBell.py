import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('entanglement_model.h5')

# Define the Bell state
bell_state_vector = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])

# Calculate the density matrix for the Bell state
bell_density_matrix = np.outer(bell_state_vector, bell_state_vector.conj())

# Flatten the density matrix into a 1D array
bell_state = bell_density_matrix.flatten()

# Reshape the state to match the input shape of the model
bell_state = bell_state.reshape(1, -1)

# Predict the label for the Bell state
label = model.predict(bell_state)

# Print the predicted label
print("Predicted label for the Bell state:", np.round(label[0][0]))
# Interpret the prediction
if np.round(label[0][0]) == 1:
    print("The model predicts that the Bell state is entangled.")
else:
    print("The model predicts that the Bell state is separable.")