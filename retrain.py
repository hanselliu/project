import numpy as np
import matplotlib.pyplot as plt
# Load your trained DCGAN
dcgan = load_model('dcgan.h5')

# Load your trained classifier
classifier = load_model('classifier.h5')

# Generate some two-qubit states
states = generate_states(1000)

# Use your classifier to predict labels for these states
predicted_labels = classifier.predict(states)

# Compute the actual labels using the PPT criterion
actual_labels = ppt_criterion(states)

# Find the states where the classifier's prediction was wrong
wrong_states = states[predicted_labels != actual_labels]

# Reshape these states into the input shape expected by the DCGAN
wrong_states_reshaped = wrong_states.reshape((-1, 1, 1, 2))

# Use the DCGAN to generate new states based on the wrong states
new_states = dcgan.predict(wrong_states_reshaped)

# Reshape these states back into the original shape
new_states_reshaped = new_states.reshape((-1, 2))

# Compute labels for these new states using the PPT criterion
new_labels = ppt_criterion(new_states_reshaped)

# Combine the new states and labels with the original states and labels
all_states = np.concatenate([states, new_states_reshaped])
all_labels = np.concatenate([predicted_labels, new_labels])

# Split the combined data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(all_states, all_labels, test_size=0.2, random_state=42)

# Retrain your classifier on the combined data
classifier.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Train the classifier on the combined data
history = classifier.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save the new model
classifier.save('new_classifier.h5')

# Plot the training and validation loss
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(12,6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
