import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from generate import generate_states

# Generate the data
states, labels = generate_states(10000)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(states, labels, test_size=0.2, random_state=42)

# Save the training and test sets
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

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

# Load the training and test sets
X_train = np.load('X_train_augmented.npy')
y_train = np.load('y_train_augmented.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

# Save the model
model.save('entanglement_model.h5')

# Save the training history
np.save('train_history.npy', history.history)
