import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

print(" Loading MNIST dataset...")
# Load the MNIST dataset (70,000 handwritten digits)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values (0-255) to (0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

print(f" Dataset loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples")

print("\n Building neural network model...")
# Create a neural network
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 output neurons (digits 0-9)
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n Training the model...")
print("This may take 2-3 minutes...\n")

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    verbose=1
)

print("\n Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f" Test Accuracy: {test_accuracy*100:.2f}%")
print(f" Test Loss: {test_loss:.4f}")

# Save the model
os.makedirs("models", exist_ok=True)
model.save("models/digit_recognition_model.h5")
print("\n Model saved to: models/digit_recognition_model.h5")

# Save training history
with open("models/training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("Training history saved!")

print("\n" + "="*50)
print("✨ MODEL TRAINING COMPLETE!")
print("="*50)
print("\nNext step: Run the Streamlit app!")
print("Command: streamlit run app.py") 