"""Formly ipynb converted to py for linting"""

import datetime
import matplotlib.pyplot as plt
import keras

# % load_ext tensorboard
# Clear any logs from previous runs
# %rm -rf ./logs/

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Display 5 images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i], cmap="gray")  # Use grayscale for better visualization
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")  # Remove axis ticks
plt.show()


def create_model():
    """creating the model"""
    return keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


model = create_model()


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(
    x=x_train,
    y=y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print the results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


# Get training history
history = model.fit(x_train, y_train, epochs=5)


# Extract loss and accuracy values from history
train_loss = history.history["loss"]
train_acc = history.history["accuracy"]


# import pickle
# with open('mnist_model.pkl', 'wb') as file:  # Use 'wb' for writing in binary mode
#     pickle.dump(model, file)

model.save("mnist_model.keras")


# Plot training loss
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(train_loss, label="Training Loss")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# Plot training accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_acc, label="Training Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# % tensorboard --logdir logs/fit
# !kill 449952
