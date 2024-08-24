import streamlit as st

st.header("Battle of the Neural Networks")

st.divider()

if st.button("Commence Battle"):
    st.image("Pictures/battle.png")

st.subheader("About the classic MNIST dataset...")

st.write(
    """The MNIST dataset is a collection of 70,000 handwritten digits (0-9) that are used to train and test machine 
         learning algorithms. The dataset is divided into 60,000 training images and 10,000 test images. Each image is a 28x28
         pixel grayscale image."""
)

st.divider()

st.subheader("The MNIST Dataset at a glance")
st.image("""Pictures/MNIST_full.png""")
st.caption(
    "This sample shows the pattern of what type of data is in our dataset. Please note it is already cleaned and gray scale. This image is from: https://towardsdatascience.com/solve-the-mnist-image-classification-problem-9a2865bcf52a"
)

st.divider()

st.subheader("The partial MNIST Dataset")
st.image("Pictures/MNIST_part.png")
st.caption(
    "This is a picture of a smaller sample of the dataset so it is easier to see from: https://datasets.activeloop.ai/docs/ml/datasets/mnist/"
)

st.divider()

st.subheader("MNIST Header")
st.image("Pictures/Sample_MNIST.png")
st.caption("This is a picture of the first 5 images in the dataset we loaded to use.")

col1, col2 = st.columns(2)

with col1:
    st.header("Tensorflow")
    st.caption("The length of code for the Tensorflow Neural Network.")
    st.code(
        """
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

""",
        language="Python",
    )

with col2:
    st.header("Pytorch")
    st.caption("The length of code for the Pytorch Neural Network.")
    st.code(
        """
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # Dense layer equivalent
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)  # Dense layer equivalent

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNISTModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Equivalent to sparse_categorical_crossentropy
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Test Accuracy: {:.2f}%'.format(100 * correct / total))

""",
        language="Python",
    )

st.markdown(
    """
### Tensorflow vs. Pytorch
            
- **TensorFlow**: TensorFlow is often favored for its concise syntax and ease of implementation. It provides a more streamlined approach, allowing developers to quickly build and deploy models with less code. This makes it an excellent choice for projects where rapid development and deployment are key.

- **PyTorch**: On the other hand, PyTorch offers more flexibility and customizability. While it may require more code to implement certain features, PyTorch's dynamic nature allows for finer control over model architecture and behavior. This makes it a preferred choice for researchers and developers who need to experiment with novel ideas and custom solutions.

- **Pros and Cons**: TensorFlow's simplicity comes at the cost of reduced flexibility, while PyTorch's extensive customizability can lead to longer development times. The choice between the two often depends on the specific needs of the project.

"""
)
