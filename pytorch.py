
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision


# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# Import the data
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transforms.ToTensor()
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)


# Use this block to show images for the data and familiarize myself with the data
# Load MNIST without transforming to tensors
train_dataset_pic = datasets.MNIST(root="./data", train=True, download=True)

# Display 5 images
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    image, label = train_dataset_pic[i]  # Get image and label
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.show()


# Transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# make the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# init the nn class with an init function and a forward pass
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(
            28 * 28, 512
        )  # Input: 28x28 image, Hidden layer: 512 neurons
        self.fc2 = nn.Linear(512, 10)  # Output: 10 classes (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))  # what should this be if F is wrong?
        x = self.fc2(x)
        return x


model = torchvision.models.resnet50(False)
# Have ResNet model take in grayscale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
writer.add_image("images", grid, 0)
writer.add_graph(model, images)
writer.close()


# establish the model, loss function and the optim (brett used Adam so lets go with that one. the list is insanely long to choose from)
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Initialize lists to store metrics - gonna use this for graphing my learning
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


# Training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        _, predicted = torch.max(
            output.data, 1
        )  # the underscore is our throwaway variable which is apprently a common naming convention -COOL!
        train_accuracy = (predicted == target).sum().item() / len(train_dataset) * 100
        train_accuracies.append(train_accuracy)

    # the song says "print out whats happenin'"
    print(f"Epoch {epoch}: Loss: {loss.item():.4f}")


model.eval()  # Set the model to evaluation mode
test_loss = 0
correct = 0
with torch.no_grad():  # No gradients needed during evaluation
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)  # for display
        accuracy = 100.0 * correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)  # for display


print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")


train_losses = train_losses[:10]
train_accuracies = train_accuracies[:10]
test_losses = test_losses[:10]
test_accuracies = test_accuracies[:10]
print(train_losses)
print(train_accuracies)
print(test_losses)
print(test_accuracies)


epochs = range(10)
# Plotting
plt.figure(figsize=(12, 6))

# Loss Chart
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", marker="o")
# plt.plot(epochs, test_losses, label='Epoch', marker='x')
plt.title("Learning Curve - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(axis="y")

# Accuracy Chart
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy", marker="o")
# plt.plot(epochs, test_accuracies, label='Epoch', marker='x')
plt.title("Learning Curve - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(axis="y")

plt.tight_layout()
plt.show()


# Create input values
x = np.linspace(-10, 10, 400)  # Range from -10 to 10 with 400 points for smoothness

# Calculate ReLU output
y = np.maximum(0, x)

# Plotting
plt.figure(figsize=(8, 5))  # Optional: Set figure size
plt.plot(x, y, color="blue", linewidth=2)
plt.title("ReLU Activation Function")
plt.xlabel("Input (x)")
plt.ylabel("Output (ReLU(x))")
plt.axhline(0, color="gray", linestyle="dashed")  # Dashed line at y=0
plt.axvline(0, color="gray", linestyle="dashed")  # Dashed line at x=0
plt.grid(alpha=0.4)  # Add a light grid
plt.show()


%tensorboard --logdir logs/fit


# Run tensorboard with
# tensorboard --logdir=runs








