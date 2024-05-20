import streamlit as st

st.header("Pytorch")

st.divider()

st.subheader("The full MNIST Dataset")
st.image("Pictures\MNIST_full.png")
st.caption("This is the full dataset of images from: https://towardsdatascience.com/solve-the-mnist-image-classification-problem-9a2865bcf52a")

st.divider()

st.subheader("The partial MNIST Dataset")
st.image("Pictures\MNIST_part.png")
st.caption("This is a picture of part of the dataset so it is easier to see from: https://datasets.activeloop.ai/docs/ml/datasets/mnist/")

st.divider()

st.subheader("Straight from my code")
st.image("Pictures\code_mnist.png")
st.caption("This is a picture of the code I wrote to display one index (image) at a time.")

st.divider()

st.subheader("Loss Accuracy Graphs")
st.image("Pictures/NN-loss-acc.png")
st.caption("On the left, we see the loss per epoch. \nOn the right, we see the accuaracy increase with each epoch.")

st.divider()

st.subheader("About My Model")
st.write("My model Accuracy: 98.03%")

st.write("""
The input to this network is a 28x28 pixel grayscale image, which is flattened into a 1D vector of 784 elements (28 * 28 = 784). This flattening is done in the forward method using x.view(-1, 28*28).
         
We then used a hidden layer (nn.Linear) with 512 neurons. Each of the 784 input values is connected to each of the 512 neurons in this layer.The output of this layer is then passed through a ReLU (Rectified Linear Unit) activation function. ReLU graphs are just linear with a bend.
         
Lastly we have the output layer which takes in the 512 neurons and outputs 1 of 10 (to represent the catogories of number [0-9]. This is the classifcation step.)
""")

st.image("Pictures/Relu.png")
st.caption("The Relu function for reference.")

st.divider()

