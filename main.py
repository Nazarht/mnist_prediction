import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from streamlit_drawable_canvas import st_canvas

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# Streamlit App
st.title("MNIST Digit Recognizer")
st.write("Draw a digit or upload an image of a digit.")

# Drawable Canvas
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert canvas image to grayscale and resize to 28x28
    image = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")
    # image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image) / 255.0  # Normalize pixel values
    image = (image * 2) - 1
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        print(image)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    st.write(f"Predicted Digit: {predicted.item()}")
