import streamlit as st
import torch
import torchvision.transforms as transforms
import pydicom
import numpy as np
from PIL import Image
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model architecture same as training
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 128),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(128, 1),
    torch.nn.Sigmoid()
)

# Load trained weights
model.load_state_dict(torch.load('tb_detection_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to preprocess and predict
def predict_image(image, model):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probability = output.item()
        prediction = 1 if probability >= 0.5 else 0

    label = "TB Positive" if prediction == 1 else "TB Negative"
    return label, probability

# Function to handle DICOM
def read_dicom(file):
    dicom = pydicom.dcmread(file)
    img_array = dicom.pixel_array
    img_array = img_array.astype(np.float32)
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    img_array = (img_array * 255).astype(np.uint8)
    img_array = np.stack((img_array,)*3, axis=-1)
    image = Image.fromarray(img_array)
    return image

# Streamlit UI
st.title("TB Detection from Chest X-ray")
st.write("Upload a Chest X-ray image (.jpg or .dcm) to predict Tuberculosis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "dcm"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.dcm'):
        image = read_dicom(uploaded_file)
    else:
        image = Image.open(uploaded_file).convert('RGB')

    st.image(image, caption='Uploaded X-ray.', use_column_width=True)

    st.write("")
    st.write("Predicting...")
    label, probability = predict_image(image, model)

    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability:** {probability:.2f}")
