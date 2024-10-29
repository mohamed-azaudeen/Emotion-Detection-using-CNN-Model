import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
from io import BytesIO

class EDM30LayerCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EDM30LayerCNN, self).__init__()

        layers = []
        in_channels = 1
        out_channels = 16

        for i in range(15):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())

            if i % 3 == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels
            if out_channels < 256:
                out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


model = EDM30LayerCNN(num_classes=len(classes))


state_dict = torch.load('edm30_model.pth')
model.load_state_dict(state_dict, strict=False)


image_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    ToTensor(),
    transforms.Normalize(mean=[0.5194], std=[0.2067])
])


def predict_model(model, image_transforms, image, classes):
    model.eval()  
    transformed_image = image_transforms(image).float()
    transformed_image = transformed_image.unsqueeze(0)  

    with torch.no_grad():
        output = model(transformed_image)
        _, predicted = torch.max(output.data, 1)

    predicted_class = classes[predicted.item()]
    return predicted_class


st.title("Emotion Detection with EDM30LayerCNN")


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(BytesIO(uploaded_image.read()))
    predicted_class = predict_model(model, image_transforms, image, classes)
    st.write(f"**Predicted Emotion Class:** {predicted_class}")
    st.image(image.convert("L").resize((224, 224)), caption=f"Predicted Class: {predicted_class}", use_column_width=True)
