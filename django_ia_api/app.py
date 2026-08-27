import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
# from utils import load_and_prep_image, classes_and_models, update_logger, predict_json

# Cargar el model
modelo = torch.load("model.pth")
modelo.eval()

# Transformaciones
transformaciones = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


### Streamlit code (works as a straigtht-forward script) ###
st.title("Detector de perros y gatos 🍔📸")
st.header("Identificar entre ambos XD!")

archivo = st.file_uploader('Sube una imagen de un perro o un gato', type=['png', 'jpg', 'jpeg'])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, caption='Imagen subida', use_column_width=True)

    # Preprocesar la imagen
    imagen = transformaciones(imagen).unsqueeze(0)

    # Hacer la predicción
    with torch.no_grad():
        salida = modelo(imagen)
        probabilidad = torch.nn.functional.softmax(salida, dim=1)[0][1].item()

    # Mostrar la predicción
    if probabilidad > 0.5:
        st.write(f"Es un perro con una probabilidad de {round(probabilidad * 100)}%")
    else:
        st.write(f"Es un gato con una probabilidad de {round((1 - probabilidad)*100)}%")