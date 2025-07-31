import streamlit as st
import numpy as np
import cv2
from inference import get_model
import supervision as sv
from PIL import Image

st.set_page_config(page_title="Detecção de Anomalias em Hemácias", layout="centered")

st.title("🧪 Detecção de Anomalias em Hemácias")
st.write("Envie uma imagem para análise usando o modelo personalizado.")

# Upload de imagem
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Carrega a imagem do upload
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Exibe imagem original
    st.subheader("Imagem Original")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Carrega o modelo
    with st.spinner("Executando detecção..."):
        model = get_model(model_id="erythrocyte_abnormalities-5/2",
        api_key="JVZ9gni4qoAOwVCXjfG0")

        # Executa inferência
        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)

        # Anotadores
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Anota imagem
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Exibe imagem anotada
    st.subheader("Resultado da Detecção")
    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Resultados brutos
    with st.expander("🔍 Ver Detalhes da Inferência"):
        st.write("Tipo de `results`:", type(results))
        st.json(results.model_dump())
