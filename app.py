import streamlit as st
import joblib

# Cargar modelo
modelo = joblib.load("modelo_sentimientos.pkl")

st.title("Clasificador de comentarios (positivo o negativo)")

comentario = st.text_area("Escribe un comentario:")

if st.button("Clasificar"):
    resultado = modelo.predict([comentario])[0]
    st.write(f"**Resultado:** El comentario es **{resultado.upper()}**.")
