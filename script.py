from langchain_community.embeddings import CohereEmbeddings
import streamlit as st
import pandas as pd 
import torch

embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key="5vr3ldpOnOHoept93vDSgBPLGfqbZgBfnNeoaATZ")


st.title("Latent Space Exploration")

container1 = st.container(border=True)
container1.subheader("Original Text")
text = container1.text_input('Text', label_visibility="hidden")
container1.write(f"{text}")

query_result = embeddings.embed_query(text)

container2 = st.container(border=True)
container2.subheader("Latent Space Embedding")
container2.write(f"{query_result}")

container3 = st.container()
number = container3.slider("Negativity", 0, 100)

container4 = st.container(border=True)
container4.subheader("Reconstruction")
container4.write(f"{[x + float(number) for x in query_result]}")

container5 = st.container(border=True)
container5.subheader("Visualization")
#project both embeddings into 2d subspace using PCA and display on app and show before and after
