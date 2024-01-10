from langchain_community.embeddings import CohereEmbeddings
import streamlit as st
import pandas as pd 
import torch
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from model import Visualizations
from matplotlib import pyplot as plt

st.title("Latent Space Exploration")

##### Container 1 #####
container1 = st.container(border=True)
container1.header("Original Text")
text = container1.text_area('Text', label_visibility="hidden")
# container1.write(f"{text}")
#allows user to choose whether to make text more negative or more positive
option = container1.selectbox(
    'How would you like to modify your text?',
    ('very negative', 'negative', 'positive', 'very positive'), index=None, placeholder='Choose an option...')


#initializing embedding model
embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key="5vr3ldpOnOHoept93vDSgBPLGfqbZgBfnNeoaATZ")

#initializing language model
co = cohere.Client("5vr3ldpOnOHoept93vDSgBPLGfqbZgBfnNeoaATZ")

if text and option:

    response = co.generate(
    prompt=f"Please make the following block of text sound {option}: {text}"
    )

    original_embedding = embeddings.embed_query(text)
    new_embedding = embeddings.embed_query(response[0])

    ##### Container 2 #####
    container2 = st.container(border=True)
    container2.header("Modified Text")
    container2.write(f"{response[0]}")

    ##### Container 3 #####
    container3 = st.container(border=True)
    container3.header("Visualization")

    #calculates cosin similarity
    e1 = np.array(original_embedding)
    e2 = np.array(new_embedding)
    # Reshape the arrays to match the expected input shape of cosine_similarity
    e1_embedding = e1.reshape(1, -1)
    e2_embedding = e2.reshape(1, -1)
    # Calculate cosine similarity
    similarity = cosine_similarity(e1_embedding, e2_embedding)[0][0]
    container3.write(f"Cosine similarity score: {similarity}")

    #creates visualization for PCA
    visualize = Visualizations(original_embedding, new_embedding)
    principal_components = visualize.principal_component_analysis()

    for i in principal_components:
        fig = plt.scatter(i[:, 0], i[:, 1], label=(f'Original Embedding' if i == 0 else f'Modified Embedding'))
        container3.pyplot(fig)