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
    prompt=f"Please make the following block of text sound {option}: {text}",
    num_generations=5
    )

    #generate embeddings for original text and generated samples
    original_embedding = embeddings.embed_query(text)
    new_embeddings = [embeddings.embed_query(element) for element in response]

    ##### Container 2 #####
    container2 = st.container(border=True)
    container2.header("Modified Text")
    container2.write(f"{response[0]}")

    ##### Container 3 #####
    container3 = st.container(border=True)
    container3.header("Visualization")
    container3.caption(f"Synthetically generates more samples and visualizes spatial relatedness via PCA")

    #calculates cosin similarity
    e1 = np.array(original_embedding)
    e2 = np.array(new_embeddings[0])
    # Reshape the arrays to match the expected input shape of cosine_similarity
    e1_embedding = e1.reshape(1, -1)
    e2_embedding = e2.reshape(1, -1)
    # Calculate cosine similarity
    similarity = cosine_similarity(e1_embedding, e2_embedding)[0][0]
    container3.write(f"Cosine similarity score: {similarity}")

    # creates dataframe for easy visualization
    d = {
        "Text": [text] + [element for element in response], 
    "Embedding": [original_embedding] + new_embeddings
    }
    df = pd.DataFrame(data=d)

    #creates visualization for PCA
    container3.subheader(f"PCA")
    #add option to visualize in 2d (pc2), 3d (pc3), or with heatmap (pc5)
    visualize = Visualizations(df)
    principal_components = visualize.principal_component_analysis(2)
    df_pc2 = pd.concat([df, pd.DataFrame(principal_components)], axis=1)
    container3.dataframe(df_pc2)
    df_pc2.columns = df_pc2.columns.astype(str)
    container3.scatter_chart(data=df_pc2, x='0', y='1')

    #creates visualization for t-SNE
    # container3.subheader(f"t-SNE")
    # tsne_d = visualize.stochastic_neighbors(2)
    # df_tsne = pd.concat([df, pd.DataFrame(tsne_d)], axis=1)
    # container3.dataframe(df_tsne)
    # df_tsne.columns = df_tsne.columns.astype(str)
    # container3.scatter_chart(data=df_tsne, x='0', y='1')


    # # Creating 3d plot
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
    # ax.scatter3D(df_pc3['0'], df_pc3['1'], df_pc3['2'])
    # container3.pyplot(fig=fig)
