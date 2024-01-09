from langchain_community.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key="YOUR_API_KEY")
text = "Test"
query_result = embeddings.embed_query(text)
print(query_result)