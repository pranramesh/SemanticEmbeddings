from langchain_community.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key="5vr3ldpOnOHoept93vDSgBPLGfqbZgBfnNeoaATZ")
text = "Test"
query_result = embeddings.embed_query(text)
print(query_result)