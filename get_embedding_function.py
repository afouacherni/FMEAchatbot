from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")
