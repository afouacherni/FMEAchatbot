from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"  # 👈 obligatoire pour éviter l'erreur
    )
