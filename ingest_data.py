# ingest_data.py

import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function


CHROMA_PATH = "fmea_chroma"
DATA_PATH = "data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("🧹 Resetting FMEA vector database...")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=80, length_function=len
    )
    return splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    chunks = calculate_chunk_ids(chunks)
    existing_ids = set(db.get(include=[])["ids"])

    new_chunks = [chunk for chunk in chunks if chunk.metadata["id"] not in existing_ids]
    if new_chunks:
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
        db.persist()
        print(f"✅ Added {len(new_chunks)} new FMEA chunks.")
    else:
        print("🔁 No new chunks to add.")

def calculate_chunk_ids(chunks):
    last_page_id, current_chunk_index = None, 0
    for chunk in chunks:
        src = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{src}:{page}"
        current_chunk_index = current_chunk_index + 1 if current_page_id == last_page_id else 0
        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
