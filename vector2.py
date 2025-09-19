from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db2"
add_documents = not os.path.exists(db_location)
file_path = "rag_test2.json"

def make_documents(file_path) -> list[Document]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for element in data:
        text_for_embedding = element.get("text_for_embedding", "")

        original_metadata = element.get("metadata", {})

        path_to_element = original_metadata.get("path_to_element", 'N/A')

        doc_metadata = {
            "path_to_element": path_to_element
        }

        doc = Document(
            page_content=text_for_embedding,
            metadata=doc_metadata
        )
        documents.append(doc)
    
    print(f"Successfully created {len(documents)} documents from {file_path}")
    return documents


vector_store = Chroma(
    collection_name="website_layout",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents = make_documents(file_path)
    vector_store.add_documents(documents=documents)

retriever = vector_store.as_retriever()