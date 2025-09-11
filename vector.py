from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import json

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)
file_path = "rag_test.json"

def make_document(node, path=""):
    # each document contain information for an interactive element: a, button, input, img
    # and a metadata containing the path to reach that element
    # do recursive for the children

    documents = []
    current_path = f"{path} > {node.get('tag', 'item')}" if path else node.get('tag', 'root')

    has_text = 'text' in node and node['text']
    is_interactive = node.get('tag') in ['a', 'button', 'input', 'img']
    
    if has_text and is_interactive:
        node_content = {k: v for k, v in node.items() if k!= 'children'}
        content_string = json.dumps(node_content, separators=(',', ':'))

        documents.append(
            Document(
                page_content=content_string,
                metadata={'path': current_path}
            )
        )
    if 'children' in node and node['children']:
        for child in node['children']:
            documents.extend(make_document(child, path=current_path))
    return documents

def retrieve_data(file_path):  
    with open(file_path, 'r') as f:
        data = json.load(f)
    all_documents = []
    for key, root_node in data.items():
        all_documents.extend(make_document(root_node, path=key))
    print(f"created {len(all_documents)} documents with paths")
    return all_documents

if add_documents:
    documents = retrieve_data(file_path)

vector_store = Chroma(
    collection_name="website_layout",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents)

retriever = vector_store.as_retriever()