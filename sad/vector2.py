from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
import os
import json
import typing

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/text-embedding-004")

db_location = "./components_db"
add_documents = not os.path.exists(db_location)
file_path = "rag_test2.json"

def generate_description(element: dict) -> str:
    tag = element.get("tag", "N/A")
    text = element.get("text", "").strip()
    attributes = element.get("attributes", {})

    description = f"A {tag} element. "
    if text:
        description += f"Contains the text: '{text}'. "
    if 'id' in attributes:
        description += f"The id is {attributes['id']}. "
    if 'class' in attributes:
        description += f"The class is {attributes['class']}. "
    if 'name' in attributes:
        description += f"The name is {attributes['name']}. "
    if 'href' in attributes:
        description += f"The href is {attributes['href']}. "
    if 'src' in attributes:
        description += f"The src is {attributes['src']}. "
    if 'alt' in attributes:
        description += f"The alt is {attributes['alt']}. "
    if 'type' in attributes:
        description += f"The type is {attributes['type']}. "
    if 'value' in attributes:
        description += f"The placeholder is {attributes['placeholder']}. "
    if 'role' in attributes:
        description += f"The role is {attributes['role']}. "
    if 'aria-label' in attributes:
        description += f"The aria-label is {attributes['aria-label']}. "

    return description

def make_documents(file_path) -> list[Document]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for web_page, element_list in data.items():
        web_page_id = web_page
        for element in element_list:
            text_for_embedding = generate_description(element)
            path_to_element = element.get("locator", "")

            doc_metadata = {
                "path_to_element": path_to_element,
                "web_page_id": web_page_id
            }

            doc = Document(
                page_content=text_for_embedding,
                metadata=doc_metadata
            )
            documents.append(doc)
        
        print(f"Successfully created {len(documents)} documents from {file_path}")
        return documents

# docs = make_documents(file_path)
# print(docs[1])
# print(docs[1].page_content)
# print(docs[1].metadata)

vector_store = Chroma(
    collection_name="website_layout",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents = make_documents(file_path)
    vector_store.add_documents(documents=documents)
    print("Succesfully stored in vector database")
else: 
    print("Database already existed.")

retriever2 = vector_store.as_retriever()

