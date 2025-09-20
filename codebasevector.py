from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from git import Repo
import os

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define the path using raw strings or os.path.join for consistency
repo_path = r"C:\Users\ThinkPad\Documents\Mitou_Fukuoka_25\saim-code\mitou-fukuoka"

# Load documents from the cloned repository
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="test.py", # **/ meaning load recursively, search for that filename in every folder of the directory
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)
documents = loader.load() # number of documents depend on the number of files
# print(len(documents))

# python code spliter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000, # maximum number of characters in each chunk
    chunk_overlap=200 # how many characters from the end of one chunk are repeated at the beginning of the next, helps maintain context between chunks
)

# split python file into chunks of code
chunks = python_splitter.split_documents(documents)
# print(len(chunks))

# print(chunks[2].page_content)

# make the vectorstore
db_location = "./codebase_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="codebase_python",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=chunks)
    print("Successfully added codebase chunks")
else:
    print("Vectorstore existed already")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)