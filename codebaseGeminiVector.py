from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/text-embedding-004")

repo_path = r"C:\Users\ThinkPad\Documents\Mitou_Fukuoka_25\saim-code\mitou-fukuoka\reactapp\my-react-app\src"

# Load documents from repository
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="App.jsx", 
    suffixes=[".jsx"],
    parser=LanguageParser(language=Language.JS, parser_threshold=50)
)
documents = loader.load() # number of documents depend on the number of files
# print(len(documents))

# code spliter
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=500, # maximum number of characters in each chunk
    chunk_overlap=50 # how many characters from the end of one chunk are repeated at the beginning of the next, helps maintain context between chunks
)

# split python file into chunks of code
chunks = js_splitter.split_documents(documents)
# print(len(chunks))

# print(chunks[2].page_content)

# make the vectorstore
db_location = "./codebase_db2"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="codebase",
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





