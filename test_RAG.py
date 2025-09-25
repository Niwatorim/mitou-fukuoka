import vertexai
from vertexai import rag
from google.genai.types import Tool, Retrieval, VertexRagStore
import google.genai as genai

# --- FIX: ADD THIS INITIALIZATION BLOCK ---
vertexai.init(
    project="gothic-handbook-472610-h3", # Replace with your project ID
    location="asia-northeast1"   # Use the Tokyo region
)
# -----------------------------------------

# 1. Create the RAG Corpus
EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"

rag_corpus = rag.create_corpus(
    display_name="my-codebase-corpus",
    description="Codebase files from local",
)

# 2. Import your files from Google Cloud Storage
GCS_IMPORT_URI = "gs://mitou-fukuoka-code-storage-2025/" # Use your bucket name

import_response = rag.import_files(
    corpus_name=rag_corpus.name,
    paths=[GCS_IMPORT_URI],
    chunking_config=rag.ChunkingConfig(
        chunk_size=500,
        chunk_overlap=100
    ),
)

# 3. Create the RAG retrieval tool
rag_retrieval_tool = Tool.from_retrieval(
    Retrieval(
        source=VertexRagStore(
            rag_corpora=[rag_corpus.name],
        ),
    )
)

# 4. Configure and Initialize the Gemini Model
genai.configure(project="gothic-handbook-472610-h3") # Replace with your project ID
model = genai.GenerativeModel(
    "gemini-1.5-flash-001",
    tools=[rag_retrieval_tool],
)

# 5. Query the model
response = model.generate_content(
    "What is the primary purpose or main functionality of this codebase?"
)

print(response.text)