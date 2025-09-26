from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from codebaseGeminiVector import retriever
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash-latest")

template = """
Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
context of codebase: {code_context}
user questions: {user_input}
"""

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    user_input = input("question: (q to quit)")
    if user_input == "q":
        break

    code_context = retriever.invoke(user_input)

    formatted_context = format_docs(code_context)

    print("----CONTEXT-----")
    print(formatted_context)
    print("----------------")

    result = chain.invoke({"code_context": code_context, "user_input": user_input})
    response_text = result.content
    print(response_text)

    # what does the function get_elements do?