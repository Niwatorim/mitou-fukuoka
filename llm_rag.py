from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from langchain_core.documents import Document

model = OllamaLLM(model="llama3.2")

template = """
Using the website layout context, give a brief answer about the website.
website layout context: {web_context}
User input: {user_input}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_docs(docs: list[Document]) -> str:
    formatted_strings = []
    for doc in docs:
        formatted_string = (
            f"Element Path: {doc.metadata.get('path', 'N/A')}\n"
            f"Element Details: {doc.page_content}"
        )
        formatted_strings.append(formatted_string)
    return "\n\n---\n\n".join(formatted_strings)

while True:
    user_input = input("What do you want me to test?(q to quit)")
    if user_input == "q":
        break

    retrieved_docs = retriever.invoke(user_input)

    web_context = format_docs(retrieved_docs)

    print("----CONTEXT-----")
    print(web_context)
    print("----------------")

    result = chain.invoke({"web_context": web_context, "user_input": user_input})
    print(result)