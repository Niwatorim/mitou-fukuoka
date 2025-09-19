from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector2 import retriever
from langchain_core.documents import Document

model = OllamaLLM(model="llama3.2")

template = """
You are an expert QA Automation Engineer. Your task is to convert a user's request into a clear, step-by-step test plan written in natural language.
---
## AVAILABLE TOOLS
Here is the list of tools your AI agent can use. You must explicitly mention the tool name in each step.

{tool_context}
---
## WEBSITE CONTEXT
Here are the relevant interactive elements from the web page. Use the 'Selector' to pass on the location of the element in the website

{rag_context}
---
## TASK
Analyze the user's request and create a numbered list of steps to accomplish it. For each step, state which tool to use, what action to perform, and which element selector to use from the website context.

### EXAMPLE
User Request: "Log in with email 'test@example.com' and password 'password123'."

Test Plan:
1. Use the `browser_type` tool to type 'test@example.com' into the email input field, identified by the selector `#username` located at #login-form > div:nth-of-type(1) > input .
2. Use the `browser_type` tool to type 'password123' into the password input field with the selector `#password` located at #login-form > div:nth-of-type(2) > input.
3. Use the `browser_click` tool to click the 'Log In' button, which has the selector `#login-submit-btn` located at #login-form > button.
---

## YOUR TURN
User Request: "{user_input}"

Test Plan:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def format_docs(docs: list[Document]) -> str:
    formatted_strings = []
    for doc in docs:
        description = doc.page_content
        selector = doc.metadata.get('path_to_element', 'N/A')
        formatted_string = (
            f"Element description: {description}"
            f"Selector: {selector}"
        )
        formatted_strings.append(formatted_string)
    return "\n\n---\n\n".join(formatted_strings)

with open("mcp_tools.yml", 'r', encoding='utf-8') as f:
    tool_context = f.read()

while True:
    user_input = input("What do you want me to test?(q to quit)")
    if user_input == "q":
        break

    retrieved_docs = retriever.invoke(user_input)

    rag_context = format_docs(retrieved_docs)

    print("----CONTEXT-----")
    print(rag_context)
    print("----------------")

    result = chain.invoke({"tool_context": tool_context, "rag_context": rag_context, "user_input": user_input})
    print(result)