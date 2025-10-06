from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from vector2 import retriever2
from langchain_core.documents import Document

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(google_api_key=api_key, model="models/gemini-2.5-flash")
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
## CODEBASE CONTEXT
Here are the functions of each component in the website
{codebase_context}
---
## TASK
Analyze the user's request and create a numbered list of steps to accomplish it. For each step, state which tool to use, what action to perform, and which element selector to use from the website context.

### EXAMPLE
User Request: "Log in with email 'test@example.com' and password 'password123'."

Test Plan:
1. Use the `browser_type` tool to type 'test@example.com' into the email input field, identified by the selector `#username` located at #login-form > div:nth-of-type(1) > input .
3. Use the `browser_click` tool to click the 'Log In' button, which has the selector `#login-submit-btn` located at #login-form > button.

Finally, include in the instructions in the end:
* Stop the event loop and close the browser after any verification is done exactly once. 
* Tell to generate the step by step playwright code script of the actions that just performed. 
    Example: 
        from playwright.sync_api import sync_playwright, expect
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto("https://www.example.com")
            heading_element = page.locator("h1")
            expect(heading_element).to_have_text("Example Domain")
            context.close()
            browser.close()
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
        located_at_webpage = doc.metadata.get('web_page_id', 'N/A')
        formatted_string = (
            f"Element description: {description}"
            f"Selector: {selector}"
            f"Located at webpage: {located_at_webpage}"
        )
        formatted_strings.append(formatted_string)
    return "\n\n---\n\n".join(formatted_strings)

with open("mcp_tools.yml", 'r', encoding='utf-8') as f:
    tool_context = f.read()

while True:
    user_input = input("What do you want me to test?(q to quit)")
    if user_input == "q":
        break

    retrieved_docs = retriever2.invoke(user_input)

    rag_context = format_docs(retrieved_docs)

    print("----CONTEXT-----")
    print(rag_context)
    print("----------------")

    result = chain.invoke({"tool_context": tool_context, "rag_context": rag_context, "user_input": user_input})
    print(result.content)