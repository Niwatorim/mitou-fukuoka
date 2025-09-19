from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from langchain_core.documents import Document

model = OllamaLLM(model="llama3.2")

template = """
You are an expert QA Automation Engineer. Your task is to convert a user's natural language instruction into a precise, step-by-step test plan in a JSON format for a Playwright automation agent.

**CONTEXT:**
Here are the relevant interactive elements from the web page. You MUST use the provided "Selector" to identify elements for your actions.

{web_context}

**INSTRUCTIONS:**
1.  Analyze the user's request: "{user_input}".
2.  Break the request down into a logical sequence of actions (e.g., `click`, `fill`, `assert_visible`).
3.  For each action, use the most appropriate "Selector" from the CONTEXT. Do not invent selectors.
4.  Include verification steps (`assert_visible`) to confirm the outcome of actions (e.g., after login, check for a "Welcome" message).
5.  Your final output must be a single JSON array of action objects. Each object must have an "action", "selector", and optional "value" and "reason" keys.

**EXAMPLE:**
---
User Input: "Log in using the email 'tester@google.com' and password 'securepass123'."

Output:
[
  {{
    "action": "fill",
    "selector": "input[name='email']",
    "value": "tester@google.com",
    "reason": "To fill the email address as requested by the user."
  }},
  {{
    "action": "fill",
    "selector": "#password-input",
    "value": "securepass123",
    "reason": "To fill the password as requested by the user."
  }},
  {{
    "action": "click",
    "selector": "#login-button",
    "reason": "To submit the login form."
  }},
  {{
    "action": "assert_visible",
    "selector": "h1.user-dashboard-title",
    "reason": "To verify that the login was successful by checking for the dashboard title."
  }}
]
---

**YOUR TASK:**
User Input: "{user_input}"

Output:
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