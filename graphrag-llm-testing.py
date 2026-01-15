from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the Model
# Ensure the model name matches the tag you pulled in Ollama (e.g., "qwen2.5")
llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0 # Optional: set temperature for deterministic results
)

# 2. Define the Prompt Template with a System Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a web QA tester. Extract the UI components and actions from the prompt, and put them as a list. For example, Prompt: Check whether the home button has the home logo, and directs to the shop link, and whether the cat image is present. Response you should give: home button, home logo, shop link, cat image. Don't forget the quotation mark for each phrase in the list"),
    ("user", "{question}")
])

# 3. Create the Chain
# Prompt -> LLM -> Output Parser
chain = prompt | llm | StrOutputParser()

# 4. Invoke the Chain with a User Question
question = "check whether the home button takes to the shopping cart page"
response = chain.invoke({"question": question})

prompt_array = [item.strip() for item in response.split(',')]

print(response)
print(prompt_array)
