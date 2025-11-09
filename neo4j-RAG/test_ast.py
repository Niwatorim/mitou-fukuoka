import json
import subprocess
import yaml
import os
from browser_use import Agent, ChatGoogle
import asyncio

#gemini
from google import genai
from google.genai import types

#chroma db
import chromadb
#langchain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

from pathlib import Path
# load_dotenv("../.env")

# fix loop through each file in root dir, and create each own ast -> copy ast_vector code

#put ast into code_structure.json (needs to be called in main)
# Debugging version
def ast_rag(file):
    parser_path= "./parser_test.js"
    command = ["node", parser_path, file]
    values = subprocess.run(
        command,
        capture_output=True,
        text=True,
        # Temporarily remove check=True to prevent a crash
        # check=True 
    )

    # If the command failed, print the error and exit
    if values.returncode != 0:
        print("--- Subprocess failed with exit code:", values.returncode)
        print("--- STDERR ---")
        print(values.stderr) # This will show the error from the Node.js script
        print("--- STDOUT ---")
        print(values.stdout)
        # You might want to raise an exception here in real code
        raise Exception("Node.js parser failed")

    return values.stdout

#embed the file code_structor.json
def embed_ast():

    client=genai.Client()
    chroma_client=chromadb.PersistentClient(path="./codebase_db")
    collection=chroma_client.get_or_create_collection(name="attributes")
    
    loader=TextLoader("./code_structure.json")
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits= text_splitter.split_documents(docs)
    
    for i, chunk in enumerate(splits):
        chunk.metadata["document_type"]= "Code data"
        chunk.metadata["chunk_id"]=i

    chunks= [e.page_content for e in splits]
    result=client.models.embed_content(
        model="gemini-embedding-001",
        contents = [e.page_content for e in splits],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",output_dimensionality=3072)
    )
    gemini_embeddings= [e.values for e in result.embeddings]

    collection.add(
        embeddings=gemini_embeddings,
        documents=chunks,
        metadatas=[chunk.metadata for chunk in splits],
        ids=[f"code_chunk_{chunk.metadata['chunk_id']}" for chunk in splits]
    )

#helper function for cycle()
def access_code(instructions):
    client=genai.Client()
    chroma_client= chromadb.PersistentClient(path="./codebase_db")

    # get the 2 collections
    attr_collection = chroma_client.get_collection(name="attributes")
    func_collection = chroma_client.get_collection(name="functions")

    # embedding the query, in this case is the intruction for the llm to generate instructions based on the components
    print("Embedding query...")
    query=instructions
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="CODE_RETRIEVAL_QUERY",
            output_dimensionality=3072 # Must match the dimension used for storage
        )
    )
    query_embedding = [e.values for e in result.embeddings]

    # query both collections
    attr_results = attr_collection.query( #queries the thing
        query_embeddings=query_embedding, # Use query_embeddings instead of query_texts
        n_results=2
    )
    func_results = func_collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
    
    # format the result of each query
    print("Formattings docs...")
    attr_docs=[]
    func_docs=[]
    for i in range(len(attr_results["ids"][0])):
        doc = Document(
            page_content=attr_results["documents"][0][i],
            metadata=attr_results["metadatas"][0][i]
        )
        attr_docs.append(doc)
    for i in range(len(func_results["ids"][0])):
        doc = Document(
            page_content=func_results["documents"][0][i],
            metadata=func_results["metadatas"][0][i]
        )
        attr_docs.append(doc)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key="AIzaSyB46zy_IKF197pOSJZDBXy-1PjHsKg46_k",
                                 model_kwargs={
                                     "response_mime_type":"application/yaml"
                                 })
    prompt = ChatPromptTemplate.from_template("""
    You are a test automation expert. Generate test instructions in YAML format.

    Attributes: {attr_context}
    Imports, functions, conditional rendering: {func_context}
    Component: {input}

    Return instructions with this exact structure and nothing else:
    if "text" is not empty, then use it to help identify the element within the test_steps.

    Requirements:
    - Each instruction must be ONE clear action.
    - Include specific selectors (id, class, data-testid, or text)
    - Use action types: {tool_context}

    Example:                                                                  
    component: "component_name",
    url: http://localhost:5173/
    test_steps:
    - step: 1
    action: browser_navigate
    instruction: Open the application
    target: http://localhost:5173/
    - step: 2
    action: browser_click
    instruction: Click link and verify navigation to its destination
    selector: a[href="https://vite.dev"]
    expected: Navigation to https://vite.dev
""")

    print("Making message...")
    document_chain = create_stuff_documents_chain(llm, prompt)

    with open("mcp_tools.yml", 'r', encoding='utf-8') as f:
        tool_context = f.read()

    print("invoke message...")
    response = document_chain.invoke({
        "input": query,
        "attr_context": attr_docs,
        "func_context": func_docs,
        "tool_context": tool_context
    })

    print(response)
    return (yaml.safe_load(response))

#helper function for cycle()
def unique_file(name,existing_files):
    count=1
    file=f"./tests/{name}.yaml"
    while file in existing_files:
        file=f"./tests/{name}[{count}].yaml"
        count+=1
    
    existing_files.add(file)
    return file

#go through every item in code_json and generate instructions for it using RAG from embedding, thne save to tests
def cycle():
    
    os.makedirs("tests", exist_ok=True)
    with open("code_structure.json","r") as f:
        data=json.load(f)

    existing_files = set()
    for index,i in enumerate(data["components"]):
        if i["testableAttributes"]:
            instruction= f"please give instructions to test the component {i}"
            yaml_data=access_code(instruction)
            filename = unique_file(i['name'], existing_files)

            with open(filename,"w") as f:
                yaml.dump(yaml_data,f,default_flow_style=False, sort_keys=False)

async def test_browser_use():
    directory= os.listdir("./tests")
    success_files=[]
    for file in directory:
        with open(f"./tests/{file}","r") as f:
            data=yaml.safe_load(f)
        task=str(yaml.dump(data["test_steps"], default_flow_style=False, sort_keys=False))
        agent = Agent(
            task=task,
            llm=ChatGoogle(model="gemini-2.5-flash"),
        )
        history = await agent.run()
        success={"name":file,"success":history.is_successful()}
        success_files.append(success)
    print("Failures:")
    for i in success_files:
        if i["success"]==False:
            print(i["name"])
        




async def main():
    file_path = "C:\\Users\\ThinkPad\\Documents\\Mitou_Fukuoka_25\\saim-code\\mitou-fukuoka\\samples\\src\\App.jsx"

    # The .stem attribute gives you the final path component without the suffix.
    file_name = Path(file_path).stem

    print(file_name)

    # data=json.loads(ast_rag("../samples/src/App.jsx"))
    # with open("code_structure.json","w") as f:
    #     json.dump(data,f,indent=4)
    # embed_ast()
    # cycle()
    # await test_browser_use()


    
    # with open("./tests/p.yaml","r") as f:
    #     data=yaml.safe_load(f)
    # task=str(yaml.dump(data["test_steps"], default_flow_style=False, sort_keys=False))
    # agent = Agent(
    #     task=task,
    #     llm=ChatGoogle(model="gemini-2.5-flash"),
    # )
    # history = await agent.run()
    # print(history.is_successful())
    # for i in history.action_history():
    #     print(i)



if __name__ == "__main__":
    asyncio.run(main())