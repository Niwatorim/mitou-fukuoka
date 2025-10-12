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
load_dotenv("../.env")

#put ast into code_structure.json (needs to be called in main)
def ast_rag(file):
    parser_path= "/Users/niwatorimostiqo/Desktop/Coding/Mitou Fukuoka/parser_test.js"
    command = ["node",parser_path,file]
    values=subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )
    return values.stdout

#embed the file code_structor.json
def embed_ast():

    client=genai.Client()
    chroma_client=chromadb.PersistentClient(path="../Code_database")
    collection=chroma_client.get_or_create_collection(name="ast")
    
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
    chroma_client= chromadb.PersistentClient(path="../Code_database")
    collection = chroma_client.get_collection(name="ast")
    query=instructions
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="CODE_RETRIEVAL_QUERY",
            output_dimensionality=3072 # Must match the dimension used for storage
        )
    )
    print("embedding....")
    query_embedding = [e.values for e in result.embeddings]

    results = collection.query( #queries the thing
        query_embeddings=query_embedding, # Use query_embeddings instead of query_texts
        n_results=2
    )
    
    docs=[]
    for i in range(len(results["ids"][0])):
        doc = Document(
            page_content=results["documents"][0][i],
            metadata=results["metadatas"][0][i]
        )
        docs.append(doc)
    print("making IDs")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key="AIzaSyB46zy_IKF197pOSJZDBXy-1PjHsKg46_k",
                                 model_kwargs={
                                     "response_mime_type":"application/yaml"
                                 })
    prompt = ChatPromptTemplate.from_template("""
    You are a test automation expert. Generate test instructions in JSON format.

    Context: {context}
    Component: {input}

    Return instructions with this exact structure and nothing else:
    "component": "component_name",
    "url": "http://localhost:5173/",
    "test_steps": [
        "step": 1, "action": "navigate", "instruction": "Open the application", "target": "http://localhost:5173/",
        "step": 2, "action": "click", "instruction": "Click the submit button","text":"Submit", "selector": "#submit-btn", "expected": "Form submits successfully",
    ]
    

    Requirements:
    - Each instruction must be ONE clear action
    - Include specific selectors (id, class, data-testid, or text)
    - Use action types: navigate, click, type, verify, wait, select
""")

    print("Making message")
    document_chain = create_stuff_documents_chain(llm,prompt)

    print("invoke message")
    response = document_chain.invoke({
        "input": query,
        "context": docs
    })

    print(response)
    return (json.loads(response))

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
    # data=json.loads(ast_rag("/Users/niwatorimostiqo/Desktop/Coding/Mitou Fukuoka/test-project/src/App.jsx"))
    # with open("code_structure.json","w") as f:
    #     json.dump(data,f,indent=4)
    # embed_ast()
    # cycle()
    await test_browser_use()

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