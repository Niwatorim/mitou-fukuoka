#---- Dependencies
#misc
from dotenv import load_dotenv
import streamlit as st
import asyncio
from browser_use import Agent, ChatGoogle
import json
from typing import List
import networkx as nx
from urllib.parse import urljoin

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

#Scraping
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from bs4 import BeautifulSoup

# website RAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
import os
import json
import typing

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

#---- web components RAG -> need to put everything in the agent prompt chad
def generate_description(element: dict) -> str:
    tag = element.get("tag", "N/A")
    text = element.get("text", "").strip()
    attributes = element.get("attributes", {})

    description = f"A {tag} element. "
    if text:
        description += f"Contains the text: '{text}'. "
    if 'id' in attributes:
        description += f"The id is {attributes['id']}. "
    if 'class' in attributes:
        description += f"The class is {attributes['class']}. "
    if 'name' in attributes:
        description += f"The name is {attributes['name']}. "
    if 'href' in attributes:
        description += f"The url is {attributes['href']}. "
    if 'src' in attributes:
        description += f"The src is {attributes['src']}. "
    if 'alt' in attributes:
        description += f"The alt is {attributes['alt']}. "
    if 'type' in attributes:
        description += f"The type is {attributes['type']}. "
    if 'value' in attributes:
        description += f"The placeholder is {attributes['placeholder']}. "
    if 'role' in attributes:
        description += f"The role is {attributes['role']}. "
    if 'aria-label' in attributes:
        description += f"The aria-label is {attributes['aria-label']}. "

    return description

def make_documents(file_path) -> list[Document]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for web_page, element_list in data.items():
        web_page_id = web_page
        for element in element_list:
            text_for_embedding = generate_description(element)
            path_to_element = element.get("locator", "")

            doc_metadata = {
                "path_to_element": path_to_element,
                "web_page_id": web_page_id
            }

            doc = Document(
                page_content=text_for_embedding,
                metadata=doc_metadata
            )
            documents.append(doc)
        
        print(f"Successfully created {len(documents)} documents from {file_path}")
        return documents

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

#---- Embeds website for RAG
async def embed_website():
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/text-embedding-004")

    db_location = "./components_db"
    add_documents = not os.path.exists(db_location)
    file_path = "rag_test2.json"

    vector_store = Chroma(
        collection_name="website_layout",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        documents = make_documents(file_path)
        vector_store.add_documents(documents=documents)
        print("Succesfully stored in vector database")
    else: 
        print("Database already existed.")

    website_retriever = vector_store.as_retriever()

#---- Embeds Code for Rag
async def embed_code():
    client=genai.Client()
    chroma_client=chromadb.PersistentClient(path="./Code_database")
    collection=chroma_client.get_or_create_collection(name="Code")
    upload=st.file_uploader("解析するファイルをアップロード",type=["jsx","py"])
    if st.button("実行開始"):
        if upload:
            temp_path = f"./temp_{upload.name}"
            with open(temp_path, "wb") as f:
                f.write(upload.getvalue())
            loader=TextLoader(f"./temp_{upload.name}")
        else:
            loader=TextLoader("./test-project/src/App.jsx") #pull up the text
        docs=loader.load() #load the text
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
        st.success("コードがデータベースに追加されました！")

#---- User request for COMBINED rag system
async def access_code(user_input):
    # website rag retrieval
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key, 
        model="models/text-embedding-004"
    )
    
    vector_store = Chroma(
        collection_name="website_layout",
        persist_directory="./components_db",
        embedding_function=embeddings
    )
    
    website_retriever = vector_store.as_retriever()
    website_docs = website_retriever.invoke(user_input)
    web_context = format_docs(website_docs)

    # code rag retrieval
    client = genai.Client()
    chroma_client = chromadb.PersistentClient(path="./Code_database")
    collection = chroma_client.get_collection(name="Code")
    
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=user_input,
        config=types.EmbedContentConfig(
            task_type="CODE_RETRIEVAL_QUERY",
            output_dimensionality=3072
        )
    )
    query_embedding = [e.values for e in result.embeddings]
    
    code_results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )
        
    code_docs = []
    for i in range(len(code_results["ids"][0])):
        doc = Document(
            page_content=code_results["documents"][0][i],
            metadata=code_results["metadatas"][0][i]
        )
        code_docs.append(doc)
    
    # combined prompt
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key="AIzaSyB46zy_IKF197pOSJZDBXy-1PjHsKg46_k")
        with open("mcp_tools.yml", 'r', encoding='utf-8') as f:
            tool_context = f.read()
        prompt = ChatPromptTemplate.from_template("""
            You are an expert QA Automation Engineer. Your task is to convert a user's request into a clear, step-by-step test plan written in natural language.
            ---
            ## AVAILABLE TOOLS
            Here is the list of tools your AI agent can use. You must explicitly mention the tool name in each step.

            {tool_context}
            ---
            ## WEBSITE CONTEXT
            Here are the relevant interactive elements from the web page. Use the 'Selector' to pass on the location of the element in the website

            {web_components_context}
            ---
            ## CODEBASE CONTEXT
            Here are the components of the website. If there are conditional rendering, example: "authUser ? <HomePage /> : <Navigate to="/login" /> means if authUser (user is authenticated) then go to Homepage, but if not then go to /login. 
            {codebase_context}
                                                  
            ## TASK
            Analyze the user's request and create a numbered list of steps to accomplish it. For each step, state which tool to use, what action to perform, and which element selector to use from the website context.

            ### EXAMPLE
            User Request: "Log in with email 'test@example.com' and password 'password123'."

            Test Plan:
            1. Use the `browser_type` tool to type 'test@example.com' into the email input field, identified by the selector `#username` located at #login-form > div:nth-of-type(1) > input .
            3. Use the `browser_click` tool to click the 'Log In' button, which has the selector `#login-submit-btn` located at #login-form > button.

            ## Finally, include in the instructions in the end:
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
        """)

        chain = create_stuff_documents_chain(llm, prompt)

        response = chain.invoke({
        "input": user_input,
        "context": code_docs,  # LangChain documents for the chain
        "tool_context": tool_context,
        "web_context": web_context,
        "code_context": "\n\n".join([doc.page_content for doc in code_docs]),
        "user_input": user_input
    })

        st.success("指示が正常に作成されました")
        st.divider()
        st.subheader("エージェントへの指示:")
        st.write(response)
        with open("instructions-code.txt","w") as f:
            f.write(response)

#---- Browser use + added to read the generated code
async def browseruse(): #for browser use
        st.subheader("タスクを実行中")
        with open("instructions.txt","r") as f:
            task=str(f.read())
        agent = Agent(
            task=task,
            llm=ChatGoogle(model="gemini-2.5-flash"),
        )
        history = await agent.run()
        for i in history.model_actions():
            key= list(i.keys())[0]
            if key == "replace_file_str":
                st.subheader("生成されたPlaywrightコード (Generated Playwright Code)")
                # Use st.code to display it beautifully
                st.code(i[key], language="python")
            elif key == "done" and i[key]["success"] == True:
                data=i[key]["text"]
                st.success(data)
            elif key != "replace_file_str":
                with st.expander(key):
                    reg_data=i[key]
                    st.write(reg_data)

#---- To input graph and draw nodes
def path_generation(element):
    values=[]
    child=element
    for parent in child.parents: #for every parent object
        siblings=parent.find_all(child.name,recursive=False)
        if len(siblings) > 1: #if has siblings
            count = 1
            for sib in siblings:
                if sib is element:
                    values.append(f"{child.name}[{count}]")
                    break
                count+=1
        else:
            values.append(child.name)
        if parent.name == '[document]':
            break
        child=parent
    values.reverse()
    return "/" + "/".join(values)

def beautiful(data):
    soup = BeautifulSoup(data,"lxml")
    tags=["a", "button", "input", "select", "textarea", "form", "h1", "h2", "h3", "p", "img", "li"]
    interaction_map=[]
    for element in soup.find_all(tags):
        xpath=path_generation(element)

        attributes={}
        attributes_to_find=["id", "class", "name", "href", "src", "alt", "type", "value", "placeholder", "role", "aria-label"]
        for attribute in element.attrs:
            if attribute in attributes_to_find:
                attributes[attribute]=element.attrs[attribute]
        item={
            "tag": element.name,
            "text":element.get_text(strip=True),
            "locator":xpath,
            "attributes":attributes
        }
        interaction_map.append(item)
    return interaction_map

async def main_scraping():
    web=st.text_input("Give site to scrape, else default to techwithtim")
    if st.button("Start"):
        if web== "":
            site="https://www.techwithtim.net"
        else:
            site=web
            st.session_state.site=web
        
        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=2,
                include_external=False,
                max_pages=7
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            verbose=True,
        )

        async with AsyncWebCrawler() as crawler:
            results = await crawler.arun(
                url=site,
                config=config
            )
            scraped=[]
            content={}
            for result in results:
                if result.url not in scraped:
                    url=str(result.url).replace(site,"")
                    if url == "":
                        url = "/"
                    content[url]=beautiful(result.html)
                    scraped.append(result.url)

            with open("temp.json","w") as f:
                f.write(json.dumps(content,indent=2))

#---- for shortest path
def getpath(data, value,original, prepath=()):#Find the path for hrefs
    if type(data) is dict:
        if "attributes" in data and type(data["attributes"]) is dict and "href" in data["attributes"]: # found key
                href=data["attributes"]["href"]
                original_page=prepath[0]
                href= urljoin(original_page,href) 
                yield [href,original_page,data]
        for k, v in data.items():
            path=prepath+(k,)#without the comma it comes out as a string in brackets and cant concatenate
            yield from getpath(v, value,original,path) # recursive call
    elif type(data) is list:
        for i in range(len(data)):
            val=data[i]
            path=prepath + (f"{i}",) #comma here makes it a tuple
            yield from getpath(val,value,original,path)

def shortestPath(G,edge_labels) -> List[str]: #Find shortest path
    if "site" not in st.session_state:
        st.session_state.site="https://www.techwithtim.net"
    source= "/"
    nodes=get_nodes()
    end = st.text_input(f"From home to node....?{nodes}")
    if st.button("Continue"):
        shortest=nx.shortest_path(G,source,end)
        links = list(zip(shortest,shortest[1:]))
        steps=[]
        for i in links:
            step="use "+edge_labels[(i[0],i[1],0)]+f" to go from {st.session_state.site}"+i[0]+f" to {st.session_state.site}"+i[1] #edge labels[0] is a problem, cuz it only shows one way to get there not all
            steps.append(step)
        return steps

def get_nodes()->List[str]:
    with open("temp.json","r") as f:
        content=json.load(f)
    nodes=[]
    for k,v in content.items():
        nodes.append(str(k))
    return nodes

def path_instructions(destination_node):
    with open("temp.json", "r") as file:
        data = json.load(file)
        nodes_data = list(getpath(data, "href", data))
        for i in nodes_data:
            i[2] = str("'" + i[2]["tag"] + "-" + i[2]["text"] + "' located in: '" + i[2]["locator"] + "' ")
        nodes = [(i[1], i[0], {"label": i[2]}) for i in nodes_data]
        G = nx.MultiDiGraph()
        G.add_edges_from(nodes)
        edge_labels = nx.get_edge_attributes(G, "label")

        contents_web_list = find_shortest_path_steps(G, edge_labels, source="/", end=destination_node)
        with open("instructions-web.txt","w") as f:
            f.writelines(contents_web_list)

        return contents_web_list

def find_shortest_path_steps(G, edge_labels, source, end) -> List[str]:
    if "site" not in st.session_state:
        st.session_state.site = "https://www.techwithtim.net"
    try:
        shortest = nx.shortest_path(G, source, end)
        links = list(zip(shortest, shortest[1:]))
        steps = []
        for i in links:
            label = edge_labels.get((i[0], i[1], 0), "[unknown action]")
            step = f"use {label} to go from {st.session_state.site}{i[0]} to {st.session_state.site}{i[1]}"
            steps.append(step)
        return steps
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        st.error(f"No path could be found from '{source}' to '{end}'. Please check if the destination node exists.")
        return None

#---- Generate interactive graph with networkx and pyvis
def generate_graph():
    pass

#--- Streamlit
async def main():

    st.title("Octagon Tester")
    st.write("Select option:")
    user = st.selectbox("Choose Action", ("Embed Code", "Scrape", "User request", "Browser Use"))

    if user == "Embed Code":
        await embed_code()            

    elif user == "Scrape":
        await main_scraping()

    elif user == "user request":
        if "success" not in st.session_state:
            st.session_state.success = False

        st.subheader("User Request")
        req = st.text_area("Request box", placeholder="please test the tutorials button...")
        
        # ADDED: A text box to specify the destination for the pathfinding logic
        destination_node = st.text_input("Destination Node for Web Path", placeholder="/tutorials")

        options = st.multiselect(
            "What will you RAG?",
            ["Code", "Web"],
            default=["Code", "Web"],
        )

        if st.button("Start"):
            st.session_state.success = False
            contents_code = ""
            contents_web_list = [] # Will be a list of strings

            if "Code" in options:
                await access_code(req)
                with open("instructions-code.txt", "r") as f:
                    contents_code = f.read()
            
            if "Web" in options:
                if destination_node: 
                    contents_web_list= path_instructions(destination_node)
                else:
                    st.warning("Please provide a destination node for the web path.")

            if options:
                with open("instructions.txt", "w") as f:
                    f.write("Code Rag contents are:\n")
                    f.write(contents_code)
                    f.write("\n\nPath to get to location from home:\n")
                    if contents_web_list:
                        f.write("\n".join(contents_web_list))
                    else:
                        f.write("")
                
                st.session_state.success = True

        if st.session_state.success:
            st.success("Written instructions")

    elif user == "Browser Use":
        if st.button("Start Agent"):
            await browseruse()

if __name__ == "__main__":
    asyncio.run(main())