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
load_dotenv()

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

#---- User request for Rag
async def access_code(question):
    client=genai.Client()
    chroma_client= chromadb.PersistentClient(path="./Code_database")
    collection = chroma_client.get_collection(name="Code")
    query=question
    with st.spinner("リクエストを処理中... "):    
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="CODE_RETRIEVAL_QUERY",
                output_dimensionality=3072 # Must match the dimension used for storage!
            )
        )
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
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key="AIzaSyB46zy_IKF197pOSJZDBXy-1PjHsKg46_k")
        prompt = ChatPromptTemplate.from_template("""
            You are an automation tester and must tell me the instructions to test an object on a website. 
            Please tell me the detailed instructions in numbered steps to check if code is working
            Answer the following question based only on the provided context.
            Provide a simple set of instructions
            to first open the website, use the following link: http://localhost:5173/
            
            <context>
            {context}
            </context>
                    
            Question: {input}                                                           
        """)

        document_chain = create_stuff_documents_chain(llm,prompt)

        response = document_chain.invoke({
            "input": query,
            "context": docs
        })

        st.success("指示が正常に作成されました")
        st.divider()
        st.subheader("エージェントへの指示:")
        st.write(response)
        with open("instructions-code.txt","w") as f:
            f.write(response)

#---- Browser use
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
            if key == "done" and i[key]["success"] == True:
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


#--- Streamlit
async def main():

    st.title("Octagon Tester")
    st.write("Select option:")
    user = st.selectbox("Choose Action", ("Embed Code", "Scrape", "Access Code", "Browser Use", "user request"))

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