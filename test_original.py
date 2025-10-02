#open-ai
from mcp import ClientSession
import asyncio

#ollama
from llama_index.core.agent.workflow import FunctionAgent,ToolCallResult,ToolCall
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from pydantic import create_model
from typing import List

#browser use
from browser_use import Agent, ChatGoogle

#crawl4ai
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from bs4 import BeautifulSoup
import json

#gemini
from dotenv import load_dotenv
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

#streamlit
import streamlit as st

#esprima
import esprima
import subprocess
load_dotenv()

if False:
    #--- for scraping
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

    if False:
        async def main():

            config = CrawlerRunConfig(
                deep_crawl_strategy=BFSDeepCrawlStrategy(
                    max_depth=2,
                    include_external=False
                ),
                scraping_strategy=LXMLWebScrapingStrategy(),
                verbose=True,
            )

            site="https://www.techwithtim.net"
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
                        content[url]=beautiful(result.html)
                        scraped.append(result.url)
                print(content)
                with open("temp.json","w") as f:
                    f.write(json.dumps(content,indent=2))

    """
    1. crawl4ai -> crawl each website to get raw html
    for crawled pages:
    2.    beautiful soup function

    Beautiful soup function-> recursive function with html object to be checked and path
    excluded tags include: ["script","style","link","meta","noscript","br","hr","span","b","i","u","strong","em","div"]
    for i in tags:
        if i not in exluded tags:
            ans = write the attributes and the path in json form
            add the path
            yield ans 
        else:
            add the path
        
        if i is dict:
            open up the stuff and recall the function with that new value (enter the dict)
        if i is list:
            for i in list:
                recall the function but with i as the new value

    3. final values are flat json with each wanted html and its corresponding path
            
    """

    def fix_schema(schema_part):
                if isinstance(schema_part, dict):
                    for key, value in list(schema_part.items()):
                        if key == "additionalProperties" and value is True:
                            schema_part[key] = {}  # The fix!
                        else:
                            fix_schema(value)
                elif isinstance(schema_part, list):
                    for item in schema_part:
                        fix_schema(item)
                return schema_part

    async def get_tools(session: ClientSession) -> List[FunctionTool]:
        await session.initialize()
        tool_definitions = await session.list_tools()
        llama_tools=[]
        for tool_def in tool_definitions.tools:
            fixed_schema = fix_schema(tool_def.inputSchema)

            pydantic_model= create_model(
                f"{tool_def.name}_Schema",
                **{prop: (dict,None) for prop in fixed_schema.get("properties",{}).keys()}
            )

            async def tool_function(tool_name:str = tool_def.name,**kwargs):
                print(f"Executing tool '{tool_name}' with arguments: {kwargs}")
                result=await session.call_tool(tool_name,kwargs)
                return result
            
            tool=FunctionTool.from_defaults(
                fn=tool_function,
                name=tool_def.name,
                description=tool_def.description,
                fn_schema=pydantic_model    
            )
            llama_tools.append(tool)
        return llama_tools

    async def get_agent(tools:list,llm,sys_prompt:str):

        agent = FunctionAgent(
            name="My Agent",
            description="An agent that can search the web",
            tools=tools,
            llm=llm,
            system_prompt=sys_prompt,
        )
        return agent

    async def handle_user_message(
        message_content: str,
        agent: FunctionAgent,
        agent_context: Context,
        verbose: bool = False,
    ):
        handler = agent.run(message_content,ctx=agent_context)
        async for event in handler.stream_events():
            if verbose and type(event)==ToolCall:
                print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
            elif verbose and type(event) == ToolCallResult:
                print(f"Tool {event.tool_name} returned {event.tool_output}")
        response = await handler
        return str(response)

#loading and splitting text
async def embed_code():
    client=genai.Client()
    chroma_client=chromadb.PersistentClient(path="./Code_database")
    collection=chroma_client.get_or_create_collection(name="Code")
    st.file_uploader("解析するファイルをアップロード",type=["jsx","py"])
    if st.button("実行開始"):
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

async def access_code():
    client=genai.Client()
    chroma_client= chromadb.PersistentClient(path="./Code_database")
    collection = chroma_client.get_collection(name="Code")
    query=st.text_area("リクエスト内容",placeholder="(例）ログインボタンをテストしてください...")
    if st.button("リクエストを処理"):
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
            with open("instructions.txt","w") as f:
                f.write(response)

#browser_use
from browser_use import Agent, ChatGoogle

#-- for browser-use
async def browseruse(): #for browser use
        st.subheader("タスクを実行中")
        with open("instructions.txt","r") as f:
            task=str(f.readlines())
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

#Streamlit
if False:
    st.title("オクタゴンテスター")
    st.write("オプションを選択してください：")
    user=st.selectbox("アクションを選択してください:",
    ("コードの埋め込み", "コードにアクセス","タスクを実行"))
    if user == "コードの埋め込み":
        asyncio.run(embed_code())            
    elif user == "コードにアクセス":
        asyncio.run(access_code())
    elif user == "タスクを実行":
        if st.button("テスト開始"):
                asyncio.run(browseruse())

#-- linking code to functions

def parse_code(code_string):
    values=subprocess.run(
        ["node","ast_generator.js"],
        input=code_string,
        capture_output=True,
        text=True,
        check=True,
    )
    return values.stdout


# ast_code=parse_code("const x = 2")
# ast_data=json.loads(ast_code)
# print(ast_data)

def ast_rag(file):
    command = ["node","parser_test.js",file]
    values=subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )
    return values.stdout
data=json.loads(ast_rag("./test-project/src/App.jsx"))
with open("code_structure.json","w") as f:
    json.dump(data,f,indent=4)
    
