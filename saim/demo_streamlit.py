#open-ai
import subprocess
from mcp import ClientSession
from mcp.client.sse import sse_client
import time
import asyncio

#ollama
# from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
# from llama_index.core.agent.workflow import FunctionAgent,ToolCallResult,ToolCall
# from llama_index.core.workflow import Context
# from llama_index.llms.ollama import Ollama
# from llama_index.core.tools import FunctionTool
from pydantic import create_model
from typing import List

#browser use
from browser_use import Agent, ChatGoogle

#crawl4ai
# from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
# from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
# from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from bs4 import BeautifulSoup
import json

#gemini
from dotenv import load_dotenv
from google import genai
from google.genai import types

#chroma db
import chromadb
from chromadb.utils import embedding_functions

#langchain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

#streamlit
import streamlit as st
load_dotenv()
import base64
import logging
import platform


#loading and splitting text
async def embed_code():
    client=genai.Client()
    chroma_client=chromadb.PersistentClient(path="./Code_database")
    collection=chroma_client.get_or_create_collection(name="Code")
    st.file_uploader("解析するファイルをアップロード",type=["jsx","py"])
    if st.button("実行開始"):
        loader=TextLoader("./my-react-app/src/App.jsx") #pull up the text
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
    chroma_client= chromadb.PersistentClient(path="./Code_database") # Code database contains error test
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
                f.write("\nif the test from the users request fails, alert the user that the test has failed")
            

# --- FIX FOR PLAYWRIGHT ON WINDOWS ---
# This must be at the top of your script
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
            max_failures=4
        )
        history = await agent.run()
        for i in history.model_actions():
            key= list(i.keys())[0]
            if key == "done" and i[key]["success"] == True:
                data=i[key]["text"]
                st.success(data)
            elif key == "done" and i[key]["success"] == False:
                st.error("Test failed:")
                st.error("Please check your code and retry the test, as the website currently is not working as expected.")
            elif key != "replace_file_str":
                with st.expander(key):
                    reg_data=i[key]
                    st.write(reg_data)

def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

#For embedding code
if True:
    st.set_page_config(layout="wide")
    st.markdown(
        """
        <style>
        html, body, [class*="st-"] {
            font-size: 20px; 
        }
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    img_path = "./octagon2.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### LLMを用いた自動化された動的ソフトウェアテスト
    """)

    st.title("オクタゴンテスター🚀")
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
