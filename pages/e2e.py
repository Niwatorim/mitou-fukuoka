import streamlit as st
import asyncio
import os
import sys
from rich.console import Console
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from functions import MCPGeminiAgent

CONSOLE = Console()

async def main(prompt,sys_prompt):
    agent = MCPGeminiAgent()
    try:
        await agent.connect()
        data = await agent.chat(prompt,sys_prompt)
        with st.expander("AI response"):
            st.write(data)
    
    
    except Exception as e:
        st.warning(f"Fatal error during execution: {e}")
        traceback.print_exc()
    
    finally:
        await agent.cleanup()


st.header("MCP agent -> prompt based graph query")

choice = st.radio(
    options=["E2E","regular"]
)

system_prompt = None
if choice == "E2E":
    system_prompt = """
            You are a graph-based testing expert. Your goal is to help the user understand and test their application by querying the Neo4j graph database.
            1. If you don't know the structure of the graph or cannot answer the user's question, USE THE TOOLS to explore the nodes and relationships.
            2. For end-to-end testing requests, generate steps to go from node to node in this format:
               Path_exists: True/False
               test_steps:
                   - step: 1
                     action: navigate
                     instruction: ...
                     target: ...
                     expected: ...
            Always prioritize using tools when factual information about the graph is needed."""

user=st.text_input(label="user-query",value="Please tell me how many nodes are in this graph")
if st.button("Send request"):
    if user:
        asyncio.run(main(user,system_prompt))
    else:
        st.warning("Please enter a query.")



    

