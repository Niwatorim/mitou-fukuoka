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

#TODO: put mcp.json outside the pages folder

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


st.header("MCP agent -> E2E with pipeline")

choice = st.radio(
    options=["E2E","regular"],
    index=0
)

st.subheader(" ###Configuration### ")

#TODO: Make this so that every time they write a new one, it saves in the config file, so that it can just reload that and they dont have to repeat

neo4j_url = st.text_input("Neo4j url",value="bolt://localhost:7687")
neo4j_password= st.text_input("Neo4j password",value="password")
app_location = st.text_input("Website URL",value="http://localhost:5173/")
max_AI_steps = st.number_input("Automatic AI tester max steps",step=1)

with st.container():
    st.write("Automatic mode")
    st.checkbox("Run in headless?") #give this functionality


#TODO: for app location, make it affect the AI general pipeline so it is dynamic, rn hardcoded

#TODO: Make it so the user chooses manual mode, or auto mode. In manual, they review everything and click check
#in auto mode, state holds "auto" so that it clicks yes to everything or makes a selection to generate or not generate code


"""
selection options:
Vector Search Node
neo4j setup: url = bolt://localhost:7687
             password = "password"

--done

MCPGraph:
app opening location-> http://localhost:5173/ 

--done

Show the instructions-done

Check if user wants to go ahead with the test, -done
or edit the instructions


MCPTester
Check if user wants to generate code - done
Show the code written, ask if they wanna rename the test file name - done
also number of steps AI can take before calling it ggs-done

and terminate button

and headless mode

"""

user=st.text_input(label="user-query",value="Please tell me how many nodes are in this graph")





    

