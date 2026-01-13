import streamlit as st
import asyncio
import os
import sys
from rich.console import Console
import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
from AI_pipeline_general import Langgraph
import yaml

config_path = os.path.join(project_root, "config.yaml")

CONSOLE = Console()

st.header("MCP agent -> E2E with pipeline")
st.subheader(" ###Configuration### ")

#load the yaml file
try:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader) or {}
except FileNotFoundError:
    config = {}

neo4j_url = st.text_input("Neo4j url", value=config.get("neo4j_uri", "bolt://localhost:7687"))
neo4j_password = st.text_input("Neo4j password", value=config.get("neo4j_password", "password"))
app_location = st.text_input("Website URL", value=config.get("app_location", "http://localhost:5173/"))

if st.button("Save content for later"):
    config["neo4j_uri"]=neo4j_url
    config["neo4j_password"] = neo4j_password
    config["app_location"]= app_location

    with open(config_path, "w") as file:
        yaml.dump(config, file)
    st.success("Configuration saved!")

test_type= st.radio("Test Type",["E2E","Unit"])

#---- sidebar ---- This is for setting all the functions that need to be set into the graph
with st.sidebar:
    st.header("Settings")
    headless= st.checkbox("Run in headless?") #give this functionality
    generate_code = st.checkbox("Generate code as well?")
    max_AI_steps= st.number_input("max AI steps",step=1,min_value=0,value=15)
    similarity_k = st.number_input("Number of k nearest nodes for graphRAG",value=20)
    st.caption("AI models. Only write AI models that are known or there will be errors")
    neo4j_ai_model= st.text_input(" AI model to choose that searches database.",value="gemini-2.0-flash")
    tester_ai = st.text_input("AI model for doing the browser usage",value="gemini-2.5-flash")
    code_generator_ai=st.text_input("AI model for generating script code",value="gemini-2.5-flash")
    auto_mode= st.checkbox(" Run in auto - mode")
    st.caption("Automode means there will be no human interaction, thus everything will run in one go. Only use when you trust the AI")


# --- session state ----
if "agent" not in st.session_state:
    st.session_state.agent=Langgraph(
        test_type=test_type,
        neo4j_url=neo4j_url,
        neo4j_pwd=neo4j_password,
        app_address=app_location,
        max_AI_steps=max_AI_steps,
        headless=headless,
        similarity_k=similarity_k,
        neo4j_ai_model=neo4j_ai_model,
        tester_ai=tester_ai,
        code_generator_ai=code_generator_ai
    )
    st.session_state.thread_id = "run_1"

#--- chat---
if "messages" not in st.session_state:
    st.session_state.messages=[]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
user_input = st.chat_input("What should I test")

#--- execution loop -----
async def run_interaction(input_text = None, resume_data = None):
    agent= st.session_state.agent
    config ={
        "configurable":{
            "thread_id":st.session_state.thread_id
        }}
    initial_state=None
    if input_text:
        initial_state={
            "messages":[("user",input_text)],
            "filename":"default.py"
        }
    if resume_data:
        agent.graph.update_state(config, resume_data)
    breakpoints = ["Tester","generate"]

    status_text = st.empty()
    
    with status_text.status("Agent Running....", expanded=True) as s:
        async for event in agent.graph.astream(
            initial_state,
            config=config,
            interrupt_before=breakpoints
        ):
            for node,output in event.items():
                s.write(f"Completed: **{node}**")
                if "messages" in output:
                    msg = output["messages"][-1]
                    content = msg.content if hasattr(msg,"content") else msg[1]
                    st.session_state.messages.append({
                        "role":"assistant",
                        "content":content
                    })
                    with st.chat_message("assistant"):
                        st.write(content)

    status_text.empty()
    st.rerun()

if user_input:
    st.session_state.messages.append({
        "role":"user",
        "content":user_input
    })
    asyncio.run(run_interaction(input_text=user_input))
# --- handle pauses ---
snapshot = st.session_state.agent.graph.get_state(
    {"configurable":{
        "thread_id":st.session_state.thread_id
    }})
if snapshot.next:
    next_step = snapshot.next[0]
    if next_step == "Tester":
        if auto_mode:
            asyncio.run(run_interaction(resume_data={}))
        else:
            st.info("Plan created, Review above")
            st.warning("Ready to launch browser test?")

            instructions= snapshot.values["instructions"]
            st.info("Here is the instructions, you can change the instructions before sent to the automatic AI tester")
            new_instructions= st.text_input("Write here",value=instructions)
            col1,col2 = st.columns(2)
            if col1.button("Run test"):
                asyncio.run(run_interaction(resume_data={"new_instructions":new_instructions})) #set new instructions
            if col2.button("Abort"):
                st.stop()

    elif next_step == "generate":
        st.success("Test execution finished")
        timestamp = datetime.datetime.now()
        unique_filename = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        if auto_mode:
            new_filename=f"{test_type}_{unique_filename}.py"
            asyncio.run(run_interaction(resume_data={"filename":new_filename}))
        else:
            if generate_code:
                new_filename= st.text_input("Save python code as:", value=f"{test_type}_{unique_filename}.py")
                col3,col4 = st.columns(2)
                if col3.button("Generate Code"):
                    asyncio.run(run_interaction(resume_data={"filename":new_filename}))
            if col4.button("Abort") or not generate_code:
                st.stop()






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




    

