import streamlit as st
import asyncio
import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
import datetime
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
from AI_pipeline_general import Langgraph
import yaml
from streamlit_float import *
import pandas as pd
import re


config_path = os.path.join(project_root, "config.yaml")
csvs_path= os.path.join(project_root,"tests","csv_s")



CONSOLE = Console()

st.header("MCP agent -> E2E with pipeline")
st.subheader(" ###Configuration### ")

#TODO: add neo4j database name

#load the yaml file
try:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader) or {}
        CONSOLE.print("[green] read file [/green]")
except FileNotFoundError:
    config = {}

#floating button if crash
float_init()
button_container = st.container()
with button_container:
    if st.button("Refresh (if error)"):
        st.rerun()    
    float_parent(css="position: fixed; bottom: 10px; right: 100px; z-index: 99999;")

neo4j_url = st.text_input("Neo4j url", value=config.get("neo4j_uri", "bolt://localhost:7687"))
neo4j_password = st.text_input("Neo4j password", value=config.get("neo4j_password", "password"))
neo4j_databse_name = st.text_input("Neo4j databse name",value=config.get("neo4j_database","neo4j"))
app_location = st.text_input("Website URL", value=config.get("app_location", "http://localhost:5173/"))

if st.button("Save content for later"):
    config["neo4j_uri"]=neo4j_url
    config["neo4j_password"] = neo4j_password
    config["app_location"]= app_location

    with open(config_path, "w") as file:
        yaml.dump(config, file)
    st.success("Configuration saved!")

test_type= st.radio("Test Type",["E2E","Parameter"])

if test_type == "Parameter":
    """
    the column names are sent to the database or the acc database sent there and all expected values are there

    """
    st.info("CSV format should be as follows: parameter, expected_response_(paramater_name)")
    
    csvfile=st.file_uploader("Upload csv values for parameter testing",type="csv")
    file_name=st.text_input("File name to be saved as?")
    if st.button("Save File"):
        if csvfile is not None and file_name:
            if not os.path.exists(csvs_path):
                os.makedirs(csvs_path)
            
            path = os.path.join(csvs_path, file_name + ".csv")
            
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(csvfile.getvalue())
                st.success(f"Saved successfully to {path}")
                
                # Extract column metadata
                df = pd.read_csv(path)
                column_names = df.columns.tolist()
                input_columns = [col for col in column_names if not re.match(r"^expected_response", col)]
                expected_columns = [col for col in column_names if re.match(r"^expected_response", col)]
                
                # Store in session state
                st.session_state.csv_file_name = file_name
                st.session_state.csv_path = path
                st.session_state.input_columns = input_columns
                st.session_state.expected_columns = expected_columns
                
                st.info(f"Input columns: {', '.join(input_columns)}")
                st.info(f"Expected result columns: {', '.join(expected_columns)}")
                
                # Force agent recreation with new metadata
                if "agent" in st.session_state:
                    del st.session_state.agent
                    st.warning("Agent will be recreated with CSV metadata on next run")
            else:
                st.warning("A file already has the same name")

        elif csvfile is None:
            st.error("Please upload a CSV file first.")
        else:
            st.error("Please enter a file name.")

    file=st.selectbox("Select file",os.listdir(csvs_path))  
            
    if file and st.button("Choose file"):
        path=os.path.join(csvs_path, file)

        # Extract column metadata
        df = pd.read_csv(path)
        column_names = df.columns.tolist()
        input_columns = [col for col in column_names if not re.match(r"^expected_response", col)]
        expected_columns = [col for col in column_names if re.match(r"^expected_response", col)]
        
        st.session_state.csv_file_name = os.path.splitext(file)[0]
        st.session_state.csv_path = path
        st.session_state.input_columns = input_columns
        st.session_state.expected_columns = expected_columns
        
        st.info(f"Input columns: {', '.join(input_columns)}")
        st.info(f"Expected result columns: {', '.join(expected_columns)}")
        
        if "agent" in st.session_state:
            del st.session_state.agent
            st.warning("Agent will be recreated with CSV metadata on next run")

#---- sidebar ---- This is for setting all the functions that need to be set into the graph
with st.sidebar:
    st.header("Settings")
    headless= st.checkbox("Run in headless?") #give this functionality
    if headless:
        st.success(f"Headless on")
    else:
        st.warning("Headless off")
    generate_code = st.checkbox("Generate code as well?")
    if generate_code:
        st.info("Generate code on")
    else:
        st.warning("Generate code off")

    max_AI_steps= st.number_input("max AI steps",step=1,min_value=0,value=15)
    similarity_k = st.number_input("Number of k nearest nodes for graphRAG",value=20)
    st.subheader("AI models")
    neo4j_ai_model= st.text_input(" AI model to choose that searches database.",value="gemini-2.0-flash")
    tester_ai = st.text_input("AI model for doing the browser usage",value="gemini-2.5-flash")
    code_generator_ai=st.text_input("AI model for generating script code",value="gemini-2.5-flash")
    st.caption("Only write AI models that are known or there will be errors")
    auto_mode= st.checkbox(" Run in auto - mode")
    st.caption("Automode means there will be no human interaction, thus everything will run in one go. Only use when you trust the AI")
if st.sidebar.button("Update / Reset Agent"):
    if "agent" in st.session_state:
        del st.session_state.agent
    st.success("Agent settings updated!")

# --- session state ----
if "agent" not in st.session_state:
    # Get CSV metadata if available (for Parameter testing)
    csv_columns = st.session_state.get("input_columns", [])
    csv_path = st.session_state.get("csv_path", None)
    
    # For Parameter mode, require CSV to be loaded before creating agent
    if test_type == "Parameter" and not csv_path:
        st.warning("Please upload or select a CSV file for Parameter testing before proceeding.")
    else:
        st.session_state.agent=Langgraph(
            test_type=test_type,
            neo4j_url=neo4j_url,
            neo4j_pwd=neo4j_password,
            neo4j_database=neo4j_databse_name,
            app_address=app_location,
            max_AI_steps=max_AI_steps,
            headless=headless,
            similarity_k=similarity_k,
            neo4j_ai_model=neo4j_ai_model,
            tester_ai=tester_ai,
            code_generator_ai=code_generator_ai,
            columns=csv_columns,
            csv_path=csv_path

        )
        st.session_state.thread_id = "run_1"

#debugging
params = {
    "Test Type": test_type,
    "Noe4j database": neo4j_databse_name,
    "Neo4j URL": neo4j_url,
    "Neo4j Pwd": "*****" if neo4j_password else "None", 
    "App Address": app_location,
    "Max AI Steps": max_AI_steps,
    "Headless": headless,
    "Similarity K": similarity_k,
    "Neo4j AI Model": neo4j_ai_model,
    "Tester AI": tester_ai,
    "Code Gen AI": code_generator_ai
}
param_str = "\n".join([f"[b]{k}:[/b] {v}" for k, v in params.items()])
CONSOLE.print(Panel(param_str, title="Langgraph Params", expand=False))


#--- chat---
if "messages" not in st.session_state:
    st.session_state.messages=[]
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if test_type == "Parameter":
    user_input = st.chat_input("Describe your test")
else:
    user_input = st.chat_input("What should I test")

#--- execution loop -----
async def run_interaction(input_text = None, resume_data = None):
    agent= st.session_state.agent
    config ={
        "configurable":{
            "thread_id":st.session_state.thread_id
        }}
    CONSOLE.print(Panel(Pretty(config),title="config"))
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
    # Validate CSV is loaded for Parameter tests
    if test_type == "Parameter":
        csv_path = st.session_state.get("csv_path", None)
        if not csv_path:
            st.error("No CSV file loaded. Please upload a CSV file or select one from the dropdown, then click the 'Choose file' button before running the test.")
            st.stop()
        
        # Force agent recreation if it doesn't have the csv_path
        if "agent" in st.session_state and st.session_state.agent.csv_path != csv_path:
            del st.session_state.agent
            st.rerun()
    
    # Ensure agent exists before running
    if "agent" not in st.session_state:
        st.error("Agent not initialized. Please check your configuration and try again.")
        st.stop()
    
    st.session_state.messages.append({
        "role":"user",
        "content":user_input
    })
    asyncio.run(run_interaction(input_text=user_input))

# --- handle pauses ---
if "agent" in st.session_state:
    snapshot = st.session_state.agent.graph.get_state(
        {"configurable":{
            "thread_id":st.session_state.thread_id
        }})
    CONSOLE.print("[yellow] Snapshot taken [/yellow]")
    if snapshot.next:
        CONSOLE.print(f"[yellow] Snapshot next:/[yellow] {snapshot.next[0]}")
        next_step = snapshot.next[0]
        if next_step == "Tester":
            

            CONSOLE.print("[bold green] Tester mode on [/bold green]")
            timestamp = datetime.datetime.now()
            unique_filename = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
            # Use agent's test_type for consistency with where file will be saved
            agent_test_type = st.session_state.agent.test_type
            new_filename=f"{agent_test_type}_{unique_filename}.py"
            if auto_mode:
                CONSOLE.print("[magenta] auto mode ON [/magenta]")
                asyncio.run(run_interaction(resume_data={"filename":new_filename}))
            else:
                CONSOLE.print("[magenta] auto mode OFF [/magenta]")
                st.info("Plan created, Review above")
                st.warning("Ready to launch browser test?")

                instructions= snapshot.values["instructions"]
                st.info("Here are the instructions, you can change the instructions before sent to the automatic AI tester")
                new_instructions= st.text_area("Write here",value=instructions,height="content")

                if agent_test_type != "Parameter":
                    new_filename= st.text_input("Save python code as:", value=f"{agent_test_type}_{unique_filename}.py")
                col1,col2 = st.columns(2)
                if col1.button("Run test"):
                    CONSOLE.print(Panel(new_instructions,title="instructions"))
                    
                    asyncio.run(run_interaction(resume_data={"new_instructions":new_instructions,
                                                             "filename":new_filename})) #set new instructions and filename
                if col2.button("Abort"):
                    st.stop()
                    st.rerun()


        
        elif next_step == "generate":
            
            CONSOLE.print("[bold green] Generate mode on [/bold green]")
            st.success("Test execution finished")
            timestamp = datetime.datetime.now()
            unique_filename = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
            # Use agent's test_type for consistency
            agent_test_type = st.session_state.agent.test_type
            new_filename=f"{agent_test_type}_{unique_filename}.py"
            if agent_test_type == "Parameter":
                csv_filename = st.session_state.get("csv_file_name", None)
                generate_code=True

                if csv_filename:
                    new_filename = csv_filename + ".py"

                else:
                    st.error("No CSV file loaded. Please upload a CSV file or select one from the dropdown, then click the 'Choose file' button before running the test.")
                    st.stop()

            if auto_mode:
            
                asyncio.run(run_interaction(resume_data={"filename":new_filename}))
            
            else:
            
                col3,col4 = st.columns(2)
                if generate_code:
            
                    if agent_test_type != "Parameter":
            
                        new_filename= st.text_input("Save python code as:", value=f"{agent_test_type}_{unique_filename}.py")
            
                    if col3.button("Generate Code"):
            
                        asyncio.run(run_interaction(resume_data={"filename":new_filename}))
            
                if col4.button("Abort") or not generate_code:
                    st.stop()
                    st.rerun()

                #TODO: Only write the AIs newest point
                #TODO: add gemini thinking streaming
                #TODO: how to fix if nothing found then repeat in vector search