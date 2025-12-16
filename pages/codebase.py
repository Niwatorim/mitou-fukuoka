import sys
import os
# Add parent directory to path to import functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_agraph import agraph, Node, Edge, Config
from functions import ast_rag,embed_ast,get_graph, graph_creation, parse_repo_url
import json
from dotenv import load_dotenv
import requests

load_dotenv()

FLASK_NGROK_URL = os.getenv('FLASK_NGROK_URL')
GITHUB_APP_INSTALLATION_LINK = os.getenv('GITHUB_APP_INSTALLATION_LINK')

if "codebase" not in st.session_state:
    st.session_state.codebase=False

st.title("Add codebase")
selected = option_menu(
        menu_title=None,  # required
        options=["Add Codebase", "Run Tests", "Test Results"],  # required
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
    )

if selected== "Run Tests":
    st.switch_page("pages/tests.py")
if selected == "Test Results":
    st.switch_page("pages/results.py")

file = st.text_input("upload file") # github URL now

"""
add code here to handle github repo URL
check if there's installation ID
if yes -> update graph when user push the code
if no -> shows the github app installation link in the UI
"""

st.subheader("Github Repository Setup")
repo_url = st.text_input("Enter Github Repository URL")

if repo_url:
    repo_name = parse_repo_url(repo_url)

    if repo_name:
        st.info(f"Checking installation status for : {repo_name}")

        try:
            response = requests.get(f"{FLASK_NGROK_URL}/check_installation", params={"repo": repo_name})
            if response.status_code == 200:
                result = response.json()

                if response.get("installed"):
                    st.success("Repository is connected!")
                    st.session_state['current_repo'] = repo_name
                    st.session_state['installation_id'] = result['installation_id']
                    # if already installed, if want to use the program again, no need to paste the github repo link again, the user can just push the code, and the graph shown in streamlit UI is updated
                
                else:
                    st.warning("Access missing. Please install the Github App.")
                    st.markdown(f"[**Click here to install the Github App**]({GITHUB_APP_INSTALLATION_LINK})")
                    st.caption("After installing, come back here and click the URL box again.")
            else:
                st.error("Could not connect to backend server.")
                
        except requests.exceptions.ConnectionError:
            st.error("Backend Flask server is not running!")
    else:
        st.error("Invalid GitHub URL format")

c1,c2=st.columns(2,gap="small")
with c1:
    if st.button("Create AST-structure"):
        try:
            if file and os.path.exists(file):
                storage={}
                data_string = ast_rag(file)
                data = json.loads(data_string)
                name=os.path.basename(data["file"])
                storage[name]=data
                code_struct_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code_structure.json")
                with open(code_struct_path,"w") as f:
                    json.dump(storage,f,indent=4)
                st.session_state.codebase=True
                st.success("Code structure parsed successfully!")
            else:
                st.error("Please provide a valid file path")
        except Exception as e:
            st.warning(f"Failed upload: {str(e)}")

with c2:
    if st.button("Start embedding process"):
        try:
            if file and os.path.exists(file):
                embed_ast(file)
                st.success("Embedding completed successfully!")
            else:
                st.error("Please provide a valid file path first")
        except Exception as e:
            st.error(f"Embedding failed: {str(e)}")

st.subheader(" Graph ")
st.divider()

if st.button("Create a Graph"):
    if not file:
        st.warning("Please select a file first")
    else:
        try:
            graph_creation(file)
            st.success("Graph created successfully ")
        except Exception as e:
            st.warning(f"Error in creating graph {e}")

if st.button("Generate graph"):
    nodes_data,edges_data,node_types=get_graph()
    if nodes_data:
        nodes=[]
        edges=[]

        for i in nodes_data:
            display=i["properties"].get("name",i["id"])

            node_type=i["properties"].get("type")
            n_color=node_types.get(node_type,"#42D4F5")

            nodes.append(Node(
                id=i["id"],
                label=display,
                size=25,
                shape="dot",
                font={
                    "color":"#FFFFFF",
                    "size":18,
                },
                color=n_color
            ))

        for i in edges_data:
            edges.append(Edge(
                source=i["source"],
                label=i["type"],
                target=i["target"],
                type="CURVE_SMOOTH"
            ))

        config=Config(
            width=750,
            height=950,
            directed=True,
            physics=True,
            # physics={
            #     "enabled": True,
            #      "barnesHut": {
            #          "gravitationalConstant": -20000,
            #          "centralGravity": 0.1,
            #          "springLength": 150,
                    
            #          "damping": 0.09
            #      },
            #     "stabilization": {
            #         "iterations": 1000
            #     }
            # },
            nodeHighlightBehavior=True,
            collapsible=True,
            heirarchial=False
        )

        return_value=agraph(nodes=nodes,edges=edges,config=config)
    else:
        st.warning("No data returned")
