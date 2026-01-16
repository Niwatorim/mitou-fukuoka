"""
All processes are in the Flask backend
"""
import sys
import os
import time
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

repo_name = None
repo_connected = False

if "codebase" not in st.session_state:
    st.session_state.codebase=False

def load_ast(filepath):
    try:
        mtime = os.path.getmtime(filepath)

        if "ast_mtime" not in st.session_state or st.session_state.ast_mtime != mtime:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            st.session_state.ast_data = data
            st.session_state.ast_mtime = mtime
            st.success(f"Codebase data updated! (Timestamp: {time.ctime(mtime)})")
            return data
        
        return st.session_state.ast_data
    
    except (FileNotFoundError, json.JSONDecodeError):
        return None

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

try:
    response = requests.get(f"{FLASK_NGROK_URL}/list_repos")
    available_repos = response.json()
except:
    available_repos = []

if available_repos:
    selected_repo = st.radio("Select a repo:", available_repos)
    if st.checkbox("Add a new Github repo"):
        repo_url = st.text_input("Enter a new Github repo URL")
    else:
        if selected_repo:
            repo_url = f"https://github.com/{selected_repo}"
else:
    repo_url = st.text_input("Enter a Github repo URL")

if repo_url:
    repo_name = parse_repo_url(repo_url)

    if repo_name:
        st.info(f"Checking installation status for : {repo_name}")

        try:
            response = requests.get(f"{FLASK_NGROK_URL}/check_installation", params={"repo": repo_name})
            if response.status_code == 200:
                result = response.json()

                if result.get("installed"):
                    st.success("Repository is connected!")
                    st.session_state['current_repo'] = repo_name
                    st.session_state['installation_id'] = result['installation_id']
                    repo_connected = True
                
                else:
                    st.warning("Access missing. Please install the Github App.")
                    
                    # 2. Show the link
                    st.markdown(f"[**Click here to install the Github App**]({GITHUB_APP_INSTALLATION_LINK})")
                    st.caption("After installing, come back here and check the box below.")
                    if st.button("I have installed the App"):
                         st.rerun()
            else:
                st.error("Could not connect to backend server.")
                
        except requests.exceptions.ConnectionError:
            st.error("Backend Flask server is not running!")
    else:
        st.error("Invalid GitHub URL format")

st.divider()



    