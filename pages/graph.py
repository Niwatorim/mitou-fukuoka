import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import subprocess
import yaml
import shutil
from functions import cpg_to_neo4j

#TODO: Add warning if ollama and neo4j not connected

# Updated to use absolute path or relative to root
CONFIG_PATH = "config.yaml"

st.header("Create into a graph")
if st.button("Create into a graph"):
    try: 
        if not os.path.exists(CONFIG_PATH):
            st.error(f"Config file not found at {os.path.abspath(CONFIG_PATH)}")
            st.stop()
            
        with open(CONFIG_PATH, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        project_path = config.get("victim_path")
    except Exception as e:
        st.error(f"Error loading config: {e}")
        st.stop()

    # Paths relative to project root
    cpg_file_path = "cpg_folder/cpg_creations"
    output_folder = "cpg_folder/victim_cpg_files"

    if os.path.exists(cpg_file_path):
        os.remove(cpg_file_path)
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(os.path.dirname(cpg_file_path), exist_ok=True)

    command1 = ["joern-parse", project_path, "--output", cpg_file_path]
    command2 = ["joern-export", cpg_file_path, "--out", output_folder, "--repr", "all", "--format", "neo4jcsv"]
    
    result1 = subprocess.run(command1, capture_output=True, text=True) #step 1
    if result1.returncode != 0:
        st.error(f"joern-parse failed: {result1.stderr}")
        st.stop()

    result2 = subprocess.run(command2, capture_output=True, text=True) #step 2
    if result2.returncode != 0:
        st.error(f"joern-export failed: {result2.stderr}")
        st.stop()
    
    try:
        cpg_to_neo4j(config=config)
    except Exception as e:
        st.error(f"Failed to push to Neo4j: {e}")