import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import subprocess
import yaml
import shutil
from functions import cpg_to_neo4j

# Updated to use absolute path or relative to root
CONFIG_PATH = "config.yaml"

st.header("Using cpg")
if st.button("Make cpg graph"):
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

    # Define paths (Use absolute paths to avoid confusion)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root of project
    cpg_file_path = os.path.join(base_dir, "cpg_folder", "cpg_creations")
    output_folder = os.path.join(base_dir, "cpg_folder", "victim_cpg_files")

    # Clean up old folders
    if os.path.exists(cpg_file_path):
        os.remove(cpg_file_path)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(os.path.dirname(cpg_file_path), exist_ok=True)

    # Prepare Joern Paths
    joern_base_path = config.get("joern_path")
    exec_parse = "joern-parse.bat" if os.name == 'nt' else "joern-parse"
    exec_export = "joern-export.bat" if os.name == 'nt' else "joern-export"
    
    path_to_parse = os.path.join(joern_base_path, exec_parse)
    path_to_export = os.path.join(joern_base_path, exec_export)

    # COMMAND 1
    command1 = [path_to_parse, project_path, "--output", cpg_file_path]

    # --- NEW: FORCE JAVA HOME ---
    # We copy the current environment and manually add JAVA_HOME
    my_env = os.environ.copy()
    
    # ⚠️ IMPORTANT: Paste your exact JDK path here (no \bin at the end)
    my_env["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21" 
    
    # Add Java to the PATH as well, just in case
    my_env["PATH"] = my_env["JAVA_HOME"] + r"\bin;" + my_env.get("PATH", "")
    # ----------------------------

    joern_work_dir = config.get("joern_path")

    st.write(f"Running command: {' '.join(command1)}")
    st.write(f"Working Directory: {joern_work_dir}")

    result1 = subprocess.run(
        command1, 
        capture_output=True, 
        text=True, 
        env=my_env, 
        cwd=joern_work_dir  # <--- CRITICAL FIX: Run inside the joern folder
    )
    
    if result1.returncode != 0:
        st.error(f"joern-parse failed: {result1.stderr}")
        # Add a hint about Java if it fails here
        st.warning("Hint: If the error says 'path specified', check your JAVA_HOME environment variable.")
        st.stop()

    # COMMAND 2
    command2 = [path_to_export, cpg_file_path, "--out", output_folder, "--repr", "all", "--format", "neo4jcsv"]
    
    result2 = subprocess.run(
        command2, 
        capture_output=True, 
        text=True, 
        env=my_env,
        cwd=joern_work_dir  # <--- CRITICAL FIX
    )
    
    if result2.returncode != 0:
        st.error(f"joern-export failed: {result2.stderr}")
        st.stop()
    
    try:
        cpg_to_neo4j(config=config)
    except Exception as e:
        st.error(f"Failed to push to Neo4j: {e}")