import streamlit as st
import asyncio
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

def new():
    try: 
        with open("./config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config_path=config.get("victim_path")
        import_path = config.get("neo4j_import_path")
    except FileNotFoundError:
        st.warning("File not found: config.yaml")


    st.title("Octagon tester")
    st.divider()

    with st.form("path_config_form"):
        st.subheader("Update System Path")
        new_path = st.text_input(
            "Absolute path for the system you want to make a graph for", 
            value=config_path # type: ignore
        )
        new_import = st.text_input(
            "Absolute path for the system you want to make a graph for", 
            value=import_path # type: ignore
        )        
        submitted = st.form_submit_button("Save and Get Started")

        if submitted:
            if new_import:
                config["neo4j_import_path"] = new_import
            if new_path:
                config["victim_path"] = new_path # type: ignore
            else:
                st.error("Please enter a valid path/import before proceeding.")
            
            if new_import or new_path:
                with open("./config.yaml", "w") as f:
                        yaml.dump(config, f) # type: ignore
                st.success("Path saved successfully!")
                st.switch_page("pages/cpg.py")
    
    if st.button("Get Started (old)"):
        st.switch_page("pages/codebase.py")
    if st.button("Get Started (new)"):
        st.switch_page("pages/cpg.py")

    

if __name__ == "__main__":
    new()