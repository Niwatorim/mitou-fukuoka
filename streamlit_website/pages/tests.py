import os
import sys
import streamlit as st
from streamlit_option_menu import option_menu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import cycle,test_browser_use,results_writer
import json
import asyncio

test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
test_files=[]

selected = option_menu(
        menu_title=None,
        options=["Add Codebase", "Run Tests", "Test Results"],
        menu_icon="cast",
        default_index=1,
        orientation="horizontal",
    )

if selected== "Test Results":
    st.switch_page("pages/results.py")
if selected == "Add Codebase":
    st.switch_page("pages/codebase.py")

st.title("Run tests")

if st.button("Run test"):
    try:
        cycle(test_path) #creates directory of test_path and stores test files in there in yaml format
        st.success("Tests generated successfully!")
    except Exception as e:
        st.error(f"Test generation failed: {str(e)}")

if os.path.exists(test_path):
    items=os.listdir(test_path)
    for item in items:
        test_files.append(item)

preview = st.radio(
    "Choose a test to preview",
    [item for item in test_files],
    index=None,
)

if preview:
    try:
        file_path = os.path.join(test_path, preview)
        if os.path.exists(file_path):
            with open(file_path,"r") as f:
                data=f.read()
            st.code(data, language='yaml')
        else:
            st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

if "results" not in st.session_state:
    st.session_state.results=[]

st.divider()
st.subheader("Sub-agent")
headless=st.toggle("Run in headless")
if st.button("Run testing agent"):
    try:
        # results=asyncio.run(test_browser_use(2,headless))
        results=[
            {
                "name":"a[1].yaml",
                "success":True
            },
            {
                "name":"a.yaml",
                "success":True
            }
        ]
        st.session_state.results=results
        for i in results:
            st.write(str(i))
        asyncio.run(results_writer(results))
    except Exception as e:
        st.warning(f"Error followed in running agent: {e}")
