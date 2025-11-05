import os
import sys
import streamlit as st
from streamlit_option_menu import option_menu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import cycle,test_browser_use,results_writer
import json
import asyncio

test_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
directories=[item for item in os.listdir(test_path)
             if os.path.isdir(os.path.join(test_path,item))
            ]

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


test_files=[]
with st.sidebar:

    option=st.selectbox("Which file to test?",
        (directories))
        
    if option:
        path = os.path.join(test_path,option)
        if os.path.exists(path):
            items=os.listdir(path)
            for item in items:
                full_item_path = os.path.join(path, item)
                if os.path.isfile(full_item_path):
                    test_files.append(item)

        preview = st.radio(
            "Choose a test to preview",
            test_files,
            index=None,
        )
    else: st.warning("no tests made, please run tests first")
    

st.divider()
st.subheader("Preview test results")

if preview:
    try:
        file_path = os.path.join(test_path, option, preview)
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
file_to_test=st.selectbox("Which file to test?",
    (directories))
headless=st.toggle("Run in headless",value=True)

c1,c2 = st.columns(2)

with c2:
    use_limit = st.checkbox("Limit the number of tests?")

    number_of_tests = None

    # If the checkbox is checked, show the number input and assign its value
    if use_limit:
        number_of_tests = st.number_input(
            "How many tests to run?", 
            min_value=1,
            max_value=len(os.listdir(os.path.join(test_path,file_to_test))),
            step=1, 
            value=1  # A sensible default
        )
    
with c1:
    if st.button("Run testing agent"):
        try:
            results=asyncio.run(test_browser_use(number_of_tests,headless,file_to_test))
            # results=[
            #     {
            #         "path":"App.jsx",
            #         "name":"a[1].yaml",
            #         "success":True
            #     },
            #     {
            #         "path":"App.jsx",
            #         "name":"a.yaml",
            #         "success":True
            #     },
            #     {
            #         "path":"NotApp.jsx",
            #         "name":"a.yaml",
            #         "success":False
            #     }
            # ]
            st.session_state.results=results
            for i in results:
                st.write(str(i))
            asyncio.run(results_writer(results))
            st.success("Completed test execution")
        except Exception as e:
            st.warning(f"Error followed in running agent: {e}")
