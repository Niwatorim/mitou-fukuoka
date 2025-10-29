import streamlit as st
from streamlit_option_menu import option_menu
import os

result_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

c1,c2,c3 = st.columns(3)

try:
    selected = option_menu(
            menu_title=None, 
            options=["Add Codebase", "Run Tests", "Test Results"],  
            menu_icon="cast", 
            default_index=2,  
            orientation="horizontal",
        )
    if selected== "Add Codebase":
        st.switch_page("pages/codebase.py")
    if selected == "Run Tests":
        st.switch_page("pages/tests.py")

    if "results" not in st.session_state:
        st.session_state.results=[]

    tests=st.session_state.results
    num_tests=len(tests)
    passed = len([item for item in tests if item["success"] == True])
    fails = len([item for item in tests if item["success"] == False])

    if num_tests == 0 or num_tests == None:
        st.warning("Run some tests first")
    
    with c1:
        st.write("Number of tests run: ",num_tests)
    with c2:
        st.write("Number of tests passed: ",passed)
    with c3:
        st.write("Number of tests failed ",fails)


    result_files=[]
    if os.path.exists(result_path):
        items=os.listdir(result_path)
        for item in items:
            result_files.append(item)

    preview = st.radio(
        "Choose a test to preview",
        [item for item in result_files],
        index=None,
    )

    if preview:
        try:
            file_path = os.path.join(result_path, preview)
            if os.path.exists(file_path):
                with open(file_path,"r") as f:
                    data=f.read()
                st.code(data, language='yaml')
            else:
                st.error(f"File not found: {file_path}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

except Exception as e:
    st.warning(f"{e}")