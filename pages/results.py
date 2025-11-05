import streamlit as st
from streamlit_option_menu import option_menu
import os

result_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
directories=[item for item in os.listdir(result_path)
             if os.path.isdir(os.path.join(result_path,item))
            ]

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
    
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write("Number of tests run: ")
        st.subheader(num_tests)
    with c2:
        st.write("Number of tests passed: ")
        st.subheader(passed)
    with c3:
        st.write("Number of tests failed ")
        st.subheader(fails)


    result_files=[]
    with st.sidebar:
        with st.sidebar:

            option=st.selectbox("Which file to test?",
                (directories))
                
            if option:
                path = os.path.join(result_path,option)
                if os.path.exists(path):
                    items=os.listdir(path)
                    for item in items:
                        full_item_path = os.path.join(path, item)
                        if os.path.isfile(full_item_path):
                            result_files.append(item)

        preview = st.radio(
            "Choose a test to preview",
            [item for item in result_files],
            index=None,
        )

    if preview:
        try:
            file_path = os.path.join(result_path, path, preview)
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