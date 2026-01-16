import streamlit as st
import os
import subprocess
import sys

st.header("Parameter Testing")

# Get correct paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csvs_path = os.path.join(project_root, "tests", "csv_s")
codes_path = os.path.join(project_root, "tests", "codeblock","param")

# Check if directories exist
if not os.path.exists(csvs_path):
    st.error(f"CSV directory not found: {csvs_path}")
    st.info("Please upload a CSV file first in the main test page.")
    st.stop()

# List available CSV files
csv_files = [f for f in os.listdir(csvs_path) if f.endswith('.csv')]
if not csv_files:
    st.warning("No CSV files found. Please upload a CSV first in the main test page.")
    st.stop()

# Select CSV file
choice = st.selectbox("Choose your parameter testing CSV file", csv_files)

# Find corresponding Python file
py_filename = choice.replace(".csv", ".py")
final_code = os.path.join(codes_path, py_filename)

# Display info
st.info(f"CSV file: `{choice}`")
st.info(f"Expected test script: `{py_filename}`")

# Run test button
if st.button("Run Parameter Tests"):
    if not os.path.exists(codes_path):
        st.error(f"Test code directory not found: {codes_path}")
        st.info("Please generate the test code first using the main test page.")
    elif not os.path.exists(final_code):
        st.error(f"Test file not found: `{py_filename}`")
        st.info("Please generate the test code first by:")
        st.markdown("""
        1. Go to the main test page
        2. Select "Parameter" test type
        3. Upload your CSV file
        4. Describe the test you want to run
        5. Let the AI generate the script
        """)
    else:
        st.info(f"Running: {py_filename}")
        
        # Create a placeholder for output
        output_placeholder = st.empty()
        
        with st.spinner("Executing tests..."):
            result = subprocess.run(
                [sys.executable, final_code],
                capture_output=True,
                text=True,
                cwd=project_root
            )
        
        # Display results
        st.subheader("Test Output:")
        if result.stdout:
            st.code(result.stdout, language="text")
        
        if result.returncode != 0:
            st.error("❌ Test execution failed!")
            if result.stderr:
                st.subheader("Error Details:")
                st.code(result.stderr, language="text")
        else:
            st.success("✅ Test execution completed!")

# Additional info section
with st.expander("ℹ️ How to use Parameter Testing"):
    st.markdown("""
    ### Parameter Testing Workflow
    
    1. **Create CSV file** with your test data:
       - Input columns: `email`, `password`, etc.
       - Expected result columns: `expected_response_email`, `expected_response_password`, etc.
    
    2. **Upload CSV** in the main test page:
       - Select "Parameter" test type
       - Upload your CSV file
       - Save it with a name
    
    3. **Generate test script**:
       - Describe what the test should do
       - Let the AI explore the website and generate the script
       - The script will automatically loop through all CSV rows
    
    4. **Run tests** here:
       - Select your CSV file
       - Click "Run Parameter Tests"
       - View results for each row
    """)