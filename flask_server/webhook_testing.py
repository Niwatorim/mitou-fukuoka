"""
Architecture note:
2 ways to use webhook:
1. In production, where user will paste their URL to the streamlit app:
...
2. For testing:
- create the sample repo
- open neo4j docker
- create the flask app
- set up ngrok payload URL
- set up github webhook to sample repo, attach the payload URL
- push sample code to repo
- if succeeded, graph will change 
"""
import os
import shutil
import git
import json
import stat  # <--- NEW IMPORT NEEDED
from flask import Flask, request, jsonify
from functions import get_graph, graph_creation

app = Flask(__name__)

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file),
    it changes the file to be writable and attempts the function again.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

@app.route('/webhook', methods=['POST'])
def use_webhook():
    data = request.json

    repo_url = data['repository']['clone_url']
    repo_name = data['repository']['name']
    
    print(f"Processing update for: {repo_name}")

    # 2. Define a temp path to clone the code
    temp_dir = f"./temp_clones/{repo_name}"
    
    # Clean up old run if exists
    if os.path.exists(temp_dir):
        # UPDATED: Added onerror callback
        shutil.rmtree(temp_dir, onerror=on_rm_error)
        
    # 3. Clone the User's Code locally
    try:
        git.Repo.clone_from(repo_url, temp_dir)
        
        # 4. Run your Graph Update Logic
        # (Make sure your graph script accepts a folder path!)
        # Pass the full path to the file inside the repo
        target_file = os.path.join(temp_dir, "App.jsx") # Construct path safely
        
        # NOTE: Your function expects a file path, so we point it to App.jsx
        if os.path.exists(target_file):
             graph_creation(file_name=target_file)
        else:
             print(f"Warning: App.jsx not found in {temp_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # 5. Cleanup
        # UPDATED: Added onerror callback
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=on_rm_error)

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)