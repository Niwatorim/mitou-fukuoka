import os
import shutil
import git
import json
import stat  
import jwt
import time
import sys
import requests
from flask import Flask, request, jsonify
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import embed_ast, ast_rag, graph_creation
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("APP_ID")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
PRIVATE_KEY_PATH = os.path.join(root_dir, "private-key.pem")

DB_FILE = "installations.json"

app = Flask(__name__)

def get_installation_access_token(installation_id):
    """
    Create a JWT using the private key. 
    Exchange JWT for an installation token from Github API
    """
    with open(PRIVATE_KEY_PATH, 'rb') as key_file:
        private_key = key_file.read()
    
    # create JWT
    payload = {
        'iat': int(time.time()),
        'exp': int(time.time()) + (10*60),
        'iss': APP_ID
    }
    encoded_jwt = jwt.encode(payload, private_key, algorithm='RS256')

    # request token
    headers = {
        "Authorization": f"Bearer {encoded_jwt}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers=headers
    )

    if response.status_code == 201:
        return response.json()['token']
    else:
        raise Exception(f"Failed to get token: {response}")

def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file),
    it changes the file to be writable and attempts the function again.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

# AST, graph and embedding updated every time user push code 
@app.route('/webhook', methods=['POST'])
def handle_webhook():
    event_type = request.headers.get('X-GitHub-Event')
    data = request.json

    # when user just installed the github app
    if event_type == 'installation':
        action = data.get('action')
        if action in ['created', 'added']: 
            installation_id = data['installation']['id']
            db = load_db()
            for repo in data['repositories']:
                full_name = repo['full_name']
                name = repo['name']
                db[full_name] = installation_id
                process_repo(full_name, name, installation_id)
            save_db(db)
            print(f"Saved new installation: {installation_id}")
        
    # when user updates repo
    elif event_type == 'push':
        repo_full_name = data['repository']['full_name']
        repo_name = data['repository']['name']
        installation_id = data['installation']['id']
        process_repo(repo_full_name, repo_name, installation_id)
    
    return jsonify({"status": "received"}), 200

def process_repo(full_name, repo_name, installation_id):
    # get token and clone
    token = get_installation_access_token(installation_id)
    clone_url = f"https://x-access-token:{token}@github.com/{full_name}.git"
    
    temp_dir = os.path.join(root_dir, "temp_clones", repo_name)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, onerror=on_rm_error)
    
    # create graph and embeddings
    try:
        git.Repo.clone_from(clone_url, temp_dir)
        print(f"Cloned {repo_name} successfully")

        print("Processing files..")

        # loop all files in directory
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".jsx"):
                    file_path = os.path.join(root, file)

                    try:

                        # print(f"Creating embeddings for: {relative_path}..")
                        # embed_ast(file_path)

                        print(f"Updating graph for: {file_path}..")
                        graph_creation(file_path)

                    except Exception as e:
                        print(f"Error in processing file {file}: {e}")
                        continue

        print(f"Pipeline process finished for {repo_name}")
    
    except Exception as e:
        print(f"Pipeline failed: {e}")
    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, onerror=on_rm_error)

@app.route('/check_installation', methods=['GET'])
def check_installation():
    repo_name = request.args.get('repo')
    db = load_db()

    if repo_name in db:
        return jsonify({"installed": True, "installation_id": db[repo_name]}), 200
    else:
        return jsonify({"installed": False}), 200

@app.route('/list_repos', methods=['GET'])
def list_repos():
    db = load_db()
    return jsonify(list(db.keys())), 200


if __name__ == '__main__':
    app.run(port=5000, debug=True)