"""
streamlit app -> add repo feature -> use ngrok -> add webhook -> test
"""

import os
from flask import Flask, request, jsonify

app = Flask(__name__)

WEBHOOK_SECRET = ""

@app.route('/webhook', methods=['POST'])
def use_webhook():
    event_type = request.headers.get('X-Github-Event')

    if event_type == 'push':
        payload = request.json
        repo_name = payload['repository']['full_name']
        branch = payload['ref'].split('/')[-1]

        print(f"Received push from {repo_name} on branch {branch}")

        # Trigger graph script here

        return jsonify({"status": "processed"}), 200
    
    return jsonify({"status": "ignored"}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)