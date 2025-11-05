# Octagon Tester: Automated React Code Analysis & Testing Platform

## 🚀 Overview
Octagon Tester is a full-stack platform for automated parsing, analysis, and testing of React codebases. It provides a seamless workflow:
- **Parse** frontend source code into ASTs
- **Embed** code structure into a vector DB using Gemini+Chroma
- **Visualize** code graphs (in Neo4j)
- **Auto-generate** test YAMLs
- **Run tests** through a browser agent
- **Display** test results & analytics

Built with [Streamlit](https://streamlit.io/) for rapid UI, with multi-page flows for code upload, graph visualization, test management, and result analysis.

---

## 📁 Folder Structure
```
streamlit_website/
│
├── main.py                # Main Streamlit launcher
├── functions.py           # Core backend logic: AST, embedding, graph, test generation
├── parser_test.js         # Modern React/JSX static code parser (Node.js)
├── code_structure.json    # Auto-generated AST data
├── docker-compose.yaml    # Neo4j setup for graph DB
│
├── pages/
│   ├─ codebase.py         # Page 1: Upload & parse code, AST/embed/graph
│   ├─ tests.py            # Page 2: Generate & run tests, preview YAML
│   └─ results.py          # Page 3: Test result analytics & history
│
├── tests/                 # Auto-generated YAMLs (mirrors parsed app structure)         
├── results/               # Test results and outputs
├── Code_database/         # Chroma/embedding DB (auto-generated)
├── victim-site/           # React codebase to analyze (App.jsx, assets, etc)
└── ...
```

## 🌐 Features
- **React AST Parsing:** Decompose files into components, variables, functions, selectors (JS/JSX, with TypeScript support)
- **Vector Embedding:** Store code structure in ChromaDB using Gemini embeddings
- **Graph Visualization:** AST and code relationships visualized live (Neo4j backend)
- **Test Generation:** Automatic YAML creation with selectors, actions, expectations
- **Headless Agent Runner:** Launch tests in a browser (with Gemini LLM support), collect rich results
- **Streamlit UI:** Step-by-step, multi-tab interface

---

## 🛠️ Requirements
- **Python 3.9+** (recommend using a virtual environment)
- **Node.js 16+** (for parser_test.js)
- **Docker** (for running Neo4j)

**Python packages:**
(Typically installed with `pip install -r requirements.txt`)
```
streamlit
streamlit-option-menu
streamlit-agraph
langchain
langchain_community
langchain_neo4j
langchain_google_genai
google-generativeai
chromadb
rich
dotenv
tree_sitter
tree_sitter_javascript
pyyaml
neo4j
```

**Node packages:**
- @babel/parser
- @babel/traverse
- (install with: `npm install @babel/parser @babel/traverse` in project root)

---

## ⚡ Quickstart
### 1. Clone & install
```bash
git clone <this-repo-url>
cd streamlit_website
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
npm install @babel/parser @babel/traverse  # For parser_test.js
```

### 2. Configure .env
Create a `.env` file in the root directory with:
```
GOOGLE_API_KEY=your_gemini_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Launch Neo4j
```bash
docker compose up  # Start Neo4j (see docker-compose.yaml for creds)
```
Access Neo4j browser at [localhost:7474](http://localhost:7474), default Bolt port is 7687.

### 4. Start the app!
```bash
streamlit run main.py
```

---

## ⛓️ Full Workflow
1. **Add Codebase:**
   - Navigate to the "Add Codebase" tab
   - Enter the file path (e.g. `victim-site/App.jsx`)
   - Click **Create AST-structure** to parse and analyze components/attributes
   - Click **Start embedding process** (stores embeddings in ChromaDB)
   - Optionally, **Create a Graph** (builds nodes/relationships in Neo4j)
   - **Generate graph** to visualize structure interactively

2. **Run Tests:**
   - Switch to the "Run Tests" tab
   - Click **Run test** to auto-generate YAMLs from parsed code
   - Select files and preview generated YAML test-cases
   - Use the **Run testing agent** button to execute tests in a real or headless browser (limit tests if needed)

3. **Results & Analytics:**
   - Switch to "Test Results"
   - View pass/fail stats & test data
   - Browse YAMLs and outcomes for each component/file

---

## 📑 FAQ
- **Where is test data saved?**
  - YAMLs generated in `/tests/`, results in `/results/`.
- **How do I customize test strategies?**
  - Edit `parser_test.js` or YAML templates.
- **Can I use my own codebase?**
  - Yes—place your app files under `victim-site/`, then use their path in the UI.
- **Credentials?**
  - See `.env` requirements above.

---

## 🐛 Troubleshooting
- **Browser agent fails?**  Ensure Chrome/Chromium dependencies are installed for headless testing.
- **Docker/Neo4j port conflicts?**  Change ports in `docker-compose.yaml`.
- **Embedding/Gemini errors?**  Ensure your Google API key is valid and has billing enabled.
- **File not found?**  Provide the full path or relative to project root.

---

## 🤝 Contributing & Support
- Open issues/PRs for bugs or feature requests
- Contact: [your_email@example.com] (replace this with your actual email)

---

## 📝 License
MIT License (or specify your license here)

---

*Made with ❤️ by [Saim x Nau]*




