* Folder structure *

streamlit_website
|
|____ pages: for all the different streamlit pages
|
|____ results: stores yaml files for all results (mirrors tests)
|
|____ tests: contains yaml files for all tests
|
|____ vitim-site: site to be tested
|
|
main.py : main file to run

functions.py: contains all processes

parser_test.js: has the ast parser

code_structure.json: once parsed, code structure is saved there

.env: contains Gemini API key, neo4j username, neo4j uri, neo4j password

* How it works *

1) Once started, go to codebase and give filename path to be tested. Should be in format:
       **victim-site/YourFileName **

2) Create AST structure to break into AST structure

3) Click on embed to embed the AST structure

4) Click on generate graph to create the graph to be viewed then view it if you wish

** Switch to tests tab **

5) Click run tests to go through the AST and create testable attributes

6) To actually do the tests, click on run agent (currently limited two only two tests for beta purposes)
    (enable headless mode if you trust the agent and want it to simply run the test)

** Switch to results tab **

7) Preview tests and view statistics


* How to start *

Run docker to start neo4j


In terminal, run:
`bash`
 streamlit run main.py
`/bash`
