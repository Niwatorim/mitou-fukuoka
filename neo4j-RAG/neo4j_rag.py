'''
make ast for analysis_results -> Dict analysis_results -> graph -> for conditional rendering add properties in how to test
make ast for testableAttributes -> structure.json -> graph file -> has attributes -> each attribute has properties of how to test
extract the conditional rendering
say in prompt how to read conditional rendering and how to test
in access_code, loop analysis_results to get conditional rendering, make into a list together with testable Attributes. then loop through the component of the combined list

for imports, 
name: main_component name -> :IMPORTS COMPONENT FUNCTION -> Navbar ON CREATE SET ./components/Navbar
for hook usage,
name: main_component name -> :IMPORTS HOOOKS -> useAuthStore ON CREATE SET ./store/useAuthStore.js
for conditional rendering,
name: main_component_name -> :HAS ROUTE -> path ON CREATE SET "isConditional": True, "condition": authUser  -> :IF TRUE -> HomePage
                                                                -> :IF FALSE -> Navigation (type) to
{
    'main_component_function': 'Unknown',
    'imports': [],
    'hook_usage': [],
    'routes': 
    [
        {
            "path": None,
            "element": {
                "isConditional": True,
                "condition": authUser
                "true": {
                    "type":
                    "name"
                }
                false: {
                    "type":
                    "to":
                }
        }
    ]
}
'''
import typing
from dotenv import load_dotenv
import json
import subprocess
import yaml
import os
from browser_use import Agent, ChatGoogle
import asyncio
# neo4j
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
from langchain_graph_retriever import GraphRetriever
from graph_retriever.strategies import Eager
# langchain RAG
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
# ast extract for nodes and relationships
from ast_traverse import ast_rag, ast_parser, traverse
from pathlib import Path
import re

load_dotenv()

root_dir = "../samples"
ast_dir = "../samples_ast"
attributes_dir = "../samples_attributes"


NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE
)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY
)

def chunk_directory(root_dir, ast_dir) -> list[dict]: # directory -> file -> AST -> chunk
    analysis_results_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Process only JavaScript and JSX files
            if filename.endswith(('.jsx')):

                analysis_results = {
                    'main_component_function': 'Unknown',
                    'imports': [],
                    'hook_usage': [],
                    'routes': []
                    }

                source_path = os.path.join(dirpath, filename)
                print(f"Processing {source_path}...")

                # 1. read the source code
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()

                # 2. parse source code into AST and save it into AST json file
                print("Parsing for analysis_results..")
                source_ast_code = ast_parser(source_code)
                ast_json = json.loads(source_ast_code)
                ast_save_path = os.path.join(ast_dir, f"{filename}.json")
                with open(ast_save_path, 'w') as w:
                    json.dump(ast_json, w, indent=4)

                # 2.1 parse source code to get testableAttributes
                print("Parsing for testableAttributes..")
                ast_attributes = ast_rag(source_path)
                attributes_data = json.loads(ast_attributes)
                attributes_save_path = os.path.join(attributes_dir, f"{filename}.json")
                with open(attributes_save_path, 'w') as a:
                    json.dump(attributes_data, a, indent=4)
                
                # 3. load and transverse the AST
                print(f"Traversing AST for {filename}...")
                traverse(ast_json['program'], analysis_results, source_code)

                analysis_results_list.append(analysis_results)
    
    return analysis_results_list

def _parse_import_statement(statement: str):
    """Parses an import statement to extract names and the source path."""
    # This regex handles `import { Name1, Name2 } from "source"` and `import Name from "source"`
    pattern = re.compile(r"import\s+(?:([\w\s{},]+)\s+from)?\s+['\"]([^'\"]+)['\"]")
    match = pattern.search(statement)
    if not match:
        return None, None

    names_part, source = match.groups()
    if not names_part:
        return [], source
        
    # Clean up the names part by removing braces and splitting
    names = [name.strip() for name in names_part.replace("{", "").replace("}", "").split(",")]
    return names, source

# --- Main Ingestion Function ---

def ingest_chunks(graph: Neo4jGraph, results: dict, attributes_dir: str):
    """
    Ingests analysis results and testable attributes into a detailed Neo4j graph.
    """
    # Main file/component node
    main_component_name = results.get('main_component_function', 'UnknownComponent')
    graph.query("MERGE (f:File {name: $name})", params={'name': main_component_name})

    # Ingest Testable Attributes from the separate JSON
    for filename in os.listdir(attributes_dir):
        full_path = os.path.join(attributes_dir, filename)
        with open(full_path,"r") as f: 
            attribute_data=json.load(f) 
    
    for component in attribute_data.get('components', []):
        for attribute in component.get('testableAttributes', []):
            graph.query("""
                MATCH (main:File {name: $main_name})
                MERGE (el:TestableElement {name: $element_name, parentFunction: $parent_func})
                MERGE (main)-[:CONTAINS_ELEMENT]->(el)
                MERGE (el)-[:HAS_ATTRIBUTE]->(attr:Attribute {name: $attr_name})
                SET
                    attr.value = $attr_value,
                    attr.action = $strategy_action,
                    attr.expectation = $strategy_expectation,
                    attr.description = $strategy_description
            """, params={
                'main_name': main_component_name,
                'element_name': component.get('name'),
                'parent_func': component.get('parentFunction'),
                'attr_name': attribute.get('name'),
                'attr_value': attribute.get('value', {}).get('value'),
                'strategy_action': attribute.get('strategy', {}).get('action'),
                'strategy_expectation': attribute.get('strategy', {}).get('expectation'),
                'strategy_description': attribute.get('strategy', {}).get('description')
            })

    # Process and ingest Imports
    for imp_statement in results.get('imports', []):
        names, source = _parse_import_statement(imp_statement)
        if not source:
            continue
        
        # Determine if it's a component, hook, or general import
        relationship_type = "IMPORTS"
        if "use" in imp_statement:
            relationship_type = "IMPORTS_HOOK"
        elif any(name[0].isupper() for name in names): # Heuristic: Component names are capitalized
            relationship_type = "IMPORTS_COMPONENT"
            
        for name in names:
            graph.query(f"""
                MATCH (main:File {{name: $main_name}})
                MERGE (imp:Importable {{name: $name, source: $source}})
                MERGE (main)-[:{relationship_type}]->(imp)
            """, params={'main_name': main_component_name, 'name': name, 'source': source})

    # Process and ingest Conditional Rendering
    for route_data in results.get('routes', []):
        path = route_data.get("path")
        element = route_data.get("element", {})
        if not path: continue

        # Create the File->Route link first
        graph.query("""
            MATCH (f:File {name: $main_name})
            MERGE (r:Route {path: $path})
            MERGE (f)-[:DEFINES_ROUTE]->(r)
        """, params={"main_name": main_component_name, "path": path})

        if not element.get("isConditional"):
            # Handle simple, non-conditional routes (e.g., element={<SettingsPage/>})
            render_info = element.get("render", {})
            if render_info and render_info.get("type") == "Component":
                graph.query("""
                    MATCH (r:Route {path: $path})
                    MERGE (c:Component {name: $component_name})
                    MERGE (r)-[:RENDERS_COMPONENT]->(c)
                """, params={"path": path, "component_name": render_info.get("name")})
        else:
            # Handle complex, conditional routes
            condition = element.get("condition")
            true_outcome = element.get("true_outcome", {})
            false_outcome = element.get("false_outcome", {})

            if not condition: continue

            # Create the Condition node and link it to the Route
            graph.query("""
                MATCH (r:Route {path: $path})
                MERGE (cond:Condition {expression: $condition})
                MERGE (r)-[:HAS_CONDITION]->(cond)
            """, params={"path": path, "condition": condition})

            # Connect the TRUE outcome
            if true_outcome.get("type") == "Component":
                graph.query("""
                    MATCH (cond:Condition {expression: $condition})
                    MERGE (c:Component {name: $name})
                    MERGE (cond)-[:IF_TRUE]->(c)
                """, params={"condition": condition, "name": true_outcome.get("name")})
            elif true_outcome.get("type") == "Navigation":
                graph.query("""
                    MATCH (cond:Condition {expression: $condition})
                    MERGE (nav:Navigation {targetPath: $target})
                    MERGE (cond)-[:IF_TRUE]->(nav)
                """, params={"condition": condition, "target": true_outcome.get("to")})

            # Connect the FALSE outcome
            if false_outcome.get("type") == "Component":
                graph.query("""
                    MATCH (cond:Condition {expression: $condition})
                    MERGE (c:Component {name: $name})
                    MERGE (cond)-[:IF_FALSE]->(c)
                """, params={"condition": condition, "name": false_outcome.get("name")})
            elif false_outcome.get("type") == "Navigation":
                graph.query("""
                    MATCH (cond:Condition {expression: $condition})
                    MERGE (nav:Navigation {targetPath: $target})
                    MERGE (cond)-[:IF_FALSE]->(nav)
                """, params={"condition": condition, "target": false_outcome.get("to")})

    print(f"Graph updated for {main_component_name}")

def create_docs(graph) -> list[Document]:
    query = """
    MATCH (main:File)
    RETURN
        main.name AS componentName,
        [(main)-[:ALWAYS_RENDERS]->(c) | c.name] AS alwaysRendered,
        [(main)-[:CONDITIONALLY_RENDERS]->(c) | c.name] AS conditionallyRendered,
        [(main)-[:USES_HOOK]->(h) | h.name] AS hooks,
        [(main)-[:IMPORTS]->(i) | i.name] AS imports
    """
    results = graph.query(query)
    
    docs = []
    for record in results:
        # Create a descriptive text string (page_content) from the graph data.
        content = f"Component `{record['componentName']}` analysis:\n"
        if record['imports']:
            content += f"- Imports: {', '.join(record['imports'])}\n"
        if record['hooks']:
            content += f"- Uses Hooks: {', '.join(record['hooks'])}\n"
        if record['alwaysRendered']:
            content += f"- Always Renders: {', '.join(record['alwaysRendered'])}\n"
        if record['conditionallyRendered']:
            content += f"- Conditionally Renders: {', '.join(record['conditionallyRendered'])}\n"
        
        # The metadata helps in filtering or referencing the original node.
        doc = Document(
            page_content=content,
            metadata={"component_name": record['componentName']}
        )
        docs.append(doc)
        
    print(f"Created {len(docs)} documents from the graph.")
    return docs

# vector index graph only method
def create_vector_index(docs) -> Neo4jVector:
    neo4j_db = Neo4jVector.from_documents(
        embedding=embeddings,
        documents=docs,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD, 
        index_name="components" # give the index a name to know which index to retrieve
    )

    retrieval_query = """
    // Find the Document node from the vector search
    MATCH (node)
    // Find the corresponding File node in the main graph
    MATCH (main:File {name: node.component_name})

    // Collect all direct imports and their types
    CALL {
        WITH main
        OPTIONAL MATCH (main)-[r]->(imp:Importable)
        WHERE type(r) STARTS WITH 'IMPORTS'
        RETURN collect(
            imp.name + ' from ' + imp.source + ' (' + replace(type(r), 'IMPORTS_', '') + ')'
        ) AS imports
    }

    // Collect all routing logic defined in this file
    CALL {
        WITH main
        OPTIONAL MATCH (main)-[:DEFINES_ROUTE]->(r:Route)
        // Handle conditional routes
        OPTIONAL MATCH (r)-[:HAS_CONDITION]->(cond)
        OPTIONAL MATCH (cond)-[:IF_TRUE]->(t_outcome)
        OPTIONAL MATCH (cond)-[:IF_FALSE]->(f_outcome)
        // Handle non-conditional routes
        OPTIONAL MATCH (r)-[:RENDERS_COMPONENT]->(nc_comp)
        
        RETURN collect(
            DISTINCT
            r.path + ': ' +
            CASE
                WHEN cond IS NOT NULL THEN
                    'If (' + cond.expression + ') then ' +
                    CASE WHEN t_outcome:Component THEN 'render ' + t_outcome.name ELSE 'navigate to ' + t_outcome.targetPath END +
                    ' else ' +
                    CASE WHEN f_outcome:Component THEN 'render ' + f_outcome.name ELSE 'navigate to ' + f_outcome.targetPath END
                ELSE 'renders ' + nc_comp.name
            END
        ) AS routes
    }
    
    // Collect testable elements
    CALL {
        WITH main
        OPTIONAL MATCH (main)-[:CONTAINS_ELEMENT]->(el:TestableElement)-[:HAS_ATTRIBUTE]->(attr)
        RETURN collect(
            el.name + ' has attribute ' + attr.name + ' with value "' + attr.value +
            '". Test strategy: ' + attr.description
        ) AS testableElements
    }

    // Combine all information into a single text block
    RETURN "Component: " + main.name + "\n" +
        "Imports:\n- " + apoc.text.join(imports, "\n- ") + "\n" +
        "Routing Logic:\n- " + apoc.text.join(routes, "\n- ") + "\n" +
        "Testable Elements:\n- " + apoc.text.join(testableElements, "\n- ")
    AS text, score, 
    {component_name: main.name} AS metadata
"""

    vector_index = Neo4jVector.from_existing_index(
        embedding=embeddings,
        index_name="components",
        retrieval_query=retrieval_query
    )

    retriever = vector_index.as_retriever()
    return retriever

def access_code(instructions, retriever):
    docs = retriever.invoke(instructions)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key=GEMINI_API_KEY,
                                 model_kwargs={
                                     "response_mime_type":"application/yaml"
                                 })
    prompt = ChatPromptTemplate.from_template("""
    You are a test automation expert. Generate test instructions in YAML format.

    Codebase knowledge graph: {context}
    Component: {input}

    Return instructions with this exact structure and nothing else:
    if "text" is not empty, then use it to help identify the element within the test_steps.

    Requirements:
    - Each instruction must be ONE clear action.
    - Include specific selectors (id, class, data-testid, or text)
    - Use action types: {tool_context}
    - Conditional rendering meaning -> condition ? expressionIfTrue : expressionIfFalse;

    Example:                                                                  
    component: "component_name",
    url: http://localhost:5173/
    test_steps:
    - step: 1
    action: browser_navigate
    instruction: Open the application
    target: http://localhost:5173/
    - step: 2
    action: browser_click
    instruction: Click link and verify navigation to its destination
    selector: a[href="https://vite.dev"]
    expected: Navigation to https://vite.dev
""")

    print("Making message...")
    document_chain = create_stuff_documents_chain(llm, prompt)

    with open("mcp_tools.yml", 'r', encoding='utf-8') as f:
        tool_context = f.read()

    print("invoke message...")
    response = document_chain.invoke({
        "input": instructions,
        "context": docs,
        "tool_context": tool_context
    })
    print(response)
    return (yaml.safe_load(response))

#helper function for cycle()
def unique_file(name,existing_files):
    count=1
    file=f"./tests/{name}.yaml"
    while file in existing_files:
        file=f"./tests/{name}[{count}].yaml"
        count+=1
    
    existing_files.add(file)
    return file

#go through every item in code_json and generate instructions for it using RAG from embedding, thne save to tests
def cycle(results_list: list, retriever):
    os.makedirs("tests", exist_ok=True)
    existing_files = set()

    print("Generating tests for testable attributes..")
    for filename in os.listdir(attributes_dir):
        full_path = os.path.join(attributes_dir, filename)
        with open(full_path,"r") as f: 
            attribute_data=json.load(f) 

    component_function_name = os.path.basename(full_path)[0]
    for index_i,i in enumerate(attribute_data["components"]):
        if i["testableAttributes"]:
            instruction= f"please give instructions to test the attribute {i} located in the main component function {component_function_name}"
            yaml_data=access_code(instruction, retriever)
            filename = unique_file(i['name'], existing_files)

            with open(filename,"w") as f:
                yaml.dump(yaml_data,f,default_flow_style=False, sort_keys=False)

    print("Generating tests for conditional routes..")
    for result in results_list:
        component_function = result.get('main_component_function', 'UnknownComponent')
        for route in result.get('routes', []):
            route_path = route.get('path')
            if not route_path:
                continue

            # Create a clear instruction for the LLM
            instruction = f"Please give instructions to test the route with path '{route_path}' in the '{component_function}' component. Its routing logic is: {json.dumps(route['element'])}"

            print(f"Generating test for route: {route_path}")
            yaml_data = access_code(instruction, retriever)
            
            # Sanitize the path for use as a filename and save
            if yaml_data:
                # Replace '/' with ' ' to create a valid filename
                sanitized_route_name = route_path.replace('/', '_').strip('_') if route_path != '/' else 'root'
                output_filename = unique_file(sanitized_route_name, existing_files)
                with open(output_filename, "w") as f_out:
                    yaml.dump(yaml_data, f_out, default_flow_style=False, sort_keys=False)

async def test_browser_use():
    directory= os.listdir("./tests")
    success_files=[]
    for file in directory:
        with open(f"./tests/{file}","r") as f:
            data=yaml.safe_load(f)
        task=str(yaml.dump(data["test_steps"], default_flow_style=False, sort_keys=False))
        agent = Agent(
            task=task,
            llm=ChatGoogle(model="gemini-2.5-flash"),
        )
        history = await agent.run()
        success={"name":file,"success":history.is_successful()}
        success_files.append(success)
    print("Failures:")
    for i in success_files:
        if i["success"]==False:
            print(i["name"])

def main():
    # graph.query("MATCH (n) DETACH DELETE n")

    # # --- 2. PARSE & INGEST ---
    # # Parse the codebase and get structured data
    # analysis_results_list = chunk_directory(root_dir, ast_dir)
    
    # # Ingest the structured data into Neo4j
    # for result in analysis_results_list:
    #     ingest_chunks(graph, result, attributes_dir)
    
    # docs = create_docs(graph)

    # retriever = create_vector_index(docs)

    # cycle(analysis_results_list, retriever)

    print("Starting browser tests...")
    asyncio.run(test_browser_use())
    print("Browser tests finished.")

if __name__ == "__main__":
    asyncio.run(main())
