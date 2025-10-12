import json
import subprocess
import os
from dotenv import load_dotenv

# langchain
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# gemini embeddings and retrieval
from google import genai
from google.genai import types

# chromadb
import chromadb
from typing import Dict
load_dotenv()


root_dir = "./samples"
ast_dir = "./samples_ast"

# to parse each file into AST json
def ast_parser(code_string):
    values=subprocess.run(
        ["node","../ast_generator.js"],
        input=code_string,
        capture_output=True,
        text=True,
        check=True,
    )
    return values.stdout

# --- The Traversal Logic ---
def get_node_text(node, source_code):
    """Slices the source code to get the text of a specific AST node."""
    return source_code[node['start']:node['end']]

def get_jsx_element_name(node):
    """Safely gets the name of a JSX element (e.g., 'Sidebar')."""
    if node and node.get('type') == 'JSXElement':
        return node.get('openingElement', {}).get('name', {}).get('name')
    return None

def traverse(node, results, source_code):
    """
    Recursively traverses the AST, collecting information into the results dictionary.
    This is the core of the solution.
    """
    if not isinstance(node, dict):
        return

    # 1. Find Imports and Hooks
    if node.get('type') == 'ImportDeclaration':
        import_text = get_node_text(node, source_code)
        results['imports'].append(import_text)
        # Check specifiers for hooks
        for specifier in node.get('specifiers', []):
            imported_name = specifier.get('local', {}).get('name', '')
            if imported_name.startswith('use'):
                results['hook_usage'].append(import_text)

    # 2. Find the Main Component Function
    # Handles `const MyComponent = () => {}` and `function MyComponent() {}`
    if node.get('type') == 'VariableDeclaration':
        for declaration in node.get('declarations', []):
            if declaration.get('init', {}).get('type') == 'ArrowFunctionExpression':
                component_name = declaration.get('id', {}).get('name')
                if component_name and component_name[0].isupper(): # Convention for components
                    results['main_component_function'] = component_name
                    # Now traverse inside this component's body
                    traverse(declaration.get('init', {}).get('body'), results, source_code)

    if node.get('type') == 'FunctionDeclaration':
        component_name = node.get('id', {}).get('name')
        if component_name and component_name[0].isupper():
            results['main_component_function'] = component_name
            # Traverse inside this component's body
            traverse(node.get('body'), results, source_code)


    # 3. Find Return Statement and analyze JSX
    if node.get('type') == 'ReturnStatement':
        # The returned content is in the 'argument'
        traverse(node.get('argument'), results, source_code)

    # 4. Find Always-Rendered and Conditional Components inside JSX
    if node.get('type') == 'JSXElement':
        component_name = get_jsx_element_name(node)
        if component_name:
            results['return']['always_rendered'].add(component_name) # Use a set to avoid duplicates

    if node.get('type') == 'ConditionalExpression':
        # Extract the code for the entire ternary expression
        conditional_code = get_node_text(node, source_code)
        results['return']['conditional_rendering'].append(conditional_code)
        
        # Also extract the names of the components being rendered
        consequent_name = get_jsx_element_name(node.get('consequent'))
        alternate_name = get_jsx_element_name(node.get('alternate'))
        
        if consequent_name:
             results['return']['always_rendered'].discard(consequent_name) # Remove from always rendered
             results['return']['conditional_components'].add(consequent_name)
        if alternate_name:
             results['return']['always_rendered'].discard(alternate_name) # Remove from always rendered
             results['return']['conditional_components'].add(alternate_name)


    # --- The Recursive Step ---
    # Continue traversing through all possible children of the current node
    for key, value in node.items():
        if isinstance(value, dict):
            traverse(value, results, source_code)
        elif isinstance(value, list):
            for item in value:
                traverse(item, results, source_code)

# create embedding document for each AST file
def create_embedding_doc(analysis_results, metadata_for_db):
    description = (
    f"Component: {analysis_results['main_component_function']}.\n\n"
    f"It always renders the {', '.join(analysis_results['return']['always_rendered'])} component(s). "
    f"It uses the following hooks: {', '.join(hook[hook.find('{')+1:hook.find('}')].strip() for hook in analysis_results['hook_usage'])}. "
    f"Based on application state, it conditionally renders one of the following components: {', '.join(analysis_results['return']['conditional_components'])}."
    )
    doc = Document(
        page_content=description,
        metadata=metadata_for_db
    )
    return doc

# loop through the directory, parse each file into AST json, and transverse each AST
def process_directory(root_dir:str, ast_dir:str)-> Dict:
    documents = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Process only JavaScript and JSX files
            if filename.endswith(('.jsx')):

                analysis_results = {
                    "file_name":filename,
                    'main_component_function': 'Unknown',
                    'imports': [],
                    'hook_usage': [],
                    'return': {
                        'always_rendered': set(),
                        'conditional_rendering': [],
                        'conditional_components': set(),
                    }
                }

                source_path = os.path.join(dirpath, filename)
                print(f"Processing {source_path}...")

                # 1. read the source code
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()

                # 2. parse source code into AST and save it into AST json file
                source_ast_code = ast_parser(source_code)
                ast_json = json.loads(source_ast_code)
                ast_save_path = os.path.join(ast_dir, f"{filename}.json")
                if not os.path.exists(ast_save_path):
                    with open(ast_save_path, 'w') as w:
                        json.dump(ast_json, w, indent=4)
                
                # 3. load and transverse the AST
                print(f"Traversing AST for {filename}...")
                traverse(ast_json['program'], analysis_results, source_code)

                return analysis_results

