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

#tree-sitter
from tree_sitter import Language,Parser,Query,QueryCursor,Node
import tree_sitter_javascript as tsj
from typing import Dict,List

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



#------- saim code finalized --------

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_javascript as tsj

FILE_NAME="../test-project/src/App.jsx"
JSLANGUAGE = Language(tsj.language()) #creates language
FUNCTIONS= ["arrow_function","function_declaration","function"]
VARIABLES= ["array_pattern"]

def tree_splitter(file:str)->dict[str,list[str]]:
    """AST walking creates dictionary in form:
        "variables":[],
        "functions":[],
        "components":[],
        "call_expression":[]
        "top_level_func":[]
        "imports":[]
    """
    parser = Parser(JSLANGUAGE) #parses language
    with open(file,"r") as f:
        content = f.read()
    tree = parser.parse(bytes(content,encoding="utf8"))
    root=tree.root_node

    # print(root)
    query=Query(JSLANGUAGE,
    """
        (function_declaration
            name: (identifier) @func_name
        ) @Function        

        (jsx_opening_element)@element

        (jsx_self_closing_element)@element

        (variable_declarator
        )  @var
        
        (call_expression
        ) @call

        (import_statement)@import
    """)

    contents={
        "variables":[],
        "functions":[],
        "components":[],
        "call_expression":[],
        "imports": []
    }
    cursor = QueryCursor(query)
    values=cursor.captures(root) #capture from the node u start

    for node in values.get("Function",[]):
        func=get_function(node)
        contents["functions"].append(func)

    for node in values.get("element",[]):
        contents["components"].append(get_frontend(node))

    for node in values.get("var",[]):
        contents["variables"].append(get_variables(node))

    for node in values.get("call",[]):
        contents["call_expression"].append(get_call(node))

    for node in values.get("import",[]):
        contents["imports"].append(get_imports(node))

    return contents

def get_function(node:Node):
    function={
        "type": node.type, #type of function (e.g. arrow function etc.)
        "params": "", #parameters
        "name": "", #name if there
        "top_level":False, #if top level or not
        "nested": [], #any functions contained within
        "parent": None
    }
    
    query=Query(JSLANGUAGE,"""
        (function_declaration
            name: (identifier) @name
            parameters: (formal_parameters) @params
        )
    """)
    cursor = QueryCursor(query)
    values=cursor.captures(node)
    if node.parent.type == "program":
        function["top_level"]=True
    else:
        parent_node=get_parent_function(node)
        for child in parent_node.children:
            if child.type == "identifier":
                function["parent"]= child.text.decode("utf8")
    # FIX: Take only the FIRST identifier (the function name)
    if values.get("name"):
        function["name"] = values["name"][0].text.decode("utf8")
    if values.get("params"):
        function["params"] = values["params"][0].text.decode("utf8")

    def find_nested_functions(n):
        nested = []
        for child in n.children:
            # FIX: Skip if it's just the "function" keyword (check child_count > 0)
            if child.type in FUNCTIONS and child.child_count > 0:
                nested.append(get_function(child))
            else:
                # Recursively search deeper
                nested.extend(find_nested_functions(child))
        return nested
    
    function["nested"] = find_nested_functions(node)
    
    return function

def get_frontend(node:Node):
    query=Query(JSLANGUAGE,"""
        (identifier)@name
        (jsx_attribute)@properties
    """)
    attribute={
        "name":"",
        "properties":[],
        "callbacks":[],
        "parent":None
    }
    cursor = QueryCursor(query)
    values=cursor.captures(node)
    
    if values.get("name"):
        attribute["name"] = values["name"][0].text.decode()
    
    parent_node=get_parent_function(node)
    for child in parent_node.children:
        if child.type == "identifier":
            attribute["parent"]= child.text.decode("utf8")
    
    for property in values.get("properties",[]):
        attribute["properties"].append(property.text.decode())
        
        for child in property.children:
            if child.type == "jsx_expression":
                # Recursively search for functions inside expressions
                def find_functions_in_expr(n):
                    results = []
                    for c in n.children:
                        # FIX: Skip garbage nodes with no children
                        if c.type in FUNCTIONS and c.child_count > 0:
                            results.append(get_function(c))
                        else:
                            results.extend(find_functions_in_expr(c))
                    return results
                
                attribute["callbacks"].extend(find_functions_in_expr(child))

    return attribute

def get_parent_function(node:Node):
    current = node.parent

    while current is not None:
        if current.type in ["function_declaration","arrow_function","function"]:
            return current
        if current.type == "program":
            return None
        current = current.parent

    return None

def get_variables(node:Node):
    variable={
        "type":"",
        "names":[],
        "value":"",
        "value_type":"",
        "top_level":False,
        "parent":None
    }
    if node.parent.type == "program":
        variable["top_level"]=True
    else:
        parent_node=get_parent_function(node)
        for child in parent_node.children:
            if child.type == "identifier":
                variable["parent"]= child.text.decode("utf8")

    left_side= None
    right_side = None
    for child in node.children:
        if child.type == "identifier":
            left_side=child
            variable["type"]="simple"
        elif child.type == "array_pattern":
            left_side=child
            variable["type"]="array_destructure"
        elif child.type == "object_pattern":
            left_side = child
            variable["type"]="object_destructure"
        elif child.type == "call_expression":
            right_side=child
            variable["value_type"]= "call_expression"
        elif child.type == "identifier" and variable["type"] != "":
            right_side = child
            variable["value_type"]="identifier"

    
    

    if left_side:
        if variable["type"]=="simple":
            variable["names"].append(left_side.text.decode("utf8"))
        elif variable["type"] == "array_destructure":
            for child in left_side.children:
                if child.type=="identifier":
                    variable["names"].append(child.text.decode("utf8"))

        elif variable["type"] == "object_destructure":
            for child in left_side.children:
                if child.type == "identifier":
                    variable["names"].append(child.text.decode("utf8"))
                elif child.type == "shorthand_property":
                    for sub in child.children:
                        if sub.type == "identifier":
                            variable["names"].append(sub.text.decode("utf8"))

    if right_side:
        variable["value"] = right_side.text.decode("utf8")
    
    return variable

def get_call(node:Node):
    """Extract function call information"""
    call={
        "function_name":"",
        "function_type":"",
        "arguments":[],
        "full_text":""
    }
    
    # Find the function being called
    for child in node.children:
        if child.type == "identifier":
            call["function_name"] = child.text.decode("utf8")
            break
    
    # Find arguments
    for child in node.children:
        if child.type == "arguments":
            call["arguments"].append(child.text.decode("utf8"))
        if child.type in FUNCTIONS:
            call["function_type"]=child.text.decode("utf8")

    call["full_text"] = node.text.decode("utf8")
    
    return call

def get_imports(node:Node):
    import_statement={
        "from":"",
        "import_items":[],
        "parent":None
    }
    query=Query(JSLANGUAGE,"""
    (import_statement
        (import_clause) @clause
    )
                """)
    cursor = QueryCursor(query)
    values=cursor.captures(node)
    for clause in values.get("clause",[]):
        for child in clause.children:
            if child.type == "identifier":
                import_statement["import_items"].append(child.text.decode("utf8"))
    
    for child in node.children:
        if child.type == "string": #imports are strings in javascript
            import_statement["from"]= child.text.decode("utf8")

    parent=get_parent_function(node)
    if parent:
        for child in parent.children:
            if child.type=="identifier":
                import_statement["parent"]=child.text.decode("utf8")
                break
    return import_statement

