from tree_sitter import Language, Parser, Query, QueryCursor, Node
from langchain_neo4j import Neo4jGraph
import tree_sitter_javascript as tsj
from rich.console import Console
from dotenv import load_dotenv
from rich.panel import Panel
import os,yaml,json



# Load .env from the current directory where main.py is run
load_dotenv()

FILE_NAME="test-project/src/App.jsx"
JSLANGUAGE = Language(tsj.language()) #creates language
FUNCTIONS= ["arrow_function","function_declaration","function"]
VARIABLES= ["array_pattern"]
CONSOLE= Console()

#----- building the ast ------
def get_ancestry(node:Node)->list[str]:
    """
    Go up the tree to find parent function names, returning list of parent names (used for ID)
    """
    ancestry=[]
    current= node.parent
    while current is not None:
        if current.type in ["function_declaration","function"]:
            # for standard functions
            name_node= current.child_by_field_name("name")
            if name_node:
                ancestry.append(name_node.text.decode("utf8"))
        
        elif current.type == "arrow_function":
            #arrow functions are different
            if current.parent.type == "variable_declarator":
                name_node = current.parent.child_by_field_name("name")
                if name_node:
                    ancestry.append(name_node.text.decode("utf8"))
        
        elif current.type == "program":
            break
        current = current.parent
    
    #reverse the list
    content = list(reversed(ancestry))
    return content

def get_function(node:Node):
    """
    get the functions details as a node
    :param node: node from Ast
    :type node: Node
    """
    function = {
        "type": node.type,
        "params": "",
        "name": "",
        "top_level": False,
        "nested": [],
        "parent":None,
        "ancestry":[]
    }

    #get the name
    if node.type == "function_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            function["name"] = name_node.text.decode("utf8")
    elif node.type in ["arrow_function", "function_expression"]:
        if node.parent.type == "variable_declarator":
            name_node = node.parent.child_by_field_name("name")
            if name_node:
                function["name"] = name_node.text.decode("utf8")
        else:
            function["name"] = "anonymous"

    #get params
    params_node = node.child_by_field_name("parameters")
    if params_node:
        function["params"] = params_node.text.decode("utf8")

    #get ancestry
    ancestry = get_ancestry(node)
    function["ancestry"] = ancestry
    if not ancestry:
        function["top_level"] = True
    else:
        function["parent"] = ancestry[-1]
    
    #recursion NEEDED for nested function (rerun it)
    def find_nested_functions(n):
        nested = []
        for child in n.children:
            if child.type in FUNCTIONS:
                nested.append(get_function(child))
            elif child.type == "variable_declarator":
                val = child.child_by_field_name("value")
                if val and val.type in ["arrow_function", "function_expression"]:
                    nested.append(get_function(val))
            else:
                nested.extend(find_nested_functions(child))
        return nested
    
    function["nested"] = find_nested_functions(node)
    return function

def get_variables(node: Node):
    variable = {
        "type": "variable",
        "names": [],
        "value": "",
        "value_type": "",
        "top_level": False,
        "parent": None,
        "ancestry": []
    }
    
    ancestry = get_ancestry(node)
    variable["ancestry"] = ancestry
    variable["top_level"] = not bool(ancestry)
    if ancestry:
        variable["parent"] = ancestry[-1]

    left_side = node.child_by_field_name("name")
    right_side = node.child_by_field_name("value")

    if left_side:
        if left_side.type == "identifier":
            variable["names"].append(left_side.text.decode("utf8"))
        # Add logic for array/object patterns if needed

    if right_side:
        variable["value"] = right_side.text.decode("utf8")
        variable["value_type"] = right_side.type
    
    return variable

def get_frontend(node: Node):
    query = Query(JSLANGUAGE, """
        (identifier) @name
        (jsx_attribute) @properties
    """)
    attribute = {
        "name": "",
        "properties": [],
        "callbacks": [],
        "parent": None,
        "ancestry": []
    }
    cursor = QueryCursor(query)
    captures = cursor.captures(node)
    
    # Captures is already a dict[name, list[Node]]
    values = captures

    if values.get("name"):
        attribute["name"] = values["name"][0].text.decode("utf8")
    
    ancestry = get_ancestry(node)
    attribute["ancestry"] = ancestry
    if ancestry:
        attribute["parent"] = ancestry[-1]
    
    for prop in values.get("properties", []):
        attribute["properties"].append(prop.text.decode("utf8"))
        
        # Look for callbacks (onClick={handleClick})
        for child in prop.children:
            if child.type == "jsx_expression":
                def find_funcs(n):
                    res = []
                    for c in n.children:
                        if c.type == "identifier":
                            res.append({"name": c.text.decode("utf8"), "type": "callback"})
                        else:
                            res.extend(find_funcs(c))
                    return res
                attribute["callbacks"].extend(find_funcs(child))

    return attribute

def get_imports(node: Node):
    import_data = {"from": "", "import_items":[]}
    source_node = node.child_by_field_name("source")
    if source_node:
        import_data["from"] = source_node.text.decode("utf8").strip("'\"")
    
    clause = node.child_by_field_name("import_clause")
    if clause:
        for child in clause.children:
            if child.type == "identifier":
                import_data["import_items"].append(child.text.decode("utf8"))
            elif child.type == "named_imports":
                for spec in child.children:
                    if spec.type == "import_specifier":
                        name=spec.child_by_field_name("name")
                        if name:
                            import_data["import_items"].append(name.text.decode("utf8"))
    
    return import_data
    
def tree_splitter(file_path:str)-> dict:
    parser = Parser(JSLANGUAGE)
    with open(file_path,"r") as f:
        content = f.read()
    tree = parser.parse(bytes(content,encoding="utf8"))
    root = tree.root_node

    query = Query(JSLANGUAGE,"""
        (function_declaration) @Function
        (variable_declarator value: [(arrow_function)(function_expression)]) @Function
        (variable_declarator) @var
        (jsx_opening_element) @element
        (jsx_self_closing_element) @element
        (import_statement) @import
    """)
    contents = { "variables": [], "functions": [], "components": [], "imports": [] }
    cursor = QueryCursor(query)
    captures = cursor.captures(root)
    nodes_by_type = captures
    
    for node in nodes_by_type.get("Function", []):
        contents["functions"].append(get_function(node))
    for node in nodes_by_type.get("element", []):
        contents["components"].append(get_frontend(node))
    for node in nodes_by_type.get("var", []):
        # Filter out variables that are actually functions (handled previously)
        val = node.child_by_field_name("value")
        if not (val and val.type in ["arrow_function", "function_expression"]):
            contents["variables"].append(get_variables(node))
    for node in nodes_by_type.get("import", []):
        contents["imports"].append(get_imports(node))
    return contents


# CHANGING IT BACK






#------------ creating the graphs --------
def create_id(file_name, name, ancestry):
    path = "::".join(ancestry) if ancestry else "root"
    return f"{file_name}::{path}::{name}"

def build_ast_functions(file_name, codebase, graph):
    all_funcs = []

    def flatten(func_list):
        for func in func_list:
            my_id = create_id(file_name, func["name"], func["ancestry"])
            parent_id = (
                create_id(file_name, func["parent"], func["ancestry"][:-1])
                if func["parent"]
                else None
            )

            all_funcs.append({
                "id": my_id,
                "name": func["name"],
                "params": func["params"],
                "type": func["type"],
                "is_top_level": func["top_level"],
                "parent_id": parent_id
            })

            if func.get("nested"):
                flatten(func["nested"])

    flatten(codebase["functions"])

    # 1️⃣ Create all function nodes
    graph.query("""
        UNWIND $batch AS item
        MERGE (f:Function {id: item.id})
        SET f.name = item.name,
            f.params = item.params,
            f.type = item.type
    """, {"batch": all_funcs})

    # 2️⃣ Link top-level functions to File
    graph.query("""
        UNWIND $batch AS item
        MATCH (file:File {name: $filename})
        MATCH (func:Function {id: item.id})
        WHERE item.is_top_level = true
        MERGE (file)-[:CONTAINS]->(func)
    """, {"batch": all_funcs, "filename": file_name})

    # 3️⃣ Link nested functions to parent functions
    graph.query("""
        UNWIND $batch AS item
        MATCH (parent:Function {id: item.parent_id})
        MATCH (child:Function {id: item.id})
        WHERE item.is_top_level = false
        MERGE (parent)-[:CONTAINS]->(child)
    """, {"batch": all_funcs})

def build_ast_variables(file_name, codebase, graph):
    all_variables = []

    for var in codebase["variables"]:
        for name in var["names"]:
            ancestry = var.get("ancestry", [])

            all_variables.append({
                "id": create_id(file_name, name, ancestry),
                "name": name,
                "type": var["type"],
                "value": str(var["value"]),
                "value_type": var["value_type"],
                "is_top_level": not bool(ancestry),
                "parent_id": (
                    create_id(file_name, ancestry[-1], ancestry[:-1])
                    if ancestry else None
                )
            })

    # Create variable nodes
    graph.query("""
        UNWIND $batch AS item
        MERGE (v:Variable {id: item.id})
        SET v.name = item.name,
            v.type = item.type,
            v.value = item.value,
            v.value_type = item.value_type
    """, {"batch": all_variables})

    # Top-level variables
    graph.query("""
        UNWIND $batch AS item
        MATCH (file:File {name: $filename})
        MATCH (v:Variable {id: item.id})
        WHERE item.is_top_level = true
        MERGE (file)-[:DEFINES]->(v)
    """, {"batch": all_variables, "filename": file_name})

    # Nested variables
    graph.query("""
        UNWIND $batch AS item
        MATCH (parent:Function {id: item.parent_id})
        MATCH (v:Variable {id: item.id})
        WHERE item.is_top_level = false
        MERGE (parent)-[:DEFINES]->(v)
    """, {"batch": all_variables})


def build_ast_ui(file_name, codebase, graph):
    all_ui = []

    for comp in codebase["components"]:
        ancestry = comp.get("ancestry", [])

        all_ui.append({
            "id": create_id(file_name, f"JSX_{comp['name']}", ancestry),
            "name": comp["name"],
            "is_top_level": not bool(ancestry),
            "parent_id": (
                create_id(file_name, ancestry[-1], ancestry[:-1])
                if ancestry else None
            )
        })

    # Create UI nodes
    graph.query("""
        UNWIND $batch AS item
        MERGE (ui:Frontend {id: item.id})
        SET ui.name = item.name
    """, {"batch": all_ui})

    # File → UI
    graph.query("""
        UNWIND $batch AS item
        MATCH (file:File {name: $filename})
        MATCH (ui:Frontend {id: item.id})
        WHERE item.is_top_level = true
        MERGE (file)-[:RENDERS]->(ui)
    """, {"batch": all_ui, "filename": file_name})

    # Function → UI
    graph.query("""
        UNWIND $batch AS item
        MATCH (parent:Function {id: item.parent_id})
        MATCH (ui:Frontend {id: item.id})
        WHERE item.is_top_level = false
        MERGE (parent)-[:RENDERS]->(ui)
    """, {"batch": all_ui})



def build_ast_imports(file_name, codebase, graph):
    imports = []
    for imp in codebase["imports"]:
        target_name = f"./victim-site{imp['from'].strip('.')}"
        if not target_name.endswith(('.js', '.jsx')): target_name += ".jsx"
        
        for item in imp["import_items"]:
            target_id = create_id(target_name, item, "root")
            imports.append({"id": target_id, "func_name": item})
    CONSOLE.print(f"[bold magenta] imports [/bold magenta]")
    for i in imports:
        CONSOLE.print(i)

    query_import = """
    UNWIND $batch AS item
    MATCH (file:File {name: $filename})
    MERGE (func:Function {id: item.id})
    ON CREATE SET func.name = item.func_name
    MERGE (file)-[:IMPORTS]->(func)
    """
    graph.query(query_import, {"batch": imports, "filename": file_name})
    query_link = """
    MATCH (file:File {name: $filename})
    MATCH (file)-[:IMPORTS]->(importedDef)
    MATCH (file)-[:CONTAINS|RENDERS*]->(usage)
    WHERE usage.name = importedDef.name 
      AND NOT usage:File 
      AND NOT usage:ImportStatement 
      AND usage <> importedDef
    MERGE (usage)-[:REFERENCES]->(importedDef)
    """
    graph.query(query_link, {"filename": file_name})
    CONSOLE.print(f"[green]Processed imports & linked usages.[/green]")

def graph_creation(file_name: str):
    graph = Neo4jGraph() 
    # graph.query("MATCH (n) DETACH DELETE n") 
    codebase = tree_splitter(file_name)
    CONSOLE.print(Panel(f"[bold green]Parsed {file_name}[/bold green]"))
    for k,v in codebase.items():
        title=str(k)
        CONSOLE.print(Panel(str(v), title=title))
    graph.query("MERGE (f:File {name: $name})", {"name": file_name})
    build_ast_functions(file_name, codebase, graph)
    build_ast_imports(file_name, codebase, graph)
    build_ast_ui(file_name, codebase, graph)
    build_ast_variables(file_name, codebase, graph)


if __name__ == "__main__":
    try:
        if os.path.exists(FILE_NAME):
            graph_creation(FILE_NAME)
        else:
            CONSOLE.print(f"[red]File not found: {FILE_NAME}[/red]")
    except Exception:
        import traceback
        traceback.print_exc()
