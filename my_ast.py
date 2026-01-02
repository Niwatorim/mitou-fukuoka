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
def get_ancestry(node: Node) -> list[dict]:
    """
    Returns a list of parent info (name and type) to distinguish 
    between function parents and UI parents.
    """
    ancestry = []
    current = node.parent
    while current is not None:
        # Capture Function Parents
        if current.type in ["function_declaration", "function"]:
            name_node = current.child_by_field_name("name")
            if name_node:
                ancestry.append({"name": name_node.text.decode("utf8"), "type": "function"})
        
        # Capture JSX Parents
        elif current.type in ["jsx_element", "jsx_opening_element"]:
            # For fragments <>, call it Fragment
            name = "Fragment"
            if current.type == "jsx_element":
                opening = current.child_by_field_name("opening_element")
                if opening:
                    name_node = opening.child_by_field_name("name")
                    if name_node:
                        name = name_node.text.decode("utf8")
            ancestry.append({"name": name, "type": "ui","line": current.start_point[0]})

        elif current.type == "program":
            break
        current = current.parent
    
    return list(reversed(ancestry))

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

    # Handle array patterns (const [a, b] = ...)
    if node.type == "array_pattern":
        # Extract all identifiers inside the array pattern
        def find_identifiers(n):
            ids = []
            for child in n.children:
                if child.type == "identifier":
                    ids.append(child.text.decode("utf8"))
                else:
                    ids.extend(find_identifiers(child))
            return ids
        variable["names"].extend(find_identifiers(node))
        
        # Try to find value from parent's parent (variable_declarator)
        # array_pattern -> parent (variable_declarator) -> value
        if node.parent and node.parent.type == "variable_declarator":
            right_side = node.parent.child_by_field_name("value")
            if right_side:
                variable["value"] = right_side.text.decode("utf8")
                variable["value_type"] = right_side.type

    else:
        # Standard variable declaration
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
    # Only look for attributes within the opening element if it exists
    search_node = node
    if node.type == "jsx_element":
        opening = node.child_by_field_name("opening_element")
        if opening:
            search_node = opening

    query = Query(JSLANGUAGE, """
        (identifier) @name
        (jsx_attribute) @properties
    """)
    attribute = {
        "name": "",
        "line": node.start_point[0],
        "column": node.start_point[1],
        "properties": [],
        "callbacks": [],
        "parent": None,
        "ancestry": []
    }

    # Run query on search_node (opening element or self-closing element)
    cursor = QueryCursor(query)
    captures = cursor.captures(search_node)
    values = captures

    # If it's a self-closing element or opening element, name is usually the first identifier
    if values.get("name"):
        attribute["name"] = values["name"][0].text.decode("utf8")
    elif node.type == "jsx_element":
        # If no name found in opening element, it's a Fragment
        attribute["name"] = "Fragment"

    # Properties should also be extracted from search_node
    # We re-run query or reuse captures logic carefully

    # Reuse the captures from above
    for prop in values.get("properties", []):
            attribute["properties"].append(prop.text.decode("utf8"))
            
            # Look for callbacks (onClick={handleClick})
            # Iterate children of the property (jsx_expression etc)
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

    ancestry = get_ancestry(node)
    attribute["ancestry"] = ancestry
    if ancestry:
        attribute["parent"] = ancestry[-1]
    
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
        (jsx_element) @element
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
        name_node = node.child_by_field_name("name") # check left side
        
        # If value is a function, skip (handled by functions)
        if val and val.type in ["arrow_function", "function_expression"]:
            continue
            
        # If name is array pattern, treat as variable
        if name_node and name_node.type == "array_pattern":
             contents["variables"].append(get_variables(name_node))
        else:
             contents["variables"].append(get_variables(node))
    for node in nodes_by_type.get("import", []):
        contents["imports"].append(get_imports(node))
    return contents

# CHANGING IT BACK

#------------ creating the graphs --------
def create_id(file_name, name, ancestry,line=None):
    path = "::".join(ancestry) if ancestry else "root"
    suffix = f"L{line}" if line is not None else ""
    return f"{file_name}::{path}::{name}{suffix}"


def functions(file_name:str,functions:list[dict],graph:Neo4jGraph): #new
    """
    take codebase["functions"]
    flatten
        get ancestry, id and everything
        
    one array of all functions: merge them based on their content
    """
    all_funcs=[]
    def flatten(func_list:list[dict]):
        for func in func_list:
            ancestry_names = [a["name"] for a in func["ancestry"]] #name of all memebers to create a list
            id= create_id(file_name,func["name"],ancestry_names)
            parent_id=None
            if func["parent"]:
                parent_ancestry= [a["name"] for a in func["ancestry"][:-1]]
                parent_id= create_id(file_name, func["parent"]["name"],parent_ancestry)

            all_funcs.append({
                "id":id,
                "name":func["name"],
                "params":func["params"],
                "type":func["type"],
                "is_top_level": func["top_level"],
                "parent_id":parent_id
            })
            if func.get("nested"):flatten(func["nested"])
    flatten(func_list=functions)

    #create node
    query_create_node="""
    UNWIND $batch as item
    MERGE (func: Function {id:item.id, name:item.name,params:item.params,type:item.type})
    WITH item, func

    FOREACH (_ IN CASE WHEN item.is_top_level = true THEN [1] ELSE [] END |
        MERGE (f: File {name: $file_name})
        MERGE (f)-[:CONTAINS]->(func)
    )

    FOREACH (_ IN CASE WHEN item.ist_op_levl = true THEN [1] ELSE [] END |
        MERGE (parent: Function {id: parent_id})
        MERGE (parent)-[:CONTAINS]->(func)
    )
    """
    graph.query(query_create_node,{"batch":all_funcs,"file_name":file_name})

def build_ast_functions(file_name, codebase, graph): #old
    all_funcs = []

    def flatten(func_list):
        for func in func_list:
            # Extract just the 'name' from each dict in ancestry
            ancestry_names = [a['name'] for a in func["ancestry"]]
            
            my_id = create_id(file_name, func["name"], ancestry_names)
            
            parent_id = None
            if func["parent"]:
                # Also extract names for the parent's ancestry path
                parent_ancestry_names = [a['name'] for a in func["ancestry"][:-1]]
                parent_id = create_id(file_name, func["parent"]['name'], parent_ancestry_names)

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
    graph.query("""
        UNWIND $batch AS item
        MERGE (f:Function {id: item.id})
        SET f.name = item.name,
            f.params = item.params,
            f.type = item.type
    """, {"batch": all_funcs})

    graph.query("""
        UNWIND $batch AS item
        MATCH (file:File {name: $filename})
        MATCH (func:Function {id: item.id})
        WHERE item.is_top_level = true
        MERGE (file)-[:CONTAINS]->(func)
    """, {"batch": all_funcs, "filename": file_name})

    graph.query("""
        UNWIND $batch AS item
        MATCH (parent:Function {id: item.parent_id})
        MATCH (child:Function {id: item.id})
        WHERE item.is_top_level = false
        MERGE (parent)-[:CONTAINS]->(child)
    """, {"batch": all_funcs})

def variables(file_name:str, variables:list[dict],graph:Neo4jGraph): #new
    all_variables= []
    for var in variables:
        ancestry_names= [a["name"] for a in var["ancestry"]]
        id=create_id(file_name,var["name"],ancestry_names)
        parent_id=None
        if var["parent"]:
            parent_ancestry= [a["name"] for a in var["ancestry"][:-1]]
            parent_id= create_id(file_name, var["parent"]["name"],parent_ancestry)
        
        all_variables.append({
            "id": id,
            "name":var["name"],
            "type":var["type"],
            "value":var["value"],
            "value_type": var["value_type"],
            "is_top_level": var["top_level"],
            "parent_id":parent_id
        })
    
    graph.query("""
    UNWIND $batch as item
    MERGE (v:Variable {id: item.id})
    SET v.name = item.name,
        v.type = item.type,
        v.value = item.value,
        v.value_type = item.value_type
    
    WITH item, v
    FOREACH (_ IN CASE WHEN item.is_top_level = true THEN [1] ELSE [] END |
        MERGE (f:File {name: $file_name})
        MERGE (f)-[:DEFINES]->(v)
    )
    
    FOREACH (_ IN CASE WHEN item.is_top_level = false THEN [1] ELSE [] END |
        MERGE (parent: Function {id: item.parent_id})
        MERGE (parent)-[:DEFINES]->(v)
    )
    """,{"batch":all_variables,"file_name":file_name})

def build_ast_variables(file_name, codebase, graph): #old
    all_variables = []

    for var in codebase["variables"]:
        for name in var["names"]:
            ancestry = var.get("ancestry", [])

            all_variables.append({
                "id": create_id(file_name, name, [a['name'] for a in ancestry]),
                "name": name,
                "type": var["type"],
                "value": str(var["value"]),
                "value_type": var["value_type"],
                "is_top_level": not bool(ancestry),
                "parent_id": (
                    create_id(file_name, ancestry[-1]['name'], [a['name'] for a in ancestry[:-1]])
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

def components(file_name,components:list[dict],graph:Neo4jGraph): #new
    all_ui=[]
    for comp in components:
        ancestry_name = comp.get("ancestry",[])
        id=create_id(file_name,comp["name"],ancestry_name)
        parent_id=None
        parent_info = ancestry_name[-1] if ancestry_name else None
        parent_type = None
        if parent_info:
            parent_type = parent_info['type']
            parent_ancestry=[a["name"] for a in comp["ancestry"][:-1]]
            parent_id=create_id(file_name,comp["parent"]["name"],parent_ancestry)
    
        all_ui.append({
            "id": id,
            "name": comp["name"],
            "properties":comp["properties"],
            "callbacks": comp["callbacks"],
            "parent_id":parent_id,
            "parent_type":parent_type,
            "is_top_level": comp["top_level"]
        })
    
    graph.query("""
        UNWIND $batch as item
        MERGE (ui: Frontend {id: item.id})
        SET v.name = item.name,
            v.type = item.type,
            v.value = item.value,
            v.value_type = item.value_type
        
        WITH item,ui
        FOREACH (_ IN CASE WHEN item.parent_type = "ui" THEN [1] ELSE [] END |
            MERGE (parent: Frontend {id: item.parent_id})
            MERGE (parent)-[:RENDERS]->(ui)           
        )
        FOREACH (_ IN CASE WHEN item.parent_type = "function" THEN [1] ELSE [] END |
            MERGE (parent: Function {id: item.parent_id})
            MERGE (parent)-[:RENDERS]->(ui)           

        )

        FOREACH (_ IN CASE WHEN item.is_top_level = true THEN [1] ELSE [] END |
            MERGE (parent: File {name: $file_name})
            MERGE (parent)-[:RENDERS]->(ui)           
        )

    """,{"batch": all_ui,"file_name":file_name})

def build_ast_ui(file_name, codebase, graph): #old
    all_ui = []

    for comp in codebase["components"]:
        ancestry = comp.get("ancestry", [])
        line_num = comp.get("line")
        parent_info = ancestry[-1] if ancestry else None
        parent_id = None
        parent_type = None

        if parent_info:
            parent_type = parent_info['type']
            prefix = "JSX_" if parent_type == "ui" else ""
            # Pass the parent's line number to create_id
            parent_id = create_id(
                file_name, 
                f"{prefix}{parent_info['name']}", 
                [a['name'] for a in ancestry[:-1]], 
                line=parent_info.get('line') # <--- Crucial
            )

        all_ui.append({
            "id": create_id(file_name, f"JSX_{comp['name']}", [a['name'] for a in ancestry],line=line_num),
            "name": comp["name"],
            "parent_id": parent_id,
            "parent_type": parent_type,
            "is_file_level": not bool(ancestry)
        })

    # 1. Create all Frontend nodes
    graph.query("""
        UNWIND $batch AS item
        MERGE (ui:Frontend {id: item.id})
        SET ui.name = item.name
    """, {"batch": all_ui})

    # 2. UI -> UI 
    graph.query("""
        UNWIND $batch AS item
        MATCH (parent:Frontend {id: item.parent_id})
        MATCH (child:Frontend {id: item.id})
        WHERE item.parent_type = 'ui'
        MERGE (parent)-[:RENDERS]->(child)
    """, {"batch": all_ui})

    # 3. Function -> UI
    graph.query("""
        UNWIND $batch AS item
        MATCH (parent:Function {id: item.parent_id})
        MATCH (child:Frontend {id: item.id})
        WHERE item.parent_type = 'function'
        MERGE (parent)-[:RENDERS]->(child)
    """, {"batch": all_ui})

    # 4. File -> UI 
    graph.query("""
        UNWIND $batch AS item
        MATCH (file:File {name: $filename})
        MATCH (ui:Frontend {id: item.id})
        WHERE item.is_file_level = true
        MERGE (file)-[:RENDERS]->(ui)
    """, {"batch": all_ui, "filename": file_name})

def imports(file_name:str, imports:list[dict],graph:Neo4jGraph): # new
    all_imports=[]
    for imp in imports:
        target_name = f"./victim-site{imp['from'].strip('.')}"
        if not target_name.endswith(('.js', '.jsx')): target_name += ".jsx"
        
        for item in imp["import_items"]:
            target_id = create_id(target_name,item,"root")
            all_imports.append({
                "id": target_id,
                "func_name":item
            })
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

def build_ast_imports(file_name, codebase, graph): #old 
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
    graph.query("MATCH (n) DETACH DELETE n") 
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

def no_del_graph_creation(file_name: str):
    graph = Neo4jGraph()
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
