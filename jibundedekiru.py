from tree_sitter import Language, Parser, Query, QueryCursor, Node
from langchain_neo4j import Neo4jGraph
import tree_sitter_javascript as tsj
from rich.console import Console
from dotenv import load_dotenv
from rich.panel import Panel
import os,yaml,json

load_dotenv()
FILE_NAME="test-project/src/App.jsx"
JSLANGUAGE = Language(tsj.language()) #creates language
FUNCTIONS= ["arrow_function","function_declaration","function"]
VARIABLES= ["array_pattern"]
CONSOLE= Console()

def create_id(file_name, name, ancestry,line=None):
    path = "::".join(ancestry) if ancestry else "root"
    return f"{file_name}::{path}::{name}"

def get_ancestry(node: Node) -> list[dict]:
    """
    Returns a list of parent info (name and type) to distinguish 
    between function parents and UI parents.
    """
    ancestry = []
    current = node.parent
    last_ui=None # for ui changes

    while current is not None:
        # Functions
        if current.type in ["function_declaration", "function"]:
            name_node = current.child_by_field_name("name")
            if name_node:
                ancestry.append({"name": name_node.text.decode("utf8"), "type": "function"})
        
        # Capture JSX Parents
        elif current.type == "jsx_element":
            opening = current.child_by_field_name("opening_element")
            if opening:
                name_node = opening.child_by_field_name("name")
                if name_node:
                    name = name_node.text.decode("utf8")
                    if last_ui != name:
                        ancestry.append({
                            "name": name,
                            "type":"ui"
                        })
                        last_ui = name
        
        elif current.type == "jsx_fragment":
            if last_ui != "Fragment":
                ancestry.append({
                    "name":"Fragment",
                    "type":"ui"
                })
                last_ui= "Fragment"
        
        elif current.type == "program":
            break

        current= current.parent
    return list(reversed(ancestry))

def graph_creation(file_name:str) -> None: #old
    """
    Creates AST graph and stores in Neo4J database
    """

    graph=Neo4jGraph()
    #Kill everything in the graph:
    graph.query("MATCH (n) DETACH DELETE n")
    console = Console()
    debug_logs=[]
    console.print("[bold magenta] Deleted everything in graph..... [/bold magenta]")


    codebase=tree_splitter(file_name)
    console.print(
        Panel(
            f"[bold green] {json.dumps(codebase,indent=2)}[/bold green]"
        )
    )
    file="App.jsx"
    
    #---- Create the file node
    create_file= "MERGE (f:File {name: $filename})"

    graph.query(create_file,{"filename":file})

    #--- Add top level functions
    for function in codebase["functions"]:
        name=function["name"]
        params=function["params"]
        func_type=function["type"]
        if function.get("top_level"):
            
            debug_logs.append("#DEBUG displaying top level")
            query="""
            MATCH (f:File {name: $filename})
            MERGE (func: Function {name: $name, params: $params, type: $type})
            MERGE (f)-[:CONTAINS]->(func)
            """
            graph.query(query,
                        {"filename":file,
                         "name":name,
                         "params":params,
                         "type":func_type
                        })
            
        def nested_func(parent:str,nested_list:list):
            debug_logs.append("#DEBUG Checking nested")
            for nested in nested_list:
                name=nested["name"]
                params=nested["params"]
                nested_type=nested["type"]

                query="""
                MERGE (nested: Function {name: $name, params: $params, type: $type})
                WITH nested
                MATCH (parent: Function {name: $parent})
                MERGE (parent)-[:CONTAINS]->(nested)
                """
                graph.query(query,{
                    "name":name,
                    "params":params,
                    "type":nested_type,
                    "parent":parent
                })
                if nested["nested"]:
                    nested_func(name,nested["nested"])


        if function["nested"]:
            nested_func(name,function["nested"])

    #--- for variables
    for variable in codebase["variables"]:
        debug_logs.append("#DEBUG Checking variables")
        names=variable["names"]
        var_type=variable["type"]
        value=variable["value"]
        value_type=variable["value_type"]
        for name in names:
            if variable.get("top_level"):
                query="""
                MATCH (f:File {name: $filename})
                MERGE (var: Variable {name: $name, type: $type, value: $value, value_type: $value_t})
                MERGE (f)-[:CONTAINS]->(var)
                """
                graph.query(query,{
                    "filename":file,
                    "name":name,
                    "type":var_type,
                    "value":value,
                    "value_type":value_type
                })
            else:
                parent = variable["parent"]
                
                if parent:
                    query="""
                    MATCH (f:Function {name: $parentname})
                    MERGE (var: Variable {name: $name, type: $type, value: $value, value_type: $value_t})
                    MERGE (f)-[:CONTAINS]->(var)
                    """
                    graph.query(query,{
                        "parentname":parent,
                        "name":name,
                        "type":var_type,
                        "value":value,
                        "value_t":value_type,
                    })
                # else:
                #     query="""
                #         MERGE (var: Variable {name: $name, type: $type, value: $value, value_type: $value_t})
                #     """
                #     graph.query(query,{
                #         "name":name,
                #         "type":var_type,
                #         "value":value,
                #         "value_t":value_type,
                #     })

    #--- for attributes
    for component in codebase["components"]:
        debug_logs.append("#DEBUG checking components")
        name=component["name"]
        properties=component["properties"]
        callback=component["callbacks"]
        parent=component["parent"]
        if parent == None:
            debug_logs.append("#DEBUG checking components - no parent")
            query="""
            MERGE (Fr: Frontend {name: $name, properties: $properties})
            """
            if callback:
                for call in callback:
                    extra="""
                    MERGE (Fr: Frontend {name: $name, properties: $properties})
                    MATCH (func: Function {name: $funcname, params: $params, type: $type}})
                    MERGE (Fr)-[:CALLS]->(func)
                    """
                    graph.query(extra,{
                        "name":name,
                        "properties":properties,
                        "funcname":call["name"],
                        "params":call["params"],
                        "type":call["type"]
                    })
            else:
                graph.query(
                    query,{
                        "name":name,
                        "properties":properties,
                    }
                )
        else:
            debug_logs.append("#DEBUG checking components - parent")
            query="""
            MATCH (f:Function {name: $parentname})
            MERGE (Fr: Frontend {name: $name, properties: $properties})
            MERGE (f)-[:CONTAINS]->(Fr)
            """
            if callback:
                for call in callback:
                    extra="""
                    MATCH (f:Function {name: $parentname})
                    MERGE (Fr: Frontend {name: $name, properties: $properties})
                    MERGE (func: Function {name: $funcname, params: $params, type: $type})
                    MERGE (Fr)-[:CALLS]->(func)
                    MERGE (f)-[:CONTAINS]->(Fr)
                    """
                    graph.query(extra,{
                        "parentname":parent,
                        "name":name,
                        "properties":properties,
                        "funcname":call["name"],
                        "params":call["params"],
                        "type":call["type"]
                    })
            else:
                graph.query(
                    query,{
                        "parentname":parent,
                        "name":name,
                        "properties":properties,
                    }
                )

    #--- for imports
    for imports in codebase["imports"]:
        debug_logs.append("#DEBUG checking imports")
        source=imports["from"]
        import_items = imports["import_items"] #------------- FOR NOW THIS IS A STRING
        parent=imports["parent"]
        if parent == None:
                query="""
                MATCH (f:File {name: $filename})
                MERGE (imp: Import {name: $name, source: $source, import_items: $imports })
                MERGE (f)-[:IMPORTS]->(imp)
                """
                graph.query(query,{
                    "filename":file,
                    "name":source,
                    "source":source,
                    "imports":import_items
                })
        else:
            if parent:
                query="""
                MATCH (f:Function {name: $parentname})
                MERGE (imp: Import {name:$name, source: $source, import_items: $imports })
                MERGE (f)-[:IMPORTS]->(imp)
                """
                graph.query(query,{
                    "parentname":parent,
                    "name":source,
                    "source":source,
                    "imports":import_items
                })

    log_content = "\n".join(debug_logs)
    console.print(
        Panel(log_content, title="[bold]DEBUG[/bold]", style="yellow", border_style="yellow")
    )

def new_graph_creation(file_name:str) -> None:
    """
    Creates AST graph and stores in Neo4J database
    """

    graph=Neo4jGraph()
    #Kill everything in the graph:
    graph.query("MATCH (n) DETACH DELETE n")
    console = Console()
    console.print("[bold magenta] Deleted everything in graph..... [/bold magenta]")


    codebase=tree_splitter(file_name)
    # console.print(
    #     Panel(
    #         f"[bold green] {json.dumps(codebase,indent=2)}[/bold green]"
    #     )
    # )
    #---- Create the file node
    create_file= "MERGE (f:File {name: $filename})"

    graph.query(create_file,{"filename":file_name})

    #--- Add functions
    functions(file_name,codebase["functions"],graph)

    #--- for variables
    variables(file_name,codebase["variables"],graph)

    #--- for attributes
    components(file_name,codebase["components"],graph)
    
    #--- for imports
    if codebase["imports"]:
        imports(file_name,codebase["imports"],graph)

    console.print("[bold green]✅ Graph creation completed successfully![/bold green]")

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
    UNWIND $batch AS item
    MERGE (func:Function {
        id: item.id,
        name: item.name,
        params: item.params,
        type: item.type
    })
    WITH item, func

    FOREACH (_ IN CASE WHEN item.is_top_level = true THEN [1] ELSE [] END |
        MERGE (f:File {name: $file_name})
        MERGE (f)-[:CONTAINS]->(func)
    )

    FOREACH (_ IN CASE WHEN item.is_top_level = false AND item.parent_id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (parent:Function {id: item.parent_id})
        MERGE (parent)-[:CONTAINS]->(func)
    )
    """
    graph.query(query_create_node,{"batch":all_funcs,"file_name":file_name})

def variables(file_name:str, variables:list[dict],graph:Neo4jGraph): #new
    all_variables= []
    for var in variables:
        for name in var["names"]:
            ancestry_names= [a["name"] for a in var["ancestry"]]
            id=create_id(file_name,name,ancestry_names)
            parent_id=None
            if var["parent"]:
                parent_ancestry= [a["name"] for a in var["ancestry"][:-1]]
                parent_id= create_id(file_name, var["parent"]["name"],parent_ancestry)
            
            all_variables.append({
                "id": id,
                "name":name,
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
    
    FOREACH (_ IN CASE WHEN item.is_top_level = false AND item.parent_id IS NOT NULL THEN [1] ELSE [] END |
        MERGE (parent:Function {id: item.parent_id})
        MERGE (parent)-[:DEFINES]->(v)
    )

    """,{"batch":all_variables,"file_name":file_name})

def old_components(file_name,components:list[dict],graph:Neo4jGraph): #new>old
    all_ui=[]
    for comp in components:
        ancestry_dicts = comp.get("ancestry",[])
        ancestry_names = [a["name"] for a in ancestry_dicts]
        id=create_id(file_name,comp["name"],ancestry_names)
        parent_id=None
        parent_info = ancestry_dicts[-1] if ancestry_dicts else None
        parent_type = None
        comp["top_level"] = not bool(comp["ancestry"])
        if parent_info:
            parent_type = parent_info['type']
            parent_ancestry=[a["name"] for a in comp["ancestry"][:-1]]
            if parent_type == "ui":
                parent_id = create_id(file_name, parent_info["name"], parent_ancestry)
            elif parent_type == "function":
                parent_id = create_id(file_name, parent_info["name"], parent_ancestry)
    
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
        SET ui.name = item.name,
            ui.properties = item.properties
        
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
    
    # Handle callbacks - link Frontend components to the Functions they call
    callbacks_batch = []
    for comp in components:
        ancestry_dicts = comp.get("ancestry",[])
        ancestry_names = [a["name"] for a in ancestry_dicts]
        ui_id = create_id(file_name, comp["name"], ancestry_names)
        
        for callback in comp.get("callbacks", []):
            if callback.get("name"):
                # Find the callback function's ancestry to create proper ID
                callback_ancestry = callback.get("ancestry", [])
                callback_ancestry_names = [a["name"] for a in callback_ancestry]
                callback_id = create_id(file_name, callback["name"], callback_ancestry_names)
                
                callbacks_batch.append({
                    "ui_id": ui_id,
                    "callback_id": callback_id,
                    "callback_name": callback["name"]
                })
    
    if callbacks_batch:
        graph.query("""
            UNWIND $batch as item
            MATCH (ui: Frontend {id: item.ui_id})
            MERGE (func: Function {id: item.callback_id})
            ON CREATE SET func.name = item.callback_name
            MERGE (ui)-[:CALLS]->(func)
        """, {"batch": callbacks_batch})

def components(file_name, components: list[dict], graph: Neo4jGraph):
    all_ui = []

    for comp in components:
        ancestry_names = [a["name"] for a in comp.get("ancestry", [])]
        ui_id = create_id(file_name, comp["name"], ancestry_names)


        all_ui.append({
            "id": ui_id,
            "name": comp["name"],
            "properties": comp["properties"],
            "ui_parent": (
                create_id(file_name, comp["ui_parent"], ancestry_names[:-1])
                if comp.get("ui_parent") else None
            ),
            "rendered_by": comp.get("rendered_by"),
            "handlers":[a["event"] for a in comp["handlers"]]
        })

    graph.query("""
    UNWIND $batch AS item

    MERGE (ui:Frontend {id: item.id})
    SET ui.name = item.name,
        ui.properties = item.properties,
        ui.handlers = item.handlers

    WITH ui, item

    FOREACH (h IN item.handlers |
        MERGE (handler:Handler {id: h.id, event:h.event})
        MERGE (ui)-[:HAS_HANDLER]->(handler)
    
        FOREACH(_ IN CASE WHEN h.function_id IS NOT NULL THEN [1] ELSE [] END |
            MERGE (f: Function {id: h.function_id})
            MERGE (handler)-[:CALLS]->(f)        
        )
    )
    
    FOREACH (_ IN CASE WHEN item.ui_parent IS NOT NULL THEN [1] ELSE [] END |
        MERGE (parent:Frontend {id: item.ui_parent})
        MERGE (parent)-[:CONTAINS]->(ui)
    )

    FOREACH (_ IN CASE WHEN item.rendered_by IS NOT NULL THEN [1] ELSE [] END |
        MERGE (f:Function {id: item.rendered_by})
        MERGE (f)-[:RENDERS]->(ui)
    )

    """, {"batch": all_ui})

    graph.query("""
    UNWIND $batch AS item

    MERGE (ui:Frontend {id: item.id})
    SET ui.name = item.name,

    WITH ui, item

    FOREACH (h IN item.handlers |
        MERGE (handler:Handler {id: h.id, event:h.event})
        MERGE (ui)-[:HAS_HANDLER]->(handler)
    
        FOREACH(_ IN CASE WHEN h.function_id IS NOT NULL THEN [1] ELSE [] END |
            MERGE (f: Function {id: h.function_id})
            MERGE (handler)-[:CALLS]->(f)        
        )
    )""", {"batch": all_ui})

def imports(file_name:str, imports:list[dict],graph:Neo4jGraph): # new
    all_imports=[]
    for imp in imports:
        target_name = f"./victim-site{imp['from'].strip('.')}"
        if not target_name.endswith(('.js', '.jsx')): target_name += ".jsx"
        
        for item in imp["import_items"]:
            target_id = create_id(target_name,item,[])
            all_imports.append({
                "id": target_id,
                "func_name":item
            })
    query_import = """
    UNWIND $batch AS item
    MERGE (file:File {name: $filename})
    MERGE (func:Function {id: item.id})
    ON CREATE SET func.name = item.func_name
    MERGE (file)-[:IMPORTS]->(func)
    """
    graph.query(query_import, {"batch": all_imports, "filename": file_name})

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
        "parent": None,
        "ancestry":[]
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
    if values.get("name"):
        function["name"] = values["name"][0].text.decode("utf8")
    if values.get("params"):
        function["params"] = values["params"][0].text.decode("utf8")
    
    ancestry = get_ancestry(node)
    function["ancestry"] = ancestry
    if not ancestry:
        function["top_level"] = True
    else:
        function["parent"] = ancestry[-1]

    def find_nested_functions(n):
        nested = []
        for child in n.children:
            if child.type in FUNCTIONS and child.child_count > 0:
                nested.append(get_function(child))
            else:
                nested.extend(find_nested_functions(child))
        return nested
    
    function["nested"] = find_nested_functions(node)
    
    return function

def get_frontend(node:Node):
    
    def get_enclosing_function_id(node: Node, file_name: str):
            current = node.parent

            while current:
                if current.type in ["function_declaration", "function", "arrow_function"]:
                    name_node = current.child_by_field_name("name")
                    if name_node:
                        func_name = name_node.text.decode("utf8")
                        return create_id(file_name, func_name, [])
                current = current.parent
            return None

    def get_handler(attr_node: Node, ui_id:str):
        name_node = None
        for child in attr_node.children:
            if child.type == "property_identifier":
                name_node = child
                break
        if not name_node:
            return None
    
        attr_name = name_node.text.decode("utf8")
        if not attr_name.startswith("on"):
            return None
        
        event = attr_name[2:].lower()
        for child in attr_node.children:
            if child.type == "jsx_expression":
                for expr in child.children:
                    if expr.type in ["arrow_function","function_expression"]:
                        return {
                            "event":event,
                            "function_type":"inline",
                            "code":expr.text.decode("utf8")
                        }
                    
                    if expr.type == "identifier":
                        return {
                            "event": event,
                            "function_type": "named",
                            "function_name":expr.text.decode("utf8")
                        }
        return None

    query=Query(JSLANGUAGE,"""
        (identifier)@name
        (jsx_attribute)@properties
    """)

    attribute={
        "name":"",
        "properties":[],
        "callbacks":[],
        "parent":None,
        "ancestry":[],
        "top_level":False,
        "handlers":[]
    }

    cursor = QueryCursor(query)
    values=cursor.captures(node)
    
    if values.get("properties"):
        for i in values.get("properties"):
            attribute["properties"].append(i.text.decode("utf8"))
            handler = get_handler(i)
            if handler:
                attribute["handlers"].append(handler)
    
    if values.get("name"):
        attribute["name"] = values["name"][0].text.decode("utf8")
    
    ancestry = get_ancestry(node)
    attribute["ancestry"] = ancestry

    # JSX parent (UI > UI)
    ui_parent = None
    for a in reversed(ancestry):
        if a["type"] == "ui":
            ui_parent = a["name"]
            break


    attribute["ui_parent"] = ui_parent

    # Function render root (Function > UI)
    attribute["rendered_by"] = get_enclosing_function_id(
        node,
        FILE_NAME
    )
    if not ancestry:
        attribute["top_level"]=True

    attribute["handlers"].append({
        "id":handler_id(FILE_NAME,event,ui_id),
        "event":event,
        "function_id": (
            named_function_id if handler["function_type"] == "named"
            else inline_function_id(FILE_NAME, handler_id)
        )
    })
    attr_copy = attribute.copy()
    attr_copy["handlers"] = [
        {
            "event": h["event"],
            "handler_type": h["handler_type"],
            "handler_code": h["handler_node"].text.decode("utf8")[:50] + "..." 
                if len(h["handler_node"].text.decode("utf8")) > 50 
                else h["handler_node"].text.decode("utf8")
        } if isinstance(h, dict) and "handler_node" in h else h
        for h in attribute.get("handlers", [])
    ]

    CONSOLE.print(f"[yellow] {json.dumps(attr_copy, indent=4)} [/yellow]")
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
        "parent":None,
        "ancestry": []
    }
    if node.parent.type == "program":
        variable["top_level"]=True
    else:
        parent_node=get_parent_function(node)
        for child in parent_node.children:
            if child.type == "identifier":
                variable["parent"]= child.text.decode("utf8")
    
    ancestry = get_ancestry(node)
    variable["ancestry"] = ancestry
    variable["top_level"] = not bool(ancestry)
    if ancestry:
        variable["parent"] = ancestry[-1]

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

new_graph_creation(FILE_NAME)