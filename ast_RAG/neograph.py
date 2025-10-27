from ast_parser import process_directory,tree_splitter #takes root_dir and ast_dir
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import json
from rich.console import Console
from rich.panel import Panel


load_dotenv()
root_dir = "./samples"
ast_dir = "./samples_ast"
FILE_NAME="../test-project/src/App.jsx"

def using_boss_code(root_dir,ast_dir):
    try:
        graph = Neo4jGraph()
        codebase=process_directory(root_dir=root_dir,ast_dir=ast_dir)
        file=codebase["file_name"]
        graph.query(
            "MERGE (f:File {name: $filename})",{"filename":file}
            )
        for i in codebase["imports"]:
            hook=False
            if i in codebase["hook_usage"]:
                hook=True
            query="""
            MERGE (imp: Import {name: $import_name,hook: $hook})
            WITH imp
            MATCH (f:File {name: $filename})
            MERGE (f)-[:IMPORTS]->(imp)
            """
            graph.query(query,{"filename":file,"import_name":i,"hook":hook})

        for k,v in codebase["return"].items():
            if k=="always_rendered":
                render="always rendered"
                for items in v:
                    query="""
                    MERGE (item: FRONTEND {name: $attribute, render: $render})
                    WITH item
                    MATCH (f:File {name: $filename})
                    MERGE (f)-[:DISPLAYS]->(item)
                    """
                    graph.query(query,{"attribute":items,"render":render,"filename":file})
            if k == "conditional_rendering":
                render="conditionally rendered"
                for items in v:
                    query="""
                    MERGE (item: FRONTEND {name: $attribute, render: $render})
                    WITH item
                    MATCH (f:File {name: $filename})
                    MERGE (f)-[:DISPLAYS]->(item)
                    """
                    graph.query(query,{"attribute":items,"render":render,"filename":file})

    except ValueError as e:
        print("error ", e)

def using_tree_splitter():
    graph=Neo4jGraph()
    #Kill everything in the graph:
    graph.query("MATCH (n) DETACH DELETE n")
    console = Console()
    debug_logs=[]
    console.print("[bold magenta] Deleted everything in graph..... [/bold magenta]")


    codebase=tree_splitter(FILE_NAME)
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


# using_boss_code(root_dir,ast_dir)
using_tree_splitter()


"""
variables inside functions: -> find parent function

functions inside functions: -> nested

components inside function -> nested

"""