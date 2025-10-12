from ast_parser import process_directory #takes root_dir and ast_dir
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import json
load_dotenv()
root_dir = "./samples"
ast_dir = "./samples_ast"

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





"""
{'always_rendered': {'button', 'div', 'img', 'code', 'a', 'h1', 'p'}, 
'conditional_rendering': [], 'conditional_components': set()}}

File: Calls -> Functions
File: Imports -> imports (property hook) or not

File: frontend -> for all stuff pull up and always rendered
File: frontend -> for all stuff pull up and conditional rendered


"""