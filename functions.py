# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tree_sitter import Language, Parser, Query, QueryCursor, Node # as TSNode, but planning to delete functions anyway
from streamlit_agraph import agraph, Edge, Config, Node as ANode
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from browser_use import Agent, ChatGoogle,Browser
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
import tree_sitter_javascript as tsj
from rich.console import Console
from google.genai import types
from dotenv import load_dotenv
from rich.panel import Panel
from google import genai
import streamlit as st
import os,yaml,json
import subprocess
import chromadb
import re
import shutil
import subprocess
import uuid
import csv

# Load .env from the current directory where main.py is run
load_dotenv()

#TODO: Delete all imports, functions and variables that are not needed in the newest archi

FILE_NAME="../test-project/src/App.jsx"
JSLANGUAGE = Language(tsj.language()) #creates language
FUNCTIONS= ["arrow_function","function_declaration","function"]
VARIABLES= ["array_pattern"]
gemini_API=os.getenv("GEMINI_API_KEY")
CONSOLE= Console()

def ast_rag(file:str):
    """ 
    Breaks inputted file path into ast structure 
    
    ========

    Returns ast structure
    """    
    # Use relative path to parser_test.js from the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser_path = os.path.join(current_dir, "parser_test.js")
    command = ["node",parser_path,file]
    values=subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )
    return values.stdout

#original
def embed_ast_original(file:str) -> None:
    """ 

    1) Embeds a file and creates database: Code_database
    2) Stores vectors under collection ast

    """

    client=genai.Client()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_client=chromadb.Client(path=os.path.join(current_dir, "Code_database")) # type: ignore
    collection=chroma_client.get_or_create_collection(name="ast")
    
    loader=TextLoader(file)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits= text_splitter.split_documents(docs)
    
    for i, chunk in enumerate(splits):
        chunk.metadata["document_type"]= "Code data"
        chunk.metadata["chunk_id"]=i

    chunks= [e.page_content for e in splits]
    result=client.models.embed_content(
        model="gemini-embedding-001",
        contents = [e.page_content for e in splits],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",output_dimensionality=3072)
    )
    gemini_embeddings= [e.values for e in result.embeddings] # type: ignore

    collection.add(
        embeddings=gemini_embeddings,
        documents=chunks,
        metadatas=[chunk.metadata for chunk in splits],
        ids=[f"code_chunk_{chunk.metadata['chunk_id']}" for chunk in splits]
    )

#new
def embed_ast(file: str) -> None:
    """
    1) Embeds a file and creates a Chroma database (persistent if possible)
    2) Stores vectors under collection 'ast'
    3) Falls back to in-memory Chroma if persistent DB cannot be written
    """

    st.write(gemini_API)
    client = genai.Client(api_key=gemini_API)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "Code_database")

        os.makedirs(db_path, exist_ok=True)

        if not os.access(db_path, os.W_OK):
            raise PermissionError(f"Database path not writable: {db_path}")

        chroma_client = chromadb.PersistentClient(path=db_path)
        st.info(f"Using persistent Chroma database at: {db_path}")

    except Exception as e:
        st.warning(f"Falling back to in-memory Chroma (reason: {e})")
        chroma_client = chromadb.Client()

    loader = TextLoader(file)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    for i, chunk in enumerate(splits):
        chunk.metadata["document_type"] = "Code data"
        chunk.metadata["chunk_id"] = i

    chunks = [e.page_content for e in splits]

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunks, #type: ignore
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=3072
        ),
    )

    gemini_embeddings = [e.values for e in result.embeddings] # type: ignore

    collection = chroma_client.get_or_create_collection(name="ast")

    collection.add(
        embeddings=gemini_embeddings, # type: ignore
        documents=chunks,
        metadatas=[chunk.metadata for chunk in splits],
        ids=[f"code_chunk_{chunk.metadata['chunk_id']}" for chunk in splits],
    )

    st.success("Embedding completed and stored successfully!")

def cycle(test_path:str):
    """
    Param: path to folder to be created to store tests
    1) Cycle through every component and generate instructions
    2) Store files in tests folder
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    #initialize the client and stuff
    client=genai.Client()
    chroma_client= chromadb.PersistentClient(path=os.path.join(current_dir, "Code_database"))
    collection = chroma_client.get_collection(name="ast")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                google_api_key=gemini_API, #type: ignore
                                model_kwargs={ #type: ignore
                                    "response_mime_type":"application/yaml"
                                })
    prompt = ChatPromptTemplate.from_template("""
    You are a test automation expert. Generate test instructions in a valid YAML format.

    Context: {context}
    Component: {input}

    Return a single, valid YAML document with this exact structure and nothing else:
    ---
    component: component_name
    url: http://localhost:5173/
    test_steps:
    - step: 1
        action: navigate
        instruction: Open the application
        target: "http://localhost:5173/"
        expected: Page loads successfully
    - step: 2
        action: click
        instruction: Click the submit button
        selector: "#submit-btn"
        expected: Form submits successfully
    

    Requirements:
    - Each instruction must be ONE clear action
    - Include specific selectors (id, class, data-testid, or text)
    - Use action types: navigate, click, type, verify, wait, select
    """)

    CONSOLE.print("[bold yellow] Making message [/bold yellow]")
    # document_chain = create_stuff_documents_chain(llm,prompt)
    document_chain = "deleted"
    
    def access_code(instructions):
        """
        Gets the code and makes instructions
        """

        query=instructions
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="CODE_RETRIEVAL_QUERY",
                output_dimensionality=3072 # Must match the dimension used for storage
            )
        )
        CONSOLE.print("[bold green] embedding.... [/bold green] ")
        query_embedding = [e.values for e in result.embeddings] # type: ignore

        results = collection.query( #queries the thing
            query_embeddings=query_embedding, # Use query_embeddings instead of query_texts # type: ignore
            n_results=2
        )
        
        docs=[]
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i], # type: ignore
                metadata=results["metadatas"][0][i] # type: ignore
            )
            docs.append(doc)
        CONSOLE.print("[green] making IDs [/green]")

        CONSOLE.print("[yellow]invoke message [/yellow]")
        response = document_chain.invoke({
            "input": query,
            "context": docs
        })
        CONSOLE.print(
            Panel(
            response,title="response",expand=True))
        return (yaml.safe_load(response))

    #helper function for cycle()
    def unique_file(name,existing_files):
        """
        Makes a unique file name to stop overwrites
        """

        count=1
        file=f"{name}.yaml"
        while file in existing_files:
            file=f"{name}[{count}].yaml"
            count+=1
        
        existing_files.add(file)
        return file

    files=[]
    os.makedirs(test_path, exist_ok=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_struct_path = os.path.join(current_dir, "code_structure.json")

    with open(code_struct_path,"r") as f:
        data:dict=json.load(f)
    existing_files = set()

    for key,values in data.items():
        name, extension=os.path.splitext(key)
        directory=f"{name}[{extension}]"
        full_path=os.path.join(test_path,directory)
        os.makedirs(full_path,exist_ok=True)
        for index,i in enumerate(values["components"]):

            # Check if testableAttributes exists and is not empty
            if "testableAttributes" in i and i["testableAttributes"] and len(i["testableAttributes"]) > 0:
                instruction= f"please give instructions to test the component {i}"
                yaml_data=access_code(instruction)
                filename = unique_file(i['name'], existing_files)
                files.append(filename)
                final_path=os.path.join(full_path,filename)
                with open(final_path,"w") as f:
                    if yaml_data:
                        yaml.dump(yaml_data,f,default_flow_style=False, sort_keys=False)

async def test_browser_use(limit=None,headless:bool = False, test_path:str = None)->list[dict]: # type: ignore
    """ Runs agent. If input not None, will limit number of tests """
    path=os.path.join("tests",test_path)
    directory= os.listdir(path)
    success_files=[]
    count=0

    for file in directory:
        with open(os.path.join(path,file),"r") as f:
            data=yaml.safe_load(f)
        task=str(yaml.dump(data["test_steps"], default_flow_style=False, sort_keys=False))
        browser=Browser(
            headless=headless,
        )
        try:
            agent = Agent(
                task=task,
                llm=ChatGoogle(model="gemini-2.5-flash"),
                browser=browser
            )
            history = await agent.run()
            others={
                "structured_output":history.structured_output,
                "action":history.action_names(),
                "extracted":history.extracted_content(),  
                "errors":history.errors(),                  
                "actions":history.model_actions(),           
                "model_output":history.model_outputs(),          
                "last action":history.last_action(),          
            }

            success={"path":test_path,"name":file,"success":history.is_successful(),**others}
            success_files.append(success)
        except Exception as e:
            st.warning(str(e))
        count+=1
        if limit:
            if count == limit:
                break
        

    CONSOLE.print("[bold magenta] Failures: [/bold magenta]")
    for i in success_files:
        if i["success"]==False:
            print(i["name"])
    return success_files

async def results_writer(results: list[dict[str:str|bool]])->None: # type: ignore
    """
    Takes list of dictionaries and writes yaml files to new folder called "results"
    dictionary format: path: full path to be saved
                       name: str
                       success: bool
                       **kwargs
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_path= os.path.join(current_dir,"results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for i in results:
        directory=os.path.join(results_path,i["path"])
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = os.path.join(directory, i["name"])
        output_data = i.copy()
        if "path" in output_data:
            del output_data["path"]

        output = "\n".join(f"{key}: {value}" for key, value in output_data.items())
        with open(file_path,"w") as f:
            f.write(output)

def get_graph():
    """
    Obtains graph from Neo4J database and returns all nodes and edges

    Returns list,list, 
    dictionary{
        string: string
    }
    """
    graph=Neo4jGraph()
    nodes = {} 
    edges = []
    node_types = {}
    colors = ["#FFC0CB", "#ADD8E6", "#90EE90", "#FFD700", "#F08080", "#B0E0E6", "#DDA0DD"]

    # 2. Updated Query: explicitly return n (start), type(r) (rel), m (end)
    # This prevents ambiguity about what r[0] or r[2] is.
    result = graph.query("""
        MATCH (n)-[r]->(m)
        RETURN n, type(r) as rel_type, m
        LIMIT 25
    """)

    for record in result:
        source_data = record['n'] # This is a dictionary of properties
        target_data = record['m']
        rel_type = record['rel_type']

        # --- HELPER: Safe ID and Label Extraction ---
        def get_node_info(node_dict):
            # Use the internal ID if available, otherwise hash the content
            # Most Neo4j dicts from LangChain include an 'id' key
            node_id = str(node_dict.get('id', hash(json.dumps(node_dict, sort_keys=True))))
            
            # Smart Labeling: Try specific fields first, fall back to ID
            # We explicitly AVOID using 'CODE' as the label to prevent the image error
            label = node_dict.get('NAME') or \
                    node_dict.get('FULL_NAME') or \
                    node_dict.get('TYPE_FULL_NAME') or \
                    node_dict.get('label') or \
                    f"Node {node_id}"
            
            # Truncate label if it's too long (e.g. for Java signatures)
            if len(label) > 20:
                label = label[:20] + "..."
                
            return node_id, label, node_dict.get('type', 'Unknown')

        # Process Source Node
        s_id, s_label, s_type = get_node_info(source_data)
        if s_type not in node_types:
            node_types[s_type] = colors[len(node_types) % len(colors)]

        if s_id not in nodes:
            nodes[s_id] = ANode(
                id=s_id,
                label=s_label, # CLEAN LABEL
                size=25,
                shape="dot",
                color=node_types[s_type],
                # Store full data in title for hover effect, NOT in label
                title=json.dumps(source_data, indent=2) 
            )

        # Process Target Node
        t_id, t_label, t_type = get_node_info(target_data)
        if t_type not in node_types:
            node_types[t_type] = colors[len(node_types) % len(colors)]

        if t_id not in nodes:
            nodes[t_id] = ANode(
                id=t_id,
                label=t_label, # CLEAN LABEL
                size=25,
                shape="dot",
                color=node_types[t_type],
                title=json.dumps(target_data, indent=2)
            )

        # Add Edge
        edges.append(Edge(
            source=s_id,
            target=t_id,
            label=rel_type,
            type="CURVE_SMOOTH"
        ))

    return list(nodes.values()), edges, node_types

#------ Graph Creation ---
def graph_creation(file_name:str) -> None:
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
        name=function["name"] # type: ignore
        params=function["params"] # type: ignore
        func_type=function["type"] # type: ignore
        if function.get("top_level"): # type: ignore
            
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


        if function["nested"]: # type: ignore
            nested_func(name,function["nested"]) # type: ignore

    #--- for variables
    for variable in codebase["variables"]:
        debug_logs.append("#DEBUG Checking variables")
        names=variable["names"] # type: ignore
        var_type=variable["type"] # type: ignore
        value=variable["value"] # type: ignore
        value_type=variable["value_type"] # type: ignore
        for name in names:
            if variable.get("top_level"): # type: ignore
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
                parent = variable["parent"] # type: ignore
                
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
        name=component["name"] # type: ignore
        properties=component["properties"] # type: ignore
        callback=component["callbacks"] # type: ignore
        parent=component["parent"] # type: ignore
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
                        "funcname":call["name"], # type: ignore
                        "params":call["params"], # type: ignore
                        "type":call["type"] # type: ignore
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
                        "funcname":call["name"], # type: ignore
                        "params":call["params"], # type: ignore
                        "type":call["type"] # type: ignore
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
        source=imports["from"] # type: ignore
        import_items = imports["import_items"] #------------- FOR NOW THIS IS A STRING # type: ignore
        parent=imports["parent"] # type: ignore
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
    if node.parent.type == "program": # type: ignore
        function["top_level"]=True
    else:
        parent_node=get_parent_function(node)
        for child in parent_node.children: # type: ignore
            if child.type == "identifier":
                function["parent"]= child.text.decode("utf8") # type: ignore
    # FIX: Take only the FIRST identifier (the function name)
    if values.get("name"):
        function["name"] = values["name"][0].text.decode("utf8") # type: ignore
    if values.get("params"):
        function["params"] = values["params"][0].text.decode("utf8") # type: ignore

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
        attribute["name"] = values["name"][0].text.decode() # type: ignore
    
    parent_node=get_parent_function(node)
    for child in parent_node.children: # type: ignore
        if child.type == "identifier":
            attribute["parent"]= child.text.decode("utf8") # type: ignore
    
    for property in values.get("properties",[]):
        attribute["properties"].append(property.text.decode()) # type: ignore
        
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
    if node.parent.type == "program": # type: ignore
        variable["top_level"]=True
    else:
        parent_node=get_parent_function(node)
        for child in parent_node.children: # type: ignore
            if child.type == "identifier":
                variable["parent"]= child.text.decode("utf8") # type: ignore

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
            variable["names"].append(left_side.text.decode("utf8")) # type: ignore
        elif variable["type"] == "array_destructure":
            for child in left_side.children:
                if child.type=="identifier":
                    variable["names"].append(child.text.decode("utf8")) # type: ignore

        elif variable["type"] == "object_destructure":
            for child in left_side.children:
                if child.type == "identifier":
                    variable["names"].append(child.text.decode("utf8")) # type: ignore
                elif child.type == "shorthand_property":
                    for sub in child.children:
                        if sub.type == "identifier":
                            variable["names"].append(sub.text.decode("utf8")) # type: ignore

    if right_side:
        variable["value"] = right_side.text.decode("utf8") # type: ignore
     
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
            call["function_name"] = child.text.decode("utf8") # type: ignore
            break
    
    # Find arguments
    for child in node.children:
        if child.type == "arguments":
            call["arguments"].append(child.text.decode("utf8")) # type: ignore
        if child.type in FUNCTIONS:
            call["function_type"]=child.text.decode("utf8") # type: ignore

    call["full_text"] = node.text.decode("utf8") # type: ignore
    
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
                import_statement["import_items"].append(child.text.decode("utf8")) # type: ignore
    
    for child in node.children:
        if child.type == "string": #imports are strings in javascript
            import_statement["from"]= child.text.decode("utf8") # type: ignore

    parent=get_parent_function(node)
    if parent:
        for child in parent.children:
            if child.type=="identifier":
                import_statement["parent"]=child.text.decode("utf8") # type: ignore
                break
    return import_statement

#----------- CPG creation -------
from cpg_folder.joern_cpg_to_neo4j.cpg_to_neo4j import cpgToNeo4j

# from cpg to neo4j only, we're running to WEBHOOK DATABASE
def cpg_to_neo4j(config:dict) -> None:
    export_path = config.get("export_path")     # Source (UUID folder)
    import_path = config.get("neo4j_import_path") # Dest (Neo4j Import)

    # --- FIX: FORCE CLEANUP DESTINATION ---
    # Delete old CSVs in the Neo4j import folder to prevent "Files already up to date" error
    # and ensure we aren't loading stale data.
    if os.path.exists(import_path):
        for file in os.listdir(import_path):
            if file.endswith(".csv") or file == "import_header.csv":
                try:
                    os.remove(os.path.join(import_path, file))
                except Exception as e:
                    print(f"Warning: Could not clear old file {file}: {e}")

    # Pipeline: From CPG to Neo4j
    pipe = cpgToNeo4j(
        config.get("neo4j_uri"),
        config.get("neo4j_user"),
        config.get("neo4j_password"),
    )

    pipe.copy_data_to_neo4j_import_folder(
        config.get("export_path"),
        config.get("neo4j_import_path")
    )
    pipe.upload_nodes(
        config.get("export_path")
    )
    pipe.upload_edges(
        config.get("export_path")
    )

def joern_pipeline(input_path:str, config:dict):
    """
    run all the commands to use joern to parse codebase into cpg. Original commands can be found in cpg.py
    """
    # Define paths (Use absolute paths to avoid confusion)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root of project
    cpg_file_path = os.path.join(base_dir, "cpg_folder", "cpg_creations")
    output_folder = config.get("export_path")
    joern_base_path = config.get("joern_path")

    # Clean up old folders
    if os.path.exists(cpg_file_path):
        os.remove(cpg_file_path)
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(os.path.dirname(cpg_file_path), exist_ok=True)

    # set up Java 21 environment
    my_env = os.environ.copy()
    my_env["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21" 
    my_env["PATH"] = my_env["JAVA_HOME"] + r"\bin;" + my_env.get("PATH", "")

    # Prepare Joern executables
    exec_parse = "joern-parse.bat" if os.name == 'nt' else "joern-parse"
    exec_export = "joern-export.bat" if os.name == 'nt' else "joern-export"

    path_to_parse = os.path.join(joern_base_path, exec_parse)
    path_to_export = os.path.join(joern_base_path, exec_export)

    # execute: joern parse (code -> binary)
    command1 = [path_to_parse, input_path, "--output", cpg_file_path]
    print(f"Running Joern parse on {input_path}")
    result1 = subprocess.run(
        command1, 
        capture_output=True, 
        text=True, 
        env=my_env, 
        cwd=joern_base_path 
    )

    if result1.returncode != 0:
        print(f"joern-parse failed: {result1.stderr}")
        # Add a hint about Java if it fails here

    # execute: joern export (binary -> neo4jcsv)
    command2 = [path_to_export, cpg_file_path, "--out", output_folder, "--repr", "all", "--format", "neo4jcsv"]
    print(f"Running joern export on {input_path}")
    result2 = subprocess.run(
        command2, 
        capture_output=True, 
        text=True, 
        env=my_env,
        cwd=joern_base_path  
    )
    
    if result2.returncode != 0:
        print(f"joern-export failed: {result2.stderr}")

    print("Pushing to neo4j..")
    try:
        cpg_to_neo4j(config=config)
    except Exception as e:
        st.error(f"Failed to push to Neo4j: {e}")

# for initial load
def create_cpg_repo(repo_path:str, config:dict):
    """
    create cpg for the entire repo, then convert cpg into neo4j
    """
    print(f"Running Joern pipeline for entire {repo_path}..")
    base_export_dir = config.get("export_path")
    unique_id = str(uuid.uuid4())
    unique_export_path = os.path.join(base_export_dir, unique_id)

    run_config = config.copy()
    run_config["export_path"] = unique_export_path

    try:
        joern_pipeline(repo_path, config)
    except Exception as e:
        print(f"Error in create_cpg_repo: {e}")
        raise e

# TODO: Move delete_file_nodes here and refactor the code in main_server.py 

# for updating the nodes
def create_cpg_files(changed_files:list, config:dict):
    """
    create sub cpg graph for changed nodes, then convert cpg into neo4j, merge it into the main graph
    """
    print(f"Running joern pipeline only for changed files: {changed_files}")
    success_files = 0
    base_export_dir = config.get("export_path")
    for file_path in changed_files:
        unique_id = str(uuid.uuid4())
        unique_export_path = os.path.join(base_export_dir, unique_id)

        run_config = config.copy()
        run_config["export_path"] = unique_export_path

        try:
            print(f"Processing {os.path.basename(file_path)} in temp dir: {unique_id}")
            joern_pipeline(file_path, run_config)
            sanitize_cpg_export(unique_export_path, file_path)
            success_files += 1
        except Exception as e:
            print(f"Error in processing file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            try:
                if os.path.exists(unique_export_path):
                    shutil.rmtree(unique_export_path)
            except OSError:
                print(f"Warning: Could not immediately delete temp folder {unique_id}. Windows might clean it up later.")
    print(f"Finished Joern pipeline for changed files: {changed_files}")

def sanitize_cpg_export(export_path, original_filename):
    """
    Reads the Joern CSVs and replaces the absolute temp path with the clean relative filename.
    Example: Replaces "C:/.../uuid-123/App.jsx" with just "App.jsx"
    """
    print(f"Sanitizing CSVs in {export_path}...")
    
    # We primarily care about the FILE nodes and any source file references
    for root, dirs, files in os.walk(export_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                
                # Read the CSV data
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Perform the replacement
                # We assume the absolute path contains the UUID folder structure.
                # A simple heuristic is to replace the full path with the simple filename
                # wherever the full path appears.
                
                # NOTE: We need to be careful. The easiest way is to match 
                # any path ending in the filename separator.
                
                # Simpler approach: normalize slashes and replace
                clean_name = os.path.basename(original_filename) # e.g., "App.jsx"
                
                # This logic assumes Joern outputted the full path. 
                # We simply want the DB to store "App.jsx".
                # We can't regex easily without knowing the exact random path, 
                # but we know the current run's path!
                
                # We passed the full path to Joern, so Joern put that full path in the CSV.
                # We just find that string and replace it.
                current_full_path_win = original_filename.replace("/", "\\")
                current_full_path_unix = original_filename.replace("\\", "/")
                
                if current_full_path_win in content:
                    content = content.replace(current_full_path_win, clean_name)
                if current_full_path_unix in content:
                    content = content.replace(current_full_path_unix, clean_name)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

"""
----- graphrag
"""
# ---- graphRAG to find related components
# need to create a shared label called "TestableComponent"
# MATCH (n) WHERE labels(n) IN [['TEMPLATE_DOM'], ['METHOD'], ['CALL'], ['IDENTIFIER']]
# SET n:TestableComponent


def embed_nodes():
    """
    Creating embeddings for METHOD, CALL and IDENTIFIER nodes
    """
    neo4j_url = "bolt://localhost:7687"
    neo4j_password = "password"

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )

    retrieval_query = """
    RETURN 
        node.CODE AS text,
        score,
        {
            id: elementId(node),
            name:coalesce(node.NAME, node.FULL_NAME, 'Unnamed'),
            labels: labels(node)
        } AS metadata
    """

    retrieval_query_multiple = """
    // 1. ZOOM OUT to Component Root
    OPTIONAL MATCH (node)<-[:AST|CONTAINS*0..20]-(m:METHOD)
    WITH node, score, collect(DISTINCT m) AS methods
    WITH coalesce(head(methods), node) AS root, score

    // 2. GATHER ALL UNIQUE CHILDREN FIRST (Including Routes)
    MATCH (root)-[:CONTAINS|AST*]->(child)
    WHERE 
       // HTML Elements
       (
            child.NAME IN ['a', 'Link', 'img', 'Image', 'input', 'button'] 
            OR child.CODE STARTS WITH '<a' 
            OR child.CODE STARTS WITH '<img' 
            OR child.CODE STARTS WITH '<button' 
            OR child.CODE STARTS WITH '<input'
            OR child.NAME IN ['push', 'navigate', 'redirect', 'go', 'back']
        )
        AND NOT child.NAME IN ['JSXOpeningElement', 'JSXClosingElement']

    WITH root, score, child.CODE as code, head(collect(child)) as unique_node
    
    // 4. COLLECT THE UNIQUE NODES INTO A LIST
    WITH root, score, collect(unique_node) as unique_children

    // 4. CATEGORIZE
    RETURN
        root.CODE as text,
        score,
        {
            id: elementId(root),
            name: root.NAME,
            labels: labels(root),
            
            links: [c IN unique_children 
                    WHERE c.NAME IN ['a', 'Link'] OR c.CODE STARTS WITH '<a' 
                    | {id: elementId(c), code: c.CODE}],

            images: [c IN unique_children 
                     WHERE c.NAME IN ['img', 'Image'] OR c.CODE STARTS WITH '<img' 
                     | {id: elementId(c), code: c.CODE}],
            
            inputs: [c IN unique_children 
                     WHERE c.NAME IN ['input'] OR c.CODE STARTS WITH '<input' 
                     | {id: elementId(c), code: c.CODE}],
            
            buttons: [c IN unique_children 
                      WHERE c.NAME IN ['button'] OR c.CODE STARTS WITH '<button' 
                      | {id: elementId(c), code: c.CODE}],
            
            routes: [c IN unique_children 
                     WHERE c.NAME IN ['push', 'navigate', 'redirect', 'go', 'back'] 
                     | {id: elementId(c), name: c.NAME, code: c.CODE}]
        } as metadata
    """

    vector_store = Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=neo4j_url,
        password=neo4j_password,
        index_name="testable_components",
        node_label="TestableComponent",
        text_node_properties=["NAME", "CODE"],
        embedding_node_property="embedding",
        retrieval_query=retrieval_query
    )

    return vector_store

# using user prompt
def retrieve_components(user_prompt, vector_store, threshold):
    results = vector_store.similarity_search_with_score(user_prompt, k=1)

    if not results:
        print("Component not found")
        return

    document, score = results[0]

    if score < threshold:
        print(f"Match rejected! Score {score:.4f} is below threshold {threshold}.")
        print(f"Best guess was: {document.metadata.get('name')} (irrelevant)")
        return

    meta = document.metadata

    print(f"Found: {meta['name']}, ID: {meta['id']}, type: {meta['labels']}, score:{score}")

    def print_elements(title, element_list):
        print(f"\n--{title}--")
        if not element_list:
            print("None found")
            return
        
        for i, item in enumerate(element_list, 1):
            raw_code = item.get('code', '')
            
            # THE FIX: Replace \n, \r, and \t with a single space using Regex
            clean_code = re.sub(r'\s+', ' ', raw_code).strip()
            
            # Truncate for display (first 80 chars)
            display_code = clean_code[:100] + "..." if len(clean_code) > 100 else clean_code
            print(f"{i}, ID: {item.get('id')}, code: {display_code}")

    print_elements("links", meta.get('links', []))
    print_elements("images", meta.get('images', []))
    print_elements("inputs", meta.get('inputs', []))
    print_elements("buttons", meta.get('buttons', []))
    print_elements("routes", meta.get('routes', []))

def retrieve_components2(user_prompt, vector_store):
    seen_ids = set()
    unique_components = []

    print(f"--- Searching for {len(prompt_array)} items: {prompt_array} ---")

    # 2. Iterate through each extracted phrase (e.g., "react logo", "react link")
    for search_term in prompt_array:
        print(f"Searching for: '{search_term}'")
        
        # Search specifically for this term
        results = vector_store.similarity_search_with_score(search_term, k=3) # Lower k (e.g., 3) per term to keep it focused

        for document, score in results:
            # 3. Apply threshold
            if score < 0.70: 
                continue
            
            # 4. Deduplication logic: Check if we've already added this node ID
            node_id = document.metadata.get('id')
            if node_id in seen_ids:
                continue
            
            seen_ids.add(node_id)
            
            meta = document.metadata
            item_str = f"Name: {meta.get('name', 'Unnamed')} | ID: {node_id} | Code: {meta.get('code')} | Score: {score:.4f}"
            unique_components.append(item_str)

    # 5. Final Output
    if not unique_components:
        print("No components found matching the criteria.")
        return

    final_response = "\n".join(unique_components)
    print("--- Final Aggregated Results ---")
    print(final_response)

def parse_repo_url(url):
    """
    Helper functionn to extract "owner/repo" from URL
    """
    match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return None

if __name__ == "__main__":
    RETRIEVAL_QUERY_GENERAL = """
    RETURN 
        node.CODE AS text,
        score,
        {
            id: elementId(node),
            name:coalesce(node.NAME, node.FULL_NAME, 'Unnamed'),
            labels: labels(node),
            code: node.CODE
        } AS metadata
"""

    neo4j_url = "bolt://localhost:7687"
    neo4j_password = "password"
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
    
    # connects to the DB.
    store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=neo4j_url,
        password=neo4j_password,
        index_name="general_components",
        retrieval_query=RETRIEVAL_QUERY_GENERAL
    )

    llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0 
    )

    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a web QA tester. Extract the UI components and actions from the prompt, and put them as a list. For example, Prompt: Check whether the home button has the home logo, and directs to the shop link, and whether the cat image is present. Response you should give: home button, home logo, shop link, cat image. Don't forget the quotation mark for each phrase in the list"),
    ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    user_prompt = "check whether the react logo has the react link, and clicking the button will increment the number by 1"
    response = chain.invoke({"question": user_prompt})
    prompt_array = [item.strip() for item in response.split(',')]
    print(prompt_array)

    retrieve_components2(prompt_array, store) 
    # TODO: add UI elemnts extractor llm to the pipeline, and update code of retriever, then test.
    # TODO: pull/copy paste boss's newest code 