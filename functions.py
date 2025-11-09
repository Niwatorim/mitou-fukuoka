from langchain_text_splitters import RecursiveCharacterTextSplitter
from tree_sitter import Language, Parser, Query, QueryCursor, Node
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from browser_use import Agent, ChatGoogle,Browser
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
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

# Load .env from the current directory where main.py is run
load_dotenv()

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
    chroma_client=chromadb.Client(path=os.path.join(current_dir, "Code_database"))
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
    gemini_embeddings= [e.values for e in result.embeddings]

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
        contents=chunks,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=3072
        ),
    )

    gemini_embeddings = [e.values for e in result.embeddings]

    collection = chroma_client.get_or_create_collection(name="ast")

    collection.add(
        embeddings=gemini_embeddings,
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
                                google_api_key=gemini_API,
                                model_kwargs={
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

    # Helper function to format documents into a single string
    def format_docs(docs):
        """Format document list into a single context string"""
        return "\n\n".join(doc.page_content for doc in docs)

    CONSOLE.print("[bold yellow] Making message [/bold yellow]")
    # Create modern LCEL chain using StrOutputParser
    document_chain = prompt | llm | StrOutputParser()
    
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
        query_embedding = [e.values for e in result.embeddings]

        results = collection.query( #queries the thing
            query_embeddings=query_embedding, # Use query_embeddings instead of query_texts
            n_results=2
        )
        
        docs=[]
        for i in range(len(results["ids"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            )
            docs.append(doc)
        CONSOLE.print("[green] making IDs [/green]")

        CONSOLE.print("[yellow]invoke message [/yellow]")
        # Invoke chain with formatted context
        response = document_chain.invoke({
            "input": query,
            "context": format_docs(docs)  # Format docs into string context
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

        
async def test_browser_use(limit=None,headless:bool = False, test_path:str = None)->list[dict]:
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

async def results_writer(results: list[dict[str:str|bool]])->None:
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
    nodes={} #dict to remove duplicates
    edges=[]
    node_types={}
    colors = ["#FFC0CB", "#ADD8E6", "#90EE90", "#FFD700", "#F08080", "#B0E0E6", "#DDA0DD"]
    def process(record):
        n=record["n"]
        m=record["m"]
        r=record["r"]

        def colors_assign(node_data):
            node_type=node_data.get("type")
            if node_type and node_type not in node_types:
                color=colors[len(node_types) % len(colors)]
                node_types[node_type]=color

        colors_assign(r[0])
        colors_assign(r[2])

        #need to created ids
        source_id=json.dumps(r[0],sort_keys=True)
        target_id=json.dumps(r[2],sort_keys=True)
        relation_type=r[1]

        if source_id not in nodes:
            nodes[source_id]={
                "id":source_id, 
                "properties":r[0]
            }
        if target_id not in nodes:
            nodes[target_id]={
                "id":target_id,
                "properties":r[2]
            }
        edges.append({
            "source":source_id,
            "target":target_id,
            "type":relation_type
        })
        

    result = graph.query("""
            MATCH (n)-[r]->(m)
            RETURN n, r, m
            LIMIT 25
        """)

    
    for record in result:
        process(record)
    
    final_nodes=list(nodes.values())

    return final_nodes,edges,node_types

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
