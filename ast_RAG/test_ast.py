import json
import subprocess
import yaml
import os
from browser_use import Agent, ChatGoogle
import asyncio

#gemini
from google import genai
from google.genai import types

#chroma db
import chromadb
#langchain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv("../.env")

from tree_sitter import Language,Parser,Query,QueryCursor,Node
import tree_sitter_javascript as tsj
from rich.console import Console
from rich.panel import Panel


#put ast into code_structure.json (needs to be called in main)
def ast_rag(file):
    parser_path= "/Users/niwatorimostiqo/Desktop/Coding/Mitou Fukuoka/parser_test.js"
    command = ["node",parser_path,file]
    values=subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )
    return values.stdout

#embed the file code_structor.json
def embed_ast():

    client=genai.Client()
    chroma_client=chromadb.PersistentClient(path="../Code_database")
    collection=chroma_client.get_or_create_collection(name="ast")
    
    loader=TextLoader("./code_structure.json")
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

CONSOLE= Console()
#helper function for cycle()
def access_code(instructions):
    client=genai.Client()
    chroma_client= chromadb.PersistentClient(path="../Code_database")
    collection = chroma_client.get_collection(name="ast")
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
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                 google_api_key="AIzaSyB46zy_IKF197pOSJZDBXy-1PjHsKg46_k",
                                 model_kwargs={
                                     "response_mime_type":"application/yaml"
                                 })
    prompt = ChatPromptTemplate.from_template("""
    You are a test automation expert. Generate test instructions in JSON format.

    Context: {context}
    Component: {input}

    Return instructions with this exact structure and nothing else:
    "component": "component_name",
    "url": "http://localhost:5173/",
    "test_steps": [
        "step": 1, "action": "navigate", "instruction": "Open the application", "target": "http://localhost:5173/",
        "step": 2, "action": "click", "instruction": "Click the submit button","text":"Submit", "selector": "#submit-btn", "expected": "Form submits successfully",
    ]
    

    Requirements:
    - Each instruction must be ONE clear action
    - Include specific selectors (id, class, data-testid, or text)
    - Use action types: navigate, click, type, verify, wait, select
""")

    CONSOLE.print("[bold yellow] Making message [/bold yellow]")
    document_chain = create_stuff_documents_chain(llm,prompt)

    CONSOLE.print("[yellow]invoke message [/yellow]")
    response = document_chain.invoke({
        "input": query,
        "context": docs
    })
    CONSOLE.print(
        Panel(
        response,title="response",expand=True))
    return (json.loads(response))

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
def cycle():
    
    os.makedirs("tests", exist_ok=True)
    with open("code_structure.json","r") as f:
        data=json.load(f)

    existing_files = set()
    for index,i in enumerate(data["components"]):
        if i["testableAttributes"]:
            instruction= f"please give instructions to test the component {i}"
            yaml_data=access_code(instruction)
            filename = unique_file(i['name'], existing_files)

            with open(filename,"w") as f:
                yaml.dump(yaml_data,f,default_flow_style=False, sort_keys=False)

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
    CONSOLE.print("[bold magenta] Failures: [/bold magenta]")
    for i in success_files:
        if i["success"]==False:
            print(i["name"])

FILE_NAME="../test-project/src/App.jsx"
#---------- Tree sitter -------
JSLANGUAGE = Language(tsj.language()) #creates language
FUNCTIONS= ["arrow_function","function_declaration","function"]
VARIABLES= ["array_pattern"]

def tree_sitter(file):
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

        
    """)

    contents={
        "variables":[],
        "functions":[],
        "components":[],
        "call_expression":[]
    }
    cursor = QueryCursor(query)
    values=cursor.captures(root) #capture from the node u start

    for node in values.get("Function",[]):
        contents["functions"].append(get_function(node))

    for node in values.get("element",[]):
        contents["components"].append(get_frontend(node))

    for node in values.get("var",[]):
        contents["variables"].append(get_variables(node))

    for node in values.get("call",[]):
        contents["call_expression"].append(get_call(node))
    return contents

def get_function(node,):
    function={
        "type": node.type,
        "params": "",
        "name": "",
        "body": "",
        "nested": []
    }
    
    query=Query(JSLANGUAGE,"""
        name: (identifier) @name
        parameters: (formal_parameters) @params
    """)
    cursor = QueryCursor(query)
    values=cursor.captures(node)

    if values.get("name"):
        function["name"] = values["name"][0].text.decode("utf8")
    if values.get("params"):
        function["params"] = values["params"][0].text.decode("utf8")

    function["body"] = node.text.decode("utf8")


    def find_nested_functions(n):
        nested = []
        for child in n.children:
            if child.type in FUNCTIONS:
                nested.append(get_function(child))
            else:
                # Recursively search deeper
                nested.extend(find_nested_functions(child))
        return nested
    
    function["nested"] = find_nested_functions(node)
    
    return function

def get_frontend(node):
    query=Query(JSLANGUAGE,"""
        (identifier)@name
        (jsx_attribute)@properties
    """)
    attribute={
        "name":"",
        "properties":[],
        "callbacks":[]
    }
    cursor = QueryCursor(query)
    values=cursor.captures(node)
    
    if values.get("name"):
        attribute["name"] = values["name"][0].text.decode()
    
    for property in values.get("properties",[]):
        attribute["properties"].append(property.text.decode())
        
        for child in property.children:
            if child.type == "jsx_expression":
                # FIX: Recursively search for functions inside expressions
                def find_functions_in_expr(n):
                    results = []
                    for c in n.children:
                        if c.type in FUNCTIONS:
                            results.append(get_function(c))
                        else:
                            results.extend(find_functions_in_expr(c))
                    return results
                
                attribute["callbacks"].extend(find_functions_in_expr(child))

    return attribute

def get_variables(node:Node):
    variable={
        "type":"",
        "names":[],
        "value":"",
        "value_type":""
    }
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
    call={
        "function_name":"",
        "arguments":[],
        "full_text":[]
    }
    
    for child in node.children:
        if child.type == "identifier":
            call["function_name"]=child.text.decode("utf8")
        if child.type == "arguments":
            call["arguments"].append(child.text.decode("utf8"))
    

    call["full_text"]=node.text.decode("utf8")
    return call

print(json.dumps(tree_sitter(FILE_NAME),indent=2))

"""

        (import_clause
            (identifier)
        ) @imports

        (import_specifier
            (identifier) @imports
        )

        (jsx_element
            (jsx_opening_element
                (identifier)@attributes
            )
        )
        
        (jsx_attribute
            (property_identifier
                )@properties
        )
    
  
  // Find all components/functions
  FOR each function_declaration or arrow_function at top level:
    CREATE Node in Neo4j: (Function {name, type, file, lineNumber})

    query:
    (function_declaration
	    name: (identifier) @func_name
        parameters: (formal_parameters) @params
    )


  // Find all JSX attributes with expressions
  FOR each jsx_attribute:
    GET the property_identifier (e.g., "onClick")
    GET the jsx_expression content
    
    (jsx_element
        (jsx_opening_element
            (identifier)@attr
            (jsx_attribute)@properties
        )  
    )    

    i = properties.
    query for i: (arrowfunction) @function
    match f:function {type: arrow, parent: attribute} to attribute -> callback function


    // Now dig into what's inside the expression
    IF jsx_expression contains an arrow_function:
      CREATE Node: (CallbackFunction {type: "arrow", parent: attribute})
      CREATE EDGE: attribute -> callbackFunction
      
      // Look inside the arrow function body
      FOR each call_expression inside this arrow function:
        GET the function_identifier being called (e.g., "setCount")
        CREATE Node: (FunctionCall {name: "setCount", context: "onClick"})
        CREATE EDGE: callbackFunction -> functionCall
        
        // If setCount is defined elsewhere, link to it
        IF setCount exists as a node:
          CREATE EDGE: functionCall -> setCount (type: "calls")
      
      // Look for nested arrow functions
      FOR each nested arrow_function:
        CREATE Node: (NestedArrowFunction {params, body})
        CREATE EDGE: callbackFunction -> nestedArrowFunction
        REPEAT the call_expression search inside this one too
    
    // Also capture any direct identifiers/variables referenced
    FOR each identifier in jsx_expression:
      IF identifier is a state variable or imported thing:
        CREATE EDGE: currentFunction -> identifier (type: "uses")

CREATE EDGES for imports/dependencies between files


"""

async def main():
    # data=json.loads(ast_rag("/Users/niwatorimostiqo/Desktop/Coding/Mitou Fukuoka/test-project/src/App.jsx"))
    # with open("code_structure.json","w") as f:
    #     json.dump(data,f,indent=4)
    # embed_ast()
    # cycle()
    # await test_browser_use()

    # with open("./tests/p.yaml","r") as f:
    #     data=yaml.safe_load(f)
    # task=str(yaml.dump(data["test_steps"], default_flow_style=False, sort_keys=False))
    # agent = Agent(
    #     task=task,
    #     llm=ChatGoogle(model="gemini-2.5-flash"),
    # )
    # history = await agent.run()
    # print(history.is_successful())
    # for i in history.action_history():
    #     print(i)
    # print(json.dumps(tree_sitter(FILE_NAME),indent=2))
    CONSOLE.print(Panel(
        """[bold green] 
            a.yaml 
            a[1].yaml
            img.yaml
            img[1].yaml
            [/bold green]
        """,title="success elements:"
    ))
    CONSOLE.print(Panel(
        """[bold magenta] 
            [/bold magenta]
        """,title="Failures"
    ))

if __name__ == "__main__":
    asyncio.run(main())