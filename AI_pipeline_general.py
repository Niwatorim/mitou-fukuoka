from typing import Annotated,Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
import os,sys,traceback
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector
import re
from rich.console import Console
from rich.panel import Panel
import datetime
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp_server import MCPPlaywright,MCPNeo4J,generator

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

def check_embeddings(uri="bolt://localhost:7687", auth=("neo4j", "password")) -> bool:
    """ Check if embeddings are there or not"""
    driver = GraphDatabase.driver(uri, auth=auth)
    
    try:
        with driver.session() as session:
            index_exists_query = """
            SHOW INDEXES YIELD name, type
            WHERE name = 'general_components' AND type = 'VECTOR'
            RETURN count(*) > 0 AS exists
            """
            index_result = session.run(index_exists_query).single()
            
            if not index_result or not index_result["exists"]:
                print("DEBUG: Vector index 'general_components' missing.")
                return False

            data_exists_query = """
            MATCH (n:GeneralComponent)
            WHERE n.embedding_general IS NOT NULL 
            RETURN count(n) > 0 AS has_data LIMIT 1
            """
            data_result = session.run(data_exists_query).single()
            
            if not data_result or not data_result["has_data"]:
                print("DEBUG: Index exists, but no nodes have 'embedding_general' property.")
                return False
                
            print("DEBUG: Embeddings verified (Index + Data found).")
            return True

    except Exception as e:
        print(f"DEBUG: Error checking embeddings: {e}")
        return False
    finally:
        driver.close()

class State(TypedDict): #create message history
    messages: Annotated[list,add_messages]
    vector_store: Any
    tool_history: list[dict]

class LabelSetupNode:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def __call__(self, state:State):
        print("Creating GeneralComponent label to certain nodes..")
        query = """
        MATCH (n) 
        WHERE any(l IN labels(n) WHERE l IN [
            'TEMPLATE_DOM', 'METHOD', 'CALL', 'IDENTIFIER', 
            'LITERAL', 'TYPE_DECL', 'METHOD_PARAMETER_IN', 'METHOD_PARAMETER_OUT'
        ])
        SET n:GeneralComponent
        """
        # If need to be more general, just add the labels here 

        with self.driver.session() as session:
            session.run(query)
            print("Verified GeneralComponent labels")
        
        return state

class EmbeddingNode: #embeds the entire graph if the thing dont exist
    def __call__(self,state:State):
        """
        Takes the entire graph and embed it
        """
        neo4j_url = "bolt://localhost:7687"
        neo4j_password = "password"

        embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"
        )

        # creating the embeddings
        vector_store = Neo4jVector.from_existing_graph(
            embedding=embeddings,
            url=neo4j_url,
            password=neo4j_password,
            index_name="general_components",
            node_label="GeneralComponent",
            text_node_properties=["NAME", "CODE"],
            embedding_node_property="embedding_general",
            retrieval_query=RETRIEVAL_QUERY_GENERAL
        )
        
        return {"vector_store": vector_store}

class VectorSearchNode:
    def __call__(self,state:State):
        print("Doing vector search..")

        store = state.get('vector_store')

        if not store:
            print("    (Re-connecting to existing Neo4j index...)")
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
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content'):
            # if an object
            user_query = last_message.content
        else:
            # a tuple ("user", "query")
            user_query = last_message[1]

        print(f"Querying for {user_query}..")

        results = store.similarity_search_with_score(user_query, k=20) # change k depending on how many nodes you want to return

        if not results:
            text = "Component not found"
            print(text)
            return {"messages": [("assistant", text)]}

        components = []
        for document, score in results:
            if score < 0.70: # can change threshold if want more general or more strict
                continue
            meta = document.metadata
            item_str = f"Name: {meta.get('name', 'Unnamed')} | ID: {meta.get('id')} | Code: {meta.get('code')} | Score: {score:.4f}"
            components.append(item_str)

        final_response = "\n".join(components)
        if not components:
            final_response = "No components found with high enough confidence."
        return {"messages": [("assistant",final_response)]}

class MCPGraph:
    def __init__(self):
        self.messages=[]
        self.llm = MCPNeo4J()
        self.e2e = """
        You are a graph-based testing expert.

        IMPORTANT RULES:
        - You MUST use tools to inspect the graph before answering.
        - Do NOT answer from memory.
        - If information is missing, explore the graph using tools.
        - Only produce a final answer AFTER tool usage.

        MENTION THE APP WILL BE OPENED ON http://localhost:5173/

        Output format:
        Path_exists: True/False
        test_steps:
        - step: 1
        action: navigate
        instruction: ...
        target: ...
        expected: ...
        """

    async def __call__(self, state:State):
        messages = state["messages"]
        agent = self.llm
        try:
            await agent.connect()
            data = await agent.chat(messages,self.e2e)
            content = data.text if data and hasattr(data, "text") else str(data)
            return {"messages":[("assistant",content)]}
        
        except Exception as e:
            print(f"Fatal error during execution: {e}")
            traceback.print_exc()
        
        finally:
            await self.llm.cleanup()

class MCPTester:
    def __init__(self):
        self.messages=[]
        self.llm = MCPPlaywright()

        self.e2e = """
        You are a useful agent who can use the browser using your tools in order to carry out instructions.
        
        RULES:
        1. Read the instructions provided in the input.
        2. Execute the steps sequentially using the browser tools.
        3. Do NOT answer from memory; use the tools.
        4. If the instruction is to click, use `browser_click`.
        5. CRITICAL: Once you have performed the requested actions and verified the result visually in the DOM, call `browser_close`.
        6. AFTER calling `browser_close`, DO NOT ATTEMPT TO RE-OPEN THE BROWSER.
        7. DO NOT try to verify the test again using `browser_run_code` or JavaScript injection. One pass is enough.
        8. Once the browser is closed, simply output the final report.

        Give a response in the following format:
        Test success: True/False
        Python code: (generate the equivalent python playwright code for the steps you took)
        """

    async def __call__(self, state:State):
        messages = state["messages"]
        agent = self.llm
        try:
            await agent.connect()
            data, tool_history = await agent.chat(messages,self.e2e)
            content = data.text if data and hasattr(data, "text") else str(data)
            console_cont = Console()
            console_cont.print("[magenta]-----------------------------[/magenta]")
            console_cont.print(Panel(f"[bold green] {content} [/bold green]",title="Final response"))
            #save to file
            path=location(results=True,test_type="E2E")
            with open(path,"w") as f:
                f.write(content)


            return {"messages":[("assistant",content)],
                    "tool_history":tool_history}
        
        
        except Exception as e:
            print(f"Fatal error during execution: {e}")
            traceback.print_exc()
        
        finally:
            await self.llm.cleanup()

def user_check(state:State):
    logger=Console()
    logger.print("[bold green] Continue with the following instructions? [/bold green]")
    last_msg = state["messages"][-1]
    plan = last_msg[1] if isinstance(last_msg,tuple) else last_msg.content

    logger.print(
        Panel(f"[yellow]{plan}[/yellow]",title="plane")
    )
    user_ans=input("Continue to 'Test'? (y/n)").strip().lower()

    if user_ans == "y":
        return "Tester"
    return END

def generate_user_check(state:State):
    logger=Console()
    logger.print("[bold green] Generate code? [/bold green]")
    tools = state["tool_history"]
    logger.print(Panel(f"[magenta]{tools}[/magenta]"))
    user_ans=input("Continue to 'Test'? (y/n)").strip().lower()
    if user_ans == "y":
        return "generate"
    return END

def clean_code_block(text: str) -> str: #gets only the acc code inside the block
    pattern = r"```(?:python)?\n(.*?)```"
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return text.strip()

def location(results:bool,test_type:str)->str:
    if results: #if its a test file
        path="./results"
    else: #if its a code block
        path="./tests/codeblock"

    if test_type == "E2E":
        new_path=os.path.join(path,test_type)
        timestamp = datetime.datetime.now()
        filename = timestamp.strftime("E2E_%Y/%m/%d_%H:%M:%S.py")
        final_path = os.path.join(new_path,filename)

    return final_path


async def generate_code(state: State): #generates the actual code from tool history
    tools = state.get("tool_history", [])
    tools_str = json.dumps(tools, indent=2)
    response_text = await generator(tools_str)
    response = clean_code_block(response_text)
    path = location(results=False,test_type="E2E")
    with open(path,"w") as f:
        f.write(response)
    return {"messages": [("assistant", response)]}

graph = StateGraph(State)
graph.add_node("Label_setup", LabelSetupNode("bolt://localhost:7687", ("neo4j", "password")))
graph.add_node("Vector_search",VectorSearchNode())
graph.add_node("MCPGraph", MCPGraph())
graph.add_node("Embedding",EmbeddingNode())
graph.add_node("Tester",MCPTester())
graph.add_edge("Label_setup", "Embedding")
graph.add_edge("Vector_search","MCPGraph")
graph.add_edge("Embedding","Vector_search")
graph.add_node("generate",generate_code)
graph.add_conditional_edges(
    "MCPGraph",
    user_check,
    {
        "Tester": "Tester",
        END:END
    }
)
graph.add_conditional_edges(
    "Tester",
    generate_user_check,
    {
        "generate":"generate",
        END:END
    }
)
graph.set_finish_point("generate")

embeddings_exist=check_embeddings("bolt://localhost:7687", ("neo4j", "password"))
if embeddings_exist:
    graph.set_entry_point("Vector_search")
if not embeddings_exist:
    graph.set_entry_point("Label_setup")

graph_final = graph.compile()
try:
    png_data = graph_final.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("Graph saved to graph.png")

except Exception as e:
    print(f"Error generating graph: {e}")

if True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
    async def run():
        async for event in graph_final.astream(
            {"messages": [("user", user_input)]}
            ):
                for value in event.values():
                    if value is not None and "messages" in value:
                        last_message = value["messages"][-1] 
                        if hasattr(last_message, 'content'):
                            # if an object
                            response = last_message.content
                        else:
                            #  a tuple ("user", "query")
                            response = last_message[1]
                        print("Assistant:", response)
                    else:
                        print("No output") # to handle NoneType

    asyncio.run(run())