from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
import os,sys,traceback
from dotenv import load_dotenv
load_dotenv()
import asyncio

#-- graphRAG imports
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from langchain_neo4j import Neo4jVector
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp_server import MCPGeminiAgent

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
            # Step 1: Check if the Index exists
            index_exists_query = """
            SHOW INDEXES YIELD name, type
            WHERE name = 'general_components' AND type = 'VECTOR'
            RETURN count(*) > 0 AS exists
            """
            index_result = session.run(index_exists_query).single()
            
            if not index_result or not index_result["exists"]:
                print("DEBUG: Vector index 'general_components' missing.")
                return False

            # Step 2: Check if nodes actually have embedding data
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
    vector_store: any

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

class MCPNode:
    def __init__(self):
        self.messages=[]
        self.llm = MCPGeminiAgent()
        self.e2e = """
        You are a graph-based testing expert.

        IMPORTANT RULES:
        - You MUST use tools to inspect the graph before answering.
        - Do NOT answer from memory.
        - If information is missing, explore the graph using tools.
        - Only produce a final answer AFTER tool usage.

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


graph = StateGraph(State)

graph.add_node("Label_setup", LabelSetupNode("bolt://localhost:7687", ("neo4j", "password")))
graph.add_node("Vector_search",VectorSearchNode())
# graph.add_node("MCP", MCPNode())
graph.add_node("Embedding",EmbeddingNode())

graph.add_edge("Label_setup", "Embedding")
# graph.add_edge("Vector_search","MCP")
graph.add_edge("Embedding","Vector_search")

# graph.set_finish_point("MCP")
graph.set_finish_point("Vector_search")
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