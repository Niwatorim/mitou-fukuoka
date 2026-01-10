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

RETRIEVAL_QUERY_E2E = """
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

def check_embeddings(uri="bolt://localhost:7687", auth=("neo4j", "password")) -> bool:
    """ Check if embeddings are there or not"""
    driver = GraphDatabase.driver(uri, auth=auth)
    
    try:
        with driver.session() as session:
            # Step 1: Check if the Index exists
            index_exists_query = """
            SHOW INDEXES YIELD name, type
            WHERE name = 'testable_components' AND type = 'VECTOR'
            RETURN count(*) > 0 AS exists
            """
            index_result = session.run(index_exists_query).single()
            
            if not index_result or not index_result["exists"]:
                print("DEBUG: Vector index 'testable_components' missing.")
                return False

            # Step 2: Check if nodes actually have embedding data
            data_exists_query = """
            MATCH (n:TestableComponent)
            WHERE n.embedding IS NOT NULL 
            RETURN count(n) > 0 AS has_data LIMIT 1
            """
            data_result = session.run(data_exists_query).single()
            
            if not data_result or not data_result["has_data"]:
                print("DEBUG: Index exists, but no nodes have 'embedding' property.")
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
        print("Creating TestableComponent label to certain nodes..")
        query = """
        MATCH (n) WHERE labels(n) IN [['TEMPLATE_DOM'], ['METHOD'], ['CALL'], ['IDENTIFIER']]
        SET n:TestableComponent
        """

        with self.driver.session() as session:
            session.run(query)
            print("Verified TestableComponent labels")
        
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
            index_name="testable_components",
            node_label="TestableComponent",
            text_node_properties=["NAME", "CODE"],
            embedding_node_property="embedding",
            retrieval_query=RETRIEVAL_QUERY_E2E
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
                index_name="testable_components",
                retrieval_query=RETRIEVAL_QUERY_E2E 
            )
        last_message = state["messages"][-1]
        user_query = last_message.content

        print(f"QUerying for {user_query}..")

        results = store.similarity_search_with_score(user_query, k=1)

        if not results:
            print("Component not found")
            return

        document, score = results[0]

        if score < 0.80:
            print(f"Match rejected! Score {score:.4f} is below threshold {0.80}.")
            print(f"Best guess was: {document.metadata.get('name')} (irrelevant)")
            return

        meta = document.metadata

        output = []
        output.append(f"Found: {meta.get('name')} (ID: {meta.get('id')}) Score: {score:.4f}")

        def format_section(title, items):
            section = [f"\n--{title}--"]
            if not items:
                section.append("None found")
            else:
                for i, item in enumerate(items, 1):
                    raw_code = item.get('code', '')
                    clean_code = re.sub(r'\s+', ' ', raw_code).strip()
                    display_code = clean_code[:100] + "..." if len(clean_code) > 100 else clean_code
                    section.append(f"{i}. ID: {item.get('id')} | Code: {display_code}")
            return "\n".join(section)

        output.append(format_section("LINKS", meta.get('links', [])))
        output.append(format_section("IMAGES", meta.get('images', [])))
        output.append(format_section("INPUTS", meta.get('inputs', [])))
        output.append(format_section("BUTTONS", meta.get('buttons', [])))
        output.append(format_section("ROUTES", meta.get('routes', [])))

        final_response = "\n".join(output)
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
                    print("Assistant:", value["messages"][-1][1])
    
    asyncio.run(run())