"""
Testing for graphRag and how it works and if possible
"""
from langchain_neo4j import Neo4jGraph,Neo4jVector
from langchain.tools import tool
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
import time
import requests

#constants -----------
load_dotenv()
graph=Neo4jGraph()
gemini_api_key=os.getenv("GEMINI_API_KEY")
prompt="Please tell me how many 'function' nodes there are "
schema = graph.schema
CONSOLE=Console()
system_prompt = f"""
You are a helpful assistant that answers questions by querying a Neo4j graph.
Be concise and accurate, speak only what is required and nothing more.
You must only use Cypher queries. You must not use any other query language.
You can only use the node labels and relationship types present in the schema.

Here is the graph schema:
{schema}
"""
CONSOLE.print(
        Panel(schema, title="[bold] Schema [/bold]", style="yellow", border_style="yellow")
    )
#tool to be called --------
@tool
def query_graph(query:str)-> str:
    """
    Cypher query from the neo4j graph
    """
    CONSOLE.print(f"\n[green][bold]Running following query:\t[/bold] {query} [/green]")
    return graph.query(query) #simple neo4j call, completely up to llm


#if tool fails, create response ------
@wrap_tool_call
def handle_errors(request,handler)->ToolMessage:
    try:
        return handler(request)
    except Exception as e:
        #return error message to model
        CONSOLE.print(f"[magenta] Error: {str(e)} [/magenta] ")
        return ToolMessage(
            content=f"Tool error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )


#create the agent ----------
def create_agents():
    llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                google_api_key=gemini_api_key)

    agent= create_agent(
                llm,
                tools=[query_graph],
                system_prompt=system_prompt,
                middleware=[handle_errors]
                )

    result=agent.invoke(
        {"messages":
            [{
                "role":"user",
                "content":prompt
            }]}
    )
    print(result["messages"][-1].content) #prints final response

def notify():
    message="embeddings done"
    requests.post(f"https://ntfy.sh/embeddings",
              data=message.encode('utf-8'),
              headers={
                  "Title": "Embeddings done",
                  "Priority": "urgent",  # or "high", "default", "low", "min"
                  "Tags": "alarm_clock",
              })

#--------- Embedding --------
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=gemini_api_key
# )

embeddings = OllamaEmbeddings(
    model="llama3",
    validate_model_on_init=True,
    timeout=30
)

def vector_search():
    """
    Creates vector store then searches it
    """
    #---- embed ----
    def create_vector_store():
        """
        Create embeddings for all nodes in the Neo4j graph and store them.
        """

        # Connect to graph
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

        # Pull all nodes and their text-like fields
        nodes = graph.query("""
        MATCH (n)
        RETURN labels(n)[0] AS label, n AS node, elementId(n) as id
        """)
        
        documents=[]
        for value in nodes:
            #value in form:
            # label | node
            # "Function"│(:Function {name: "App",type: "function_declaration",params: "()"}) 


            node=value["node"]
            label=value["label"]
            id=value["id"]

            #all the text together
            text = " ".join(str(v) for v in node.values() if isinstance(v,(str,list)))

            doc = Document(page_content=text, metadata={
                "label":label,
                "id":id
            })
            documents.append(doc)

        batch=90
        tot_docs=len(documents)
        vector_store=None
        CONSOLE.print(f"[magenta]Embedding {tot_docs} in batches of {batch}[/magenta]")
        for i in range(0,tot_docs,batch):
            batch_content = documents[i : i + batch]
            CONSOLE.print(f"Processing batch {i//batch + 1}/{(tot_docs + batch - 1)//batch}...")
            if i == 0:
                vector_store = Neo4jVector.from_documents(
                    documents=batch_content,
                    embedding=embeddings,
                    url=os.getenv("NEO4J_URI"),
                    username=os.getenv("NEO4J_USERNAME"),
                    password=os.getenv("NEO4J_PASSWORD"),
                    index_name="component_embeddings",
                    node_label="Component",
                    text_node_property="text",
                    embedding_node_property="embedding"
                )
            else:
                vector_store.add_documents(batch_content)
            if i + batch < tot_docs:
                CONSOLE.print("[yellow] waiting a few seconds to respect API Rate limit.. [/yellow]")
                time.sleep(5)
        CONSOLE.print("[green] Embeddings created and stored in Neo4j.[/green]")

    def retrieve_docs():
        vector_store = Neo4jVector.from_existing_graph(
            embeddings,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name="component_embeddings",
            node_label="Component",
            text_node_properties=["name","properties","type","value","source","imports"],
            embedding_node_property="embedding"
        )

        def semantic_search(query: str, k: int = 3):
            results = vector_store.similarity_search(query,k=k)
            CONSOLE.print(f"[cyan]Found {len(results)}  similar components")
            for k,i in enumerate(results):
                print(k,": ",i)
            return results
    
        results=semantic_search("function nodes")
        #call semantic search
        Console().print(
            Panel(str(i.metadata+" "+i.page_content for i in results),
                  title="[bold] results of semantics [/bold]",
                  style="yellow"
                  ))

    create_vector_store()
    notify()
    retrieve_docs()

if __name__ == "__main__":
    # create_agent()
    vector_search()
    # notify()




