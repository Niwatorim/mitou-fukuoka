"""
Testing for graphRag and how it works and if possible
"""
from langchain_neo4j import Neo4jGraph
from langchain.tools import tool
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os


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
