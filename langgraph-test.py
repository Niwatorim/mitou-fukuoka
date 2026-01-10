from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,END
from langgraph.graph.message import add_messages
import os,sys,traceback
from dotenv import load_dotenv
load_dotenv()
import asyncio

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mcp_server import MCPGeminiAgent


class State(TypedDict): #create message history
    messages: Annotated[list,add_messages]

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

graph.add_node("MCP",MCPNode())
graph.set_entry_point("MCP")
graph.set_finish_point("MCP")

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