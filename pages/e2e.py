import streamlit as st
import json
import time
import asyncio
import os,sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()


#TODO: search for node and possible existing paths for the e2e
#TODO: generate instructions if yes

CONSOLE = Console()

def clean_schema(schema: Any) -> Any:
    """Recursively clean the schema to only include keys Gemini supports."""
    if not isinstance(schema, dict):
        return schema
    
    # Gemini's strictly allowed keys for Tool schemas
    allowed_keys = {
        "type", "properties", "required", "description", 
        "items", "enum", "format", "nullable"
    }
    
    cleaned = {}
    for k, v in schema.items():
        if k in allowed_keys:
            if k == "properties" and isinstance(v, dict):
                # Clean each property's definition recursively
                cleaned[k] = {prop_name: clean_schema(prop_val) for prop_name, prop_val in v.items()}
            elif k == "items" and isinstance(v, dict):
                # Clean array item definitions
                cleaned[k] = clean_schema(v)
            else:
                cleaned[k] = v
                
    return cleaned

class MCPGeminiAgent: #get the gemini agent ready
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-2.0-flash"
        self.tools= None
        self.server_params= None
        self.server_name= None

    def server_choose(self):
        # Get the path to mcp.json relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_path = os.path.join(script_dir, "mcp.json")
        
        with open(mcp_path, "r") as f:
            mcp_config = json.load(f)
        servers = mcp_config["mcpServers"]
        server_names=list(servers.keys())
        #assuming just using neo4j
        self.server_name = server_names[0]
        server_cfg = servers[self.server_name]
        command = server_cfg["command"]
        args = server_cfg.get("args",[])
        env = server_cfg.get("env",None)
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
    
    async def connect(self):
        self.server_choose()
        self.stdio_transport = await self.exit_stack.enter_async_context(stdio_client(self.server_params))
        self.stdio, self.write = self.stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio,self.write))
        await self.session.initialize()
        CONSOLE.print(f"[blue] Connected to {self.server_name} [/blue]")

    async def agent_loop(self, prompt: str) -> Any:
        contents = [types.Content(
            role="user",
            parts=[types.Part(text=prompt)]
        )]
        
        mcp_tools = await self.session.list_tools()
        tools = types.Tool(function_declarations=[
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": clean_schema(getattr(tool, "inputSchema", {}))
            }
            for tool in mcp_tools.tools
        ])
        self.tools = tools

        config = types.GenerateContentConfig(
            system_instruction="""
            You are a graph-based testing expert. Your goal is to help the user understand and test their application by querying the Neo4j graph database.
            
            1. If you don't know the structure of the graph or cannot answer the user's question, USE THE TOOLS to explore the nodes and relationships.
            2. For end-to-end testing requests, generate steps to go from node to node in this format:
               Path_exists: True/False
               test_steps:
                   - step: 1
                     action: navigate
                     instruction: ...
                     target: ...
                     expected: ...
            
            Always prioritize using tools when factual information about the graph is needed.""",
            temperature=0,
            tools=[tools]
        )

        # Initial call
        CONSOLE.print("[bold magenta] Initial call [/bold magenta]")
        print(f"[yellow] Requesting initial response from Gemini with tools: {[t.name for t in mcp_tools.tools]} [yellow]")
        response = await self.genai_client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        contents.append(response.candidates[0].content)
        time.sleep(3)
        turn_count = 0
        max_tool_turns = 10
        
        while response.function_calls and turn_count < max_tool_turns:
            turn_count += 1
            tool_response_parts: List[types.Part] = []
            
            for fc_part in response.function_calls:
                tool_name = fc_part.name
                args = fc_part.args or {}
                CONSOLE.print(
                    Panel(f"Invoking MCP tool {tool_name} with args: {args}",title="tool",expand=True))
                
                try:
                    tool_result = await self.session.call_tool(tool_name, args)
                    print(f"Tool {tool_name} done")
                    # Assuming tool_result.content[0].text exists based on existing code
                    tool_content = tool_result.content[0].text if tool_result.content else "Success"
                    tool_response = {"result": tool_content}
                except Exception as e:
                    print(f"Tool {tool_name} failed: {e}")
                    tool_response = {
                        "error": f"Tool execution failed: {type(e).__name__}:{e}"
                    }
                
                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response=tool_response
                    )
                )

            # Append all tool responses at once
            contents.append(types.Content(
                role="user",
                parts=tool_response_parts
            ))
            
            CONSOLE.print(
                    Panel(f"[bold yellow] Requesting updated response from Gemini (Turn {turn_count}) [/bold yellow] ",title="turn",expand=True))
            response = await self.genai_client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            contents.append(response.candidates[0].content)
            time.sleep(2)
        if turn_count >= max_tool_turns and response.function_calls:
            print(f"Max tool count reached so stopping")
            
        return response

    async def chat(self):
        print(f"MCP-assistant connected alh")
        try:
            query = "please tell me how many function nodes there are"
            if query.lower() == "quit":
                print("Session End")
            res = await self.agent_loop(query)
            if res is not None:
                CONSOLE.print(f"[bold green]{res.text} [/bold green]")
            else:
                print(res)
        except Exception as e:
            print(f"error occured: {e}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    agent = MCPGeminiAgent()
    try:
        await agent.connect()
        await agent.chat()
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    import traceback
    try:
        asyncio.run(main())
    except Exception:
        CONSOLE.print("[bold red]Fatal error during execution:[/bold red]")
        traceback.print_exc()







# async def get_tools(session: ClientSession) -> List[Dict[str, Any]]:
#     await session.initialize()
#     tools_result = await session.list_tools()
    
#     # Format for Gemini tool definition
#     return [
#         {
#             "function_declarations": [
#                 {
#                     "name": tool.name,
#                     "description": tool.description,
#                     "parameters": tool.inputSchema,
#                 }
#             ]
#         }
#         for tool in tools_result.tools
#     ]

# async def execute_tool_calls(response_parts, session, tools_sse):
#     tool_results = []
#     messages_to_add = []
    
#     for part in response_parts:
#         if part.function_call:
#             tool_call = part.function_call
#             tool_name = tool_call.name
#             args = tool_call.args
            
#             st.write(f"Executing tool: {tool_name}")
            
#             try:
#                 # Call the MCP tool
#                 result = await session.call_tool(tool_name, arguments=args)
                
#                 # Extract content from result
#                 if result.content:
#                     content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
#                 else:
#                     content = "No content returned from tool."
                
#                 st.write(f"Tool {tool_name} result: {content[:200]}...")
                
#                 # Format for Gemini tool response
#                 messages_to_add.append(
#                     types.Part.from_function_response(
#                         name=tool_name,
#                         response={"result": content}
#                     )
#                 )
                
#                 tool_results.append({
#                     "tool": tool_name,
#                     "result": content
#                 })
                
#             except Exception as e:
#                 error_msg = f"Error executing tool {tool_name}: {str(e)}"
#                 st.error(error_msg)
#                 messages_to_add.append(
#                     types.Part.from_function_response(
#                         name=tool_name,
#                         response={"error": error_msg}
#                     )
#                 )
#                 tool_results.append({
#                     "tool": tool_name,
#                     "result": error_msg
#                 })
    
#     return tool_results, messages_to_add

# async def process_with_gemini(user_request, session):
#     """Process the user request using Gemini with tool calling capabilities"""
#     try:
#         # Get and format tools
#         mcp_tools = await get_tools(session)
        
#         client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
#         # Flatten the list of function declarations for Gemini
#         gemini_tools = []
#         for tool_group in mcp_tools:
#             gemini_tools.extend(tool_group["function_declarations"])
            
#         config = types.GenerateContentConfig(
#             tools=[types.Tool(function_declarations=gemini_tools)],
#             system_instruction="You are a Neo4j expert. You have access to tools that can execute Cypher queries and read the database schema. Analyze the user's request and use these tools to provide accurate information from the graph database. If you need to explore the schema first, do so."
#         )

#         chat = client.chats.create(model="gemini-2.0-flash", config=config)
        
#         max_iterations = 10
#         iteration = 0
        
#         current_request = user_request
        
#         while iteration < max_iterations:
#             iteration += 1
#             st.write(f"Gemini Iteration {iteration}...")
            
#             response = chat.send_message(current_request)
            
#             # Check for function calls
#             has_function_calls = any(part.function_call for part in response.candidates[0].content.parts)
            
#             if not has_function_calls:
#                 return response.text
                
#             # Execute tool calls
#             tool_results, tool_responses = await execute_tool_calls(
#                 response.candidates[0].content.parts, session, mcp_tools
#             )
            
#             # Send the tool results back to the model
#             current_request = tool_responses
            
#         return "Maximum iterations reached. The task may be complex and needs more steps"
        
#     except Exception as e:
#         st.error(f"Error in Gemini processing: {e}")
#         import traceback
#         st.code(traceback.format_exc())
#         return f"Error: {str(e)}"

# async def run_query(user_request):
#     status = st.status("Connecting to Neo4j MCP Server...")
    
#     # Configure the Neo4j MCP server parameters
#     server_params = StdioServerParameters(
#         command="uvx",
#         args=["mcp-neo4j-cypher@0.5.2", "--transport", "stdio"],
#         env={
#             "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
#             "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", "neo4j"),
#             "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password"),
#             # "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE", "neo4j"),
#             # "PATH": os.getenv("PATH") # Ensure uvx is in PATH
#         }
#     )
#     try:
#         async with stdio_client(server_params) as (read_stream, write_stream):
#             async with ClientSession(read_stream, write_stream) as session:
#                 status.update(label="Connected to Neo4j MCP Server", state="running")
#                 final_result = await process_with_gemini(user_request, session)
#                 status.update(label="Query Complete", state="complete")
#                 st.markdown("### Final Result")
#                 st.write(final_result)

#     except Exception as e:
#         status.update(label=f"An error occurred: {e}", state="error")
#         st.error(str(e))

# # Streamlit UI
# user_input = st.text_input("What would you like to know about the graph?", "List all function labels and their count")
# prompt=f"""
# You are an end to end test expert, and must take the following instruction and divide it into steps after looking at the graph:
# instructions: {user_input}

# First: Observe if there is a path that connects every node of the function that is being requested.
# Second: Generate steps to instruct how to go from node to node on the graph relative to the website in the following format:
# Path_exists: True/False
# test_steps:
#     - step: 1
#         action: navigate
#         instruction: Open the application
#         target: "http://localhost:5173/"
#         expected: Page loads successfully
#     - step: 2
#         action: click
#         instruction: Click the submit button
#         selector: "#submit-btn"
#         expected: Form submits successfully

# Each expectation must be from a connection of the node
# """

# if st.button("Query Database"):
#     if user_input:
#         asyncio.run(run_query(prompt))
#     else:
#         st.warning("Please enter a question.")
