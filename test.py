import subprocess
from mcp import ClientSession
from mcp.client.sse import sse_client
import time
import asyncio
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import FunctionAgent,ToolCallResult,ToolCall
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from pydantic import create_model
from typing import List

def fix_schema(schema_part):
            if isinstance(schema_part, dict):
                for key, value in list(schema_part.items()):
                    if key == "additionalProperties" and value is True:
                        schema_part[key] = {}  # The fix!
                    else:
                        fix_schema(value)
            elif isinstance(schema_part, list):
                for item in schema_part:
                    fix_schema(item)
            return schema_part

async def get_tools(session: ClientSession) -> List[FunctionTool]:
    await session.initialize()
    tool_definitions = await session.list_tools()
    llama_tools=[]
    for tool_def in tool_definitions.tools:
        fixed_schema = fix_schema(tool_def.inputSchema)

        pydantic_model= create_model(
            f"{tool_def.name}_Schema",
            **{prop: (dict,None) for prop in fixed_schema.get("properties",{}).keys()}
        )

        async def tool_function(tool_name:str = tool_def.name,**kwargs):
            print(f"Executing tool '{tool_name}' with arguments: {kwargs}")
            result=await session.call_tool(tool_name,kwargs)
            return result
        
        tool=FunctionTool.from_defaults(
            fn=tool_function,
            name=tool_def.name,
            description=tool_def.description,
            fn_schema=pydantic_model    
        )
        llama_tools.append(tool)
    return llama_tools

async def get_agent(tools:list,llm,sys_prompt:str):

    agent = FunctionAgent(
        name="My Agent",
        description="An agent that can search the web",
        tools=tools,
        llm=llm,
        system_prompt=sys_prompt,
    )
    return agent

async def handle_user_message(
        message_content: str,
        agent: FunctionAgent,
        agent_context: Context,
        verbose: bool = False,
    ):
    handler = agent.run(message_content,ctx=agent_context)
    async for event in handler.stream_events():
        if verbose and type(event)==ToolCall:
            print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
        elif verbose and type(event) == ToolCallResult:
            print(f"Tool {event.tool_name} returned {event.tool_output}")
    response = await handler
    return str(response)

if True:
    async def main():
        try:
            SYSTEM_PROMPT = """\
            You are an AI assistant for Tool Calling.

            Before you help a user, you need to work with tools to interact with the website
            """

            open_server = subprocess.Popen(["npx", "@playwright/mcp@latest", "--port", "8050"])
            time.sleep(10)
            
            llm=Ollama(model="llama3.1",request_timeout=300)

            async with sse_client("http://localhost:8050/sse") as (read, write):
                async with ClientSession(read_stream=read, write_stream=write) as session:
                    
                    fixed_tools=await get_tools(session)
                    agent = await get_agent(tools=fixed_tools,llm=llm,sys_prompt=SYSTEM_PROMPT)
                    agent_context = Context(agent)

                    while True:
                        user_input = input("Enter your message: ")
                        if user_input.lower() == "exit":
                            break
                        print("User: ",user_input)
                        response = await handle_user_message(user_input,agent,agent_context,verbose=True)
                        print("Agent: ",response)
        finally:
            if open_server:
                open_server.terminate()

if False:
    async def get_tools(session: ClientSession):
        await session.initialize()
        tools=[]
        result=await session.list_tools()
        for i in result.tools:
            object={
                "type":"function",
                "function": {
                    "name": i.name,
                    "description": i.description,
                    "parameters": i.inputSchema,
                },
            }
            tools.append(object)
        return tools

    async def main():
        try:
            open_server = subprocess.Popen(["npx", "@playwright/mcp@latest", "--port", "8050"])
            time.sleep(3)
            async with sse_client("http://localhost:8050/sse") as (read,write):
                async with ClientSession(read_stream=read,write_stream=write) as session:
                    tools=await get_tools(session)
                    print(tools)
        except Exception as e:
            print("couldnt open, with error: ",e)

        finally:
            pass

if __name__ == "__main__":
    asyncio.run(main())