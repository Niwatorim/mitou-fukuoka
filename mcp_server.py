from google.genai import types
from dotenv import load_dotenv
from rich.panel import Panel
from google import genai
import os,yaml,json
from typing import Any, List, Optional
import streamlit as st
import json
import time
import os
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from rich.console import Console
from rich.panel import Panel

#TODO: Make into json format

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

class MCPNeo4J: #get the gemini agent ready
    def __init__(self, model= "gemini-2.0-flash"):
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.tools= None
        self.server_params= None
        self.server_name= None

    def server_choose(self):
        # Get the path to mcp.json relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_path = os.path.join(script_dir, "pages", "mcp.json")
        
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

    async def agent_loop(self, prompt: str, system_prompt) -> Any:
        msg_content = prompt
        if isinstance(prompt, list) and len(prompt) > 0:
             last_msg = prompt[-1]
             if hasattr(last_msg, 'content'):
                 msg_content = last_msg.content
             elif isinstance(last_msg, str):
                 msg_content = last_msg
             else:
                 msg_content = str(last_msg)
        
        contents = [types.Content(
            role="user",
            parts=[types.Part(text=msg_content)]
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
            system_instruction=system_prompt,
            temperature=0,
            tools=[tools],
        )

        # Initial call
        CONSOLE.print("[bold magenta] Initial call [/bold magenta]")
        print(f"[yellow] Requesting initial response from Gemini with tools: {[t.name for t in mcp_tools.tools]} [/yellow]")
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

    async def chat(self,request = None, system_prompt = None):
        print(f"MCP-assistant connected alh")
        try:
            res = None
            if request is not None:
                res = await self.agent_loop(request,system_prompt=system_prompt)
            if res is not None:
                CONSOLE.print(f"[bold green]{res.text} [/bold green]")
            else:
                print(res)
            return res
        except Exception as e:
            error_msg = f"Error occurred: {e}"
            print(error_msg)
            return types.GenerateContentResponse(
                candidates=[types.Candidate(
                    content=types.Content(
                        parts=[types.Part(text=error_msg)],
                        role="model"
                    )
                )]
            )

    async def cleanup(self):
        await self.exit_stack.aclose()

async def generator(prompt):
    if not isinstance(prompt, str):
        prompt = str(prompt)
    system_prompt="""
    You are a Senior QA Automation Engineer.
    Convert the following execution history into a **Pytest-Playwright** test file.
    
    RULES:
    1. **Structure**: Use the standard `def test_scenario(page: Page):` format.
    2. **Assertions**: A test is meaningless without checks. You MUST include `expect()` assertions.
       - If the user clicked a button that increments a counter, assert the new text (e.g., `expect(button).to_contain_text(...)`).
       - If the user navigated, assert the URL or page title.
    3. **Cleanup**: Remove redundant steps (like repeated navigations).
    4. **Syntax**: 
       - `from playwright.sync_api import Page, expect`
       - Use `page.get_by_role` or `page.locator` with robust regex selectors.
    5. **Regex**: When using Regex selectors in Python, you MUST import re and use re.compile(r'pattern'). Do NOT pass raw regex strings.
    
    Output ONLY the python code block.
    
    """
    model="gemini-2.5-flash"
    genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    contents = [types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )]
    config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0
            )
    response = await genai_client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
    return response.candidates[0].content.parts[0].text

class MCPPlaywright: #get the gemini agent ready
    def __init__(self, model= "gemini-2.5-flash"):
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.tools= None
        self.server_params= None
        self.server_name= None

    def server_choose(self):
        # Get the path to mcp.json relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_path = os.path.join(script_dir, "pages", "mcp.json")
        
        with open(mcp_path, "r") as f:
            mcp_config = json.load(f)
        servers = mcp_config["mcpServers"]
        server_names=list(servers.keys())
        #assuming just using neo4j
        self.server_name = server_names[1]
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

    async def agent_loop(self, prompt: str, system_prompt) -> Any:
            msg_content = prompt
            if isinstance(prompt, list) and len(prompt) > 0:
                last_msg = prompt[-1]
                if hasattr(last_msg, 'content'):
                    msg_content = last_msg.content
                elif isinstance(last_msg, str):
                    msg_content = last_msg
                else:
                    msg_content = str(last_msg)

            tool_history:list[dict[str]]=[]

            contents = [types.Content(
                role="user",
                parts=[types.Part(text=msg_content)]
            )]
            
            mcp_tools = await self.session.list_tools()
            tools_def = types.Tool(function_declarations=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": clean_schema(getattr(tool, "inputSchema", {}))
                }
                for tool in mcp_tools.tools
            ])
            self.tools = tools_def

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                tools=[tools_def], # Start with tools enabled
            )

            # Initial call
            CONSOLE.print("[bold magenta] Initial call [/bold magenta]")
            print(f"[yellow] Requesting initial response from Gemini with tools: {[t.name for t in mcp_tools.tools]} [/yellow]")
            response = await self.genai_client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )
            contents.append(response.candidates[0].content)
            
            turn_count = 0
            max_tool_turns = 15
            
            # Flag to track if we should stop using tools (e.g. after browser_close)
            tools_active = True

            while response.function_calls and turn_count < max_tool_turns:
                turn_count += 1
                tool_response_parts: List[types.Part] = []
                
                # 1. Execute all tools requested
                for fc_part in response.function_calls:
                    tool_name = fc_part.name
                    args = fc_part.args or {}
                    
                    tool_history.append({
                        "step": len(tool_history)+1,
                        "tool":tool_name,
                        "parameters":args
                    })

                    if tool_name == "browser_close":
                        tools_active = False
                        
                    CONSOLE.print(
                        Panel(f"Invoking MCP tool {tool_name} with args: {args}",title="tool",expand=True))
                    
                    try:
                        tool_result = await self.session.call_tool(tool_name, args)
                        print(f"Tool {tool_name} done")
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

                # 2. Add tool outputs to history
                contents.append(types.Content(
                    role="user",
                    parts=tool_response_parts
                ))
                

                current_config = config
                if not tools_active:
                    print("Browser closed. Forcing text-only response (Removing tools).")
                    current_config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0,
                        tools=None
                    )

                CONSOLE.print(
                        Panel(f"[bold yellow] Requesting updated response from Gemini (Turn {turn_count}) [/bold yellow] ",title="turn",expand=True))
                
                response = await self.genai_client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=current_config
                )
                contents.append(response.candidates[0].content)
                
                if not tools_active:
                    break

                time.sleep(0.5)

            if turn_count >= max_tool_turns and response.function_calls:
                print(f"Max tool count reached so stopping")
            
            return response,tool_history

    async def chat(self, request=None, system_prompt=None):
        print(f"MCP-assistant connected alh")
    
        res = None
        tool_history = [] 

        try:
            if request is not None:
                res, tool_history = await self.agent_loop(request, system_prompt=system_prompt)
            
            if res is not None:
                CONSOLE.print(f"[bold green]{res.text} [/bold green]")
            else:
                print("No response generated.")

            return res, tool_history

        except Exception as e:
            error_msg = f"Error occurred: {e}"
            print(error_msg)
            
            error_res = types.GenerateContentResponse(
                candidates=[types.Candidate(
                    content=types.Content(
                        parts=[types.Part(text=error_msg)],
                        role="model"
                    )
                )]
            )
            
            return error_res, []

    async def cleanup(self):
        await self.exit_stack.aclose()
