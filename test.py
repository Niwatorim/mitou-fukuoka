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
from browser_use import Agent, ChatOllama
from crawl4ai import AsyncWebCrawler,CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from bs4 import BeautifulSoup
import json

def path_generation(element):
    values=[]
    child=element
    for parent in child.parents: #for every parent object
        siblings=parent.find_all(child.name,recursive=False)
        if len(siblings) > 1: #if has siblings
            count = 1
            for sib in siblings:
                if sib is element:
                    values.append(f"{child.name}[{count}]")
                    break
                count+=1
        else:
            values.append(child.name)
        if parent.name == '[document]':
            break
        child=parent
    values.reverse()
    return "/" + "/".join(values)


def beautiful(data):
    soup = BeautifulSoup(data,"lxml")
    tags=["a", "button", "input", "select", "textarea", "form", "h1", "h2", "h3", "p", "img", "li"]
    interaction_map=[]
    for element in soup.find_all(tags):
        xpath=path_generation(element)

        attributes={}
        attributes_to_find=["id", "class", "name", "href", "src", "alt", "type", "value", "placeholder", "role", "aria-label"]
        for attribute in element.attrs:
            if attribute in attributes_to_find:
                attributes[attribute]=element.attrs[attribute]
        item={
            "tag": element.name,
            "text":element.get_text(strip=True),
            "locator":xpath,
            "attributes":attributes
        }
        interaction_map.append(item)
    return interaction_map


async def main():

    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2,
            include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
    )

    site="https://www.techwithtim.net"
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url=site,
            config=config
        )
        scraped=[]
        content={}
        for result in results:
            if result.url not in scraped:
                url=str(result.url).replace(site,"")
                content[url]=beautiful(result.html)
                scraped.append(result.url)
        print(content)
        with open("temp.json","w") as f:
            f.write(json.dumps(content,indent=2))


"""
1. crawl4ai -> crawl each website to get raw html
for crawled pages:
2.    beautiful soup function

Beautiful soup function-> recursive function with html object to be checked and path
 excluded tags include: ["script","style","link","meta","noscript","br","hr","span","b","i","u","strong","em","div"]
 for i in tags:
    if i not in exluded tags:
        ans = write the attributes and the path in json form
        add the path
        yield ans 
    else:
        add the path
    
    if i is dict:
        open up the stuff and recall the function with that new value (enter the dict)
    if i is list:
        for i in list:
            recall the function but with i as the new value

3. final values are flat json with each wanted html and its corresponding path
        
"""




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

#For browseruse
if False:
    async def main(): #for browser use
        agent = Agent(
            task="Do the following: 1. Open https://www.techwithtim.net/tutorials. 2. Click on the text that says 'Courses' in the navbar. it will be a linked tab and will navigate you to https://www.techwithtim.net/courses. please confirm if it does this",
            llm=ChatOllama(model="llama3.1:8b"),
        )
        await agent.run()

#For llama_index
if False:
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

#For llama index
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