from smolagents import ToolCallingAgent, ToolCollection, LiteLLMModel
from mcp import StdioServerParameters

# choose the Ollama LLM model usign LiteLLM
model = LiteLLMModel(
    model_id="ollama_chat/gemma:2b",
    num_ctx=8192
)

# command to access the MCP server
server_parameters = StdioServerParameters(
    command="uv",
    args=["run", "mcp_server.py"],
    env=None
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
    agent.run("what is in the file called mcptest.txt, give me exact words written in the file")