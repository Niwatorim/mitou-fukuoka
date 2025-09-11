from mcp.server.fastmcp import FastMCP
import typing

mcp = FastMCP(
    name="calc",
    host="0.0.0.0", # only for SSE, localhost
    port=8050 # only for SSE
)

@mcp.tool()
def read_child(file_path):
    """read the following text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

if __name__ == "__main__":
    mcp.run(transport="stdio")