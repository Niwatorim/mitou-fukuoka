import asyncio
import os

from langchain_ollama import ChatOllama
from browser_use import Agent
from pydantic import BaseModel

# Create a custom ChatOllama class that is fully compatible
class MyChatOllama(ChatOllama):
    # Add the 'provider' attribute for logging
    provider: str = "ollama"

    # Add the 'model_name' property as an alias for 'model'
    @property
    def model_name(self) -> str:
        return self.model

    class Config:
        # Allow the library to add its own attributes
        extra = "allow"

async def run_search():
    agent = Agent(
        task=(
            'Go to https://www.dataedgehub.com, and list the title you see'
        ),
        # Use your final, fully compatible custom class
        llm=MyChatOllama(
            model='llama3.1:latest',
            num_ctx=128000,
        ),
        max_actions_per_step=1,
        tool_call_in_content=False
    )
    await agent.run()

if __name__ == '__main__':
    asyncio.run(run_search())
