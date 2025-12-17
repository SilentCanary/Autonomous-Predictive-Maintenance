import asyncio
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session = None
        self.read = None
        self.write = None

    async def connect(self, server_script_path: str):
        params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )
        read, write = await self.exit_stack.enter_async_context(stdio_client(params))
        self.read = read
        self.write = write

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.read, self.write)
        )
        await self.session.initialize()

    async def list_tools(self):
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, args: dict):
        return await self.session.call_tool(tool_name, args)

    async def close(self):
        if self.session:
            await self.session.close()

    async def cleanup(self):
        await self.exit_stack.aclose()