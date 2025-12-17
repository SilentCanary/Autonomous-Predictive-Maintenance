import asyncio
from mcp_client import MCPClient


async def main():
    client = MCPClient()

    try:
        await client.connect("tools/tool.py")

        tools = await client.list_tools()
        print("\nAvailable tools:")
        for t in tools.tools:
            print("-", t.name)

        result = await client.call_tool(
            "find_center",
            {"city": "Shamli"}
        )

        print("\nTool result:", result)

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())