from find_available_slots import get_available_slots
from book_slot import book_slot
from find_nearest import find_nearest_service_center
from normalise_date import normalize_datetime
import asyncio
import numpy as np
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("customer_agent")

@mcp.tool()
async def find_center(city):
    return find_nearest_service_center(city)

@mcp.tool()
async def get_slot(center_id,date):
    return get_available_slots(center_id,date)

@mcp.tool()
async def book_slot(center_id,date,time):
    return book_slot(center_id,date,time)

@mcp.tool()
async def normalise_date(date):
    return normalize_datetime(date)

if __name__ == "__main__":
    # ✅ DO NOT wrap in asyncio.run()
    # ✅ DO NOT create async main()
    mcp.run(transport="stdio")

