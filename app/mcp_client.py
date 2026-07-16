# """
# Thin MCP client used by the RAG pipeline to call the custom manifest server
# (mcp_server/server.py) over stdio. This is the project's "MCP-connected tool".

# Usage:
#     async with MCPToolClient() as tools:
#         status = await tools.get_page_status("HR-001")
#         recent = await tools.list_recent_updates(space="ENG", since_date="2026-01-01")
# """
# import sys
# import json
# from contextlib import AsyncExitStack

# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client

# from app import config


# class MCPToolClient:
#     def __init__(self, server_script: str = None):
#         self.server_script = server_script or config.MCP_SERVER_SCRIPT
#         self._stack = AsyncExitStack()
#         self.session: ClientSession | None = None

#     async def __aenter__(self):
#         params = StdioServerParameters(
#             command=sys.executable,
#             args=[self.server_script],
#         )
#         read, write = await self._stack.enter_async_context(stdio_client(params))
#         self.session = await self._stack.enter_async_context(ClientSession(read, write))
#         await self.session.initialize()
#         return self

#     async def __aexit__(self, *exc):
#         await self._stack.aclose()

#     async def list_tools(self):
#         resp = await self.session.list_tools()
#         return [t.name for t in resp.tools]

#     async def _call(self, name: str, arguments: dict):
#         result = await self.session.call_tool(name, arguments)
#         # MCP tool results come back as a list of content blocks; our server
#         # returns a single JSON-serializable text block.
#         text = result.content[0].text if result.content else "null"
#         try:
#             return json.loads(text)
#         except json.JSONDecodeError:
#             return text

#     async def get_page_status(self, page_id: str):
#         return await self._call("get_page_status", {"page_id": page_id})

#     async def list_recent_updates(self, space: str = "", since_date: str = "1900-01-01"):
#         return await self._call(
#             "list_recent_updates", {"space": space, "since_date": since_date}
#         )

#     async def search_pages(self, keyword: str):
#         return await self._call("search_pages", {"keyword": keyword})


# # Small manual smoke test: `python -m app.mcp_client HR-001`
# if __name__ == "__main__":
#     import asyncio

#     async def _main():
#         page_id = sys.argv[1] if len(sys.argv) > 1 else "HR-001"
#         async with MCPToolClient() as tools:
#             print("Available tools:", await tools.list_tools())
#             print("Status check:", await tools.get_page_status(page_id))

#     asyncio.run(_main())
"""
Thin MCP client used by the RAG pipeline to call the custom manifest server
(mcp_server/server.py) over stdio. This is the project's "MCP-connected tool".
 
Usage:
    async with MCPToolClient() as tools:
        status = await tools.get_page_status("HR-001")
        recent = await tools.list_recent_updates(space="ENG", since_date="2026-01-01")
"""
import sys
import json
from contextlib import AsyncExitStack
 
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
 
from app import config
 
 
class MCPToolClient:
    def __init__(self, server_script: str = None):
        self.server_script = server_script or config.MCP_SERVER_SCRIPT
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
 
    async def __aenter__(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self
 
    async def __aexit__(self, *exc):
        await self._stack.aclose()
 
    async def list_tools(self):
        resp = await self.session.list_tools()
        return [t.name for t in resp.tools]
 
    async def _call(self, name: str, arguments: dict):
        result = await self.session.call_tool(name, arguments)
 
        # Prefer structuredContent when the SDK provides it - it's the
        # canonical field for a tool's parsed return value.
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
                structured = structured["result"]
            return self._coerce_string_items(structured)
 
        # Fallback: MCP tool results come back as a list of content blocks;
        # our server returns a single JSON-serializable text block.
        text = result.content[0].text if result.content else "null"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
 
        # FastMCP wraps non-object return types (list, str, int, etc.) in
        # {"result": <value>} because structuredContent must be a JSON
        # object per the MCP spec. Unwrap it so callers see the tool's
        # actual return value (e.g. a list) rather than that envelope.
        if isinstance(parsed, dict) and set(parsed.keys()) == {"result"}:
            parsed = parsed["result"]
 
        return self._coerce_string_items(parsed)
 
    @staticmethod
    def _coerce_string_items(value):
        """If a list's items are themselves JSON-encoded strings (a schema
        quirk seen when a tool's return type lacks item typing, e.g. `list`
        instead of `list[dict]`), parse each one back into a dict/object.
        """
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            try:
                return [json.loads(v) for v in value]
            except json.JSONDecodeError:
                return value
        return value
 
    async def get_page_status(self, page_id: str):
        return await self._call("get_page_status", {"page_id": page_id})
 
    async def list_recent_updates(self, space: str = "", since_date: str = "1900-01-01"):
        return await self._call(
            "list_recent_updates", {"space": space, "since_date": since_date}
        )
 
    async def search_pages(self, keyword: str):
        return await self._call("search_pages", {"keyword": keyword})
 
 
# Small manual smoke test: `python -m app.mcp_client HR-001`
if __name__ == "__main__":
    import asyncio
 
    async def _main():
        page_id = sys.argv[1] if len(sys.argv) > 1 else "HR-001"
        async with MCPToolClient() as tools:
            print("Available tools:", await tools.list_tools())
            print("Status check:", await tools.get_page_status(page_id))
 
    asyncio.run(_main())
 







