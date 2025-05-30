from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import List, Dict, TypedDict
import asyncio
import nest_asyncio
import os
import json

# nest_asyncio.apply()
load_dotenv()

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class MCP_ChatBot:

    def __init__(self):
        self.sessions: List[ClientSession] = []
        # context manager that will manage the mcp client objects and their sessions and ensures that they are properly closed:
        self.exit_stack = AsyncExitStack()
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.available_tools: List[ToolDefinition] = []
        # maps the tool name to the corresponding client session:
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name:str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query):
        messages = [{"role": "user", "content": query}]
        process_query = True

        while process_query:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
                max_tokens=2024
            )

            message = response.choices[0].message
            tool_call = message.tool_calls[0] if message.tool_calls else None

            if tool_call:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id

                print(f"Calling tool {tool_name} with args {tool_args}")

                session = self.tool_to_session[tool_name] 
                result = await session.call_tool(tool_name, arguments=tool_args)

                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                        }
                    }]
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result.content
                })

            else:
                print(message.content)
                process_query = False

                message = response.choices[0].message
                tool_call = message.tool_calls[0] if message.tool_calls else None

                if tool_call:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id

                    print(f"Calling tool {tool_name} with args {tool_args}")

                    result = await self.session.call_tool(tool_name, arguments=tool_args)

                    messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "id": tool_id,
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args)
                            },
                            "type": "function"
                        }]
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result.content
                    })
                else:
                    print(message.content)
                    process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()

async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
