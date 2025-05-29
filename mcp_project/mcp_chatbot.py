from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import os
import json

nest_asyncio.apply()
load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        self.session: ClientSession = None
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.available_tools: List[dict] = []

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

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python3",
            args=["research_server.py"],
            env=None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
    
                response = await session.list_tools()
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                self.available_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in tools]

                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
