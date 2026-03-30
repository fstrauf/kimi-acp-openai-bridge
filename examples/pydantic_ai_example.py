"""Example: Using Kimi ACP Bridge with PydanticAI.

Install dependencies:
    pip install pydantic-ai

Run the bridge first:
    kimi-acp-bridge

Then run this script:
    python pydantic_ai_example.py
"""

import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


async def basic_chat():
    """Basic chat completion example."""
    model = OpenAIModel(
        "kimi-k2.5",
        base_url="http://localhost:8080/v1",
        api_key="dummy"  # Bridge ignores this
    )
    
    agent = Agent(model)
    
    result = await agent.run("Hello, Kimi! What can you help me with?")
    print("Basic Chat:")
    print(result.data)
    print()


async def with_system_prompt():
    """Chat with system prompt."""
    model = OpenAIModel(
        "kimi-k2.5",
        base_url="http://localhost:8080/v1",
        api_key="dummy"
    )
    
    agent = Agent(
        model,
        system_prompt="You are a Rust expert. Provide concise, idiomatic Rust code examples."
    )
    
    result = await agent.run("How do I read a file in Rust?")
    print("With System Prompt:")
    print(result.data)
    print()


async def with_tools():
    """Chat with tool usage."""
    model = OpenAIModel(
        "kimi-k2.5",
        base_url="http://localhost:8080/v1",
        api_key="dummy"
    )
    
    agent = Agent(model)
    
    # Define a simple tool
    @agent.tool
    async def get_current_time(ctx) -> str:
        """Get the current time."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    result = await agent.run("What time is it now?")
    print("With Tools:")
    print(result.data)
    print()


async def multi_turn_conversation():
    """Multi-turn conversation."""
    model = OpenAIModel(
        "kimi-k2.5",
        base_url="http://localhost:8080/v1",
        api_key="dummy"
    )
    
    agent = Agent(model)
    
    # First turn
    result1 = await agent.run("My name is Alice.")
    print("Turn 1:")
    print(result1.data)
    
    # Second turn - uses conversation history
    result2 = await agent.run("What's my name?", message_history=result1.all_messages())
    print("\nTurn 2:")
    print(result2.data)
    print()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Kimi ACP Bridge + PydanticAI Examples")
    print("=" * 60)
    print()
    
    try:
        await basic_chat()
        await with_system_prompt()
        await with_tools()
        await multi_turn_conversation()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the bridge is running:")
        print("  kimi-acp-bridge")


if __name__ == "__main__":
    asyncio.run(main())
