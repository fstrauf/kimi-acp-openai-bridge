"""Example: Using Kimi ACP Bridge with OpenAI Python client.

Install dependencies:
    pip install openai

Run the bridge first:
    kimi-acp-bridge

Then run this script:
    python openai_client_example.py
"""

from openai import OpenAI


def main():
    """Run examples with OpenAI client."""
    # Configure client to use local bridge
    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="dummy",  # Bridge ignores this
    )

    print("=" * 60)
    print("Kimi ACP Bridge + OpenAI Client Examples")
    print("=" * 60)
    print()

    # Example 1: Simple completion
    print("Example 1: Simple Completion")
    print("-" * 40)
    try:
        response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "Hello! What's the weather like?"}],
        )
        print(f"Response: {response.choices[0].message.content}")
        print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Example 2: Streaming
    print("Example 2: Streaming Completion")
    print("-" * 40)
    try:
        stream = client.chat.completions.create(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True,
        )

        print("Response: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Example 3: With system message
    print("Example 3: With System Message")
    print("-" * 40)
    try:
        response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
                {"role": "user", "content": "How do I define a function in Python?"},
            ],
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Example 4: With tools
    print("Example 4: With Tools")
    print("-" * 40)
    try:
        response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "What files are in the current directory?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Directory path"}
                            },
                        },
                    },
                }
            ],
        )

        message = response.choices[0].message
        if message.tool_calls:
            print(f"Tool calls: {message.tool_calls}")
        else:
            print(f"Response: {message.content}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Example 5: List models
    print("Example 5: List Models")
    print("-" * 40)
    try:
        models = client.models.list()
        for model in models.data:
            print(f"  - {model.id} (owned by: {model.owned_by})")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
