import os 
from types import SimpleNamespace

from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOLS, run_tool
from Utils.utils import log

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

MAX_LIMIT = 7
messages = []

messages.append({
    "role": "system",
    "content": (
        "You are an AI assistant for Zorvex Technologies.\n\n"
        "ROUTING RULES:\n"
        "1. WEATHER & TIME: If the user asks about weather or time, call `get_current_city_weather`, `get_current_time`, or `get_user_current_location` immediately.\n"
        "2. ZORVEX POLICIES & HR: For any question about Zorvex policies, benefits, leave, medical, or internal tech processes, you MUST call `search_knowledge_base` FIRST.\n"
        "3. GENERAL KNOWLEDGE: Answer programming, math, and general science from your internal knowledge without tools.\n\n"
        "FALLBACK RULE:\n"
        "ONLY if `search_knowledge_base` was executed and returned no relevant context, reply: 'I could not find this in the knowledge base.'"
    )
})

def stream_chat(history):
    # Force model to output tool_calls structure
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"), # type: ignore
        messages=history,
        tools=TOOLS, # type: ignore
        tool_choice="auto",  # Explicitly tell the endpoint to enable tool calling
        stream=True
    )

    full_response = ''
    calls = {}
    label_printed = False

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 1. Content Streaming
            if delta.content:
                if not label_printed:
                    print("Assistant: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

            # 2. Tool Calls Processing (Un-nested from if delta.content)
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(
                        id=None,
                        function=SimpleNamespace(name="", arguments="")
                    ), 
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments

    except KeyboardInterrupt:
        print("\n Stopped by User.")
        calls = {}
        
    if label_printed:
        print()
        
    calls_list = [calls[key] for key in sorted(calls)]
    return full_response, calls_list

def build_tool_calls_payload(tool_calls):
    payload = []
    for tool_call in tool_calls:
        tool_call_dict = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        payload.append(tool_call_dict)
    return payload

while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    messages.append({
        "role": "user",
        "content": user_input
    })

    for iteration in range(MAX_LIMIT):
        full_response, tool_calls = stream_chat(messages) 

        if not tool_calls:
            messages.append({
                "role": "assistant",
                "content": full_response
            })
            break 

        log(f"Iteration {iteration+1}: model requested {len(tool_calls)} tool call(s)")

        messages.append({
            "role": "assistant",
            "content": full_response if full_response else None,
            "tool_calls": build_tool_calls_payload(tool_calls)
        })

        for tool_call in tool_calls:
            result = run_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    else:
        fallback = "Sorry, I could not finish this within the step limit"
        messages.append({
            "role": "assistant",
            "content": fallback
        })
        print("Max iterations reached, stopping tool calls.")
        continue