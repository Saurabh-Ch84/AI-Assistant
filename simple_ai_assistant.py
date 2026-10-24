# standard library imports
import os 
from types import SimpleNamespace

# third party library imports
from openai import OpenAI
from dotenv import load_dotenv

# local modules
from tools import TOOLS, run_tool
from Utils.utils import log

""""" 
    LLMs are always stateless, no-memory and every API call is new for it.
    So we append the old messages to a new message every time to give context, this results in the 
    context window problem as conversation goes on.
    Techniques to solve context window problem: RAG, Summarization, Compaction, Sliding Window.

    Role: User-> End User(Human using Assistant), Assistant-> LLM, System-> Developer(System Level Instructions).
    Good Prompt: R: Role, C: Constraints, C: Context, T: Task, E: Examples (Few-Shot Prompting), F: Format.
    Prompt Injection: A bad prompt breaking the assitant's behavior.

    Tool-calling: LLMs can't execute code, but they can call tools to do so. Tools are functions that can be called by the LLM to perform specific tasks.  
    The LLM can call a tool by specifying the tool's name and providing the necessary parameters. 
    The tool will then execute the function and return the result to the LLM, which can then use that result to generate a response to the user.
    Main Types: 1. Simple, 2. Parallel, 3. Chained(Sequential).
"""""

load_dotenv()

client=OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

MAX_LIMIT=7

messages=[]

# small example of prompt engineering.
messages.append(
    # adding system instructions.
    {
        "role":"system",
        "content":"Use your knowledge for questions related to programming and science. Use tools call only when necessary and answer within 100 words only."
    }
)

def stream_chat(history):
    stream=client.chat.completions.create(
        model=os.getenv("MODEL"), # type: ignore
        messages=history,
        tools=TOOLS, # type: ignore
        stream=True
    )

    full_response=''
    calls={}
    label_printed=False

    try:
        # Streaming started
        for chunk in stream:
            if not chunk.choices:
                continue
            delta=chunk.choices[0].delta
            # Content Streaming.
            if delta.content:
                if not label_printed:
                    print("Assistant: ",end="",flush=True)
                    label_printed=True
                print(delta.content, end="", flush=True)
                full_response+=delta.content
                # Tool Calls
                for tc in (delta.tool_calls or []):
                    obj=calls.setdefault(
                        tc.index,
                        SimpleNamespace(id=None, function=SimpleNamespace(
                            name="",arguments=""
                        )), 
                    )
                    if tc.id:
                        obj.id=tc.id
                    if tc.function and tc.function.name:
                        obj.function.name+=tc.function.name
                    if tc.function and tc.function.arguments:
                        obj.function.arguments+=tc.function.arguments
    except KeyboardInterrupt :
        print("\n Stopped by User.")
        calls={}
        
    if label_printed:
        print() # move to the next line after the assistant's response.  
    calls_list=[]
    for key in sorted(calls):
        calls_list.append(calls[key])
    return full_response, calls_list


def build_tool_calls_payload(tool_calls):
    """converts our collected tool calls into the dictionary format that the API expects inside an assistant message"""
    payload=[]
    for tool_call in tool_calls:
        tool_call_dict={
            "id":tool_call.id,
            "type": "function",
            "function":{
                "name":tool_call.function.name,
                "arguments":tool_call.function.arguments,
            },
        }
        payload.append(tool_call_dict)

    return payload


while True:
    user_input=input("\nUser: ")
    if user_input.lower() in ["exit","quit","bye"]:
        break
    messages.append(
        {
            "role":"user",
            "content":user_input
        }
    )

    # Chained Tool Calling.
    for iteration in range(MAX_LIMIT):
        # openai client api
        full_response, tool_calls=stream_chat(messages) 

        if not tool_calls:
            messages.append({
                "role":"assistant",
                "content":full_response
            })
            break 

        log(f"Iteration {iteration+1}: model requested {len(tool_calls)} tool call(s)")

        messages.append({
            "role":"assistant",
            "content":full_response if full_response else None,
            "tool_calls":build_tool_calls_payload(tool_calls)
        })

        for tool_call in tool_calls:
            result=run_tool(tool_call)

            messages.append({
                "role":"tool",
                "tool_call_id": tool_call.id,
                "content":str(result)
            })

    else :
        # In Python, the else block after a for loop is executed only if the loop completes without encountering a break statement.
        fallback="Sorry, I could not finish this within the step limit"
        messages.append({
            "role": "assistant",
            "content": fallback
        })
        print("Max iterations reached, stopping tool calls.")
        continue
