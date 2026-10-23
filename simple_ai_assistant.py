# standard library imports
import os 
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
        "content":"Use you knowledge for questions related to programming and science. Use tools call only when necessary and answer within 100 words only."
    }
)

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
        response=client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages,
            tools=TOOLS, # passing the tools to the model so that it can call them if needed.
        )

        msg=response.choices[0].message

        if not msg.tool_calls:
            break

        log(f"Iteration {iteration+1}: model requested {len(msg.tool_calls)} tool call(s)")

        messages.append(msg)
        for tool_call in msg.tool_calls:
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

    messages.append({
        "role": "assistant",
        "content": msg.content
    })

    print("Jarvis: "+(msg.content or ""))