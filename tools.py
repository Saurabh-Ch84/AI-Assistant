from datetime import datetime
import requests
from Utils.utils import log
import json

from rag import search_top_k

def get_current_time(**kwargs):
    return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

def get_current_city_weather(city, **kwargs):
    url = f'https://wttr.in/{city}?format=3'
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text

def get_user_current_location(**kwargs):
    return 'kolkata'

def search_knowledge_base(query, **kwargs):
    results = search_top_k(query, 2)
    log(f" Knowledge Base Search: {query}")
    for r in results:
        # Fixed nested f-string quotes for Python < 3.12 compatibility
        score_val = r['score']
        source_val = r['source']
        chunk_val = r['chunk_id']
        log(f" {score_val:.2f}-{source_val}-#{chunk_val}")
    
    blocks = []
    for r in results:
        sourceFileName = r["source"]
        chunkId = r["chunk_id"]
        chunkText = r["text"]
        header = f"[{sourceFileName} #{chunkId}]"
        block = f"{header}\n{chunkText}"
        blocks.append(block)
    return "\n\n".join(blocks)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Fetches the current local time and date. Call this whenever the user asks 'what time is it', 'what is the date', or 'current time'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_current_location",
            "description": "CRITICAL: Call this tool FIRST whenever the user asks about 'my location', 'here', 'where I am', or asks for local weather without specifying a city name. NEVER ask the user for their location; ALWAYS call this tool to fetch it automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_city_weather",
            "description": "Fetches the weather forecast for a SPECIFIC city. If the user does not provide a city name (e.g., says 'my location'), you MUST call get_user_current_location first to get the city name before calling this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The exact name of the city to get the weather for (e.g., 'Kolkata', 'Delhi')."
                    }
                },
                "required": ["city"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches the Zorvex Technologies company knowledge base: leave, medical, HR, hiring, rca-docs, tech, incident-reports and travel expense policies. Use it for any question about company policies. Returns the most relevant passages with their source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {  # Fixed: Changed from 'city' to 'query'
                        "type": "string",
                        "description": "The user's question, rewritten as a short, self-contained search query."
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_current_city_weather": get_current_city_weather,
    "get_user_current_location": get_user_current_location,
    "search_knowledge_base": search_knowledge_base
}

def run_tool(tool_call):
    function_name = tool_call.function.name
    try:
        function_arguments = json.loads(tool_call.function.arguments or "{}")
        log(f" {function_name}({function_arguments})")
        function = FUNCTION_NAME_TO_FUNCTION_MAP[function_name]
        result = function(**function_arguments)
    except Exception as e:
        result = f"Error while running {function_name}: {e}"    
    log(f"    {result}")
    return result