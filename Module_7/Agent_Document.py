import os
import json
import ollama
from pypdf import PdfReader

MODEL = "llama3.1"
ALLOWED_DIR = os.path.abspath(r"C:\Users\rajarajan.ramanathan\Documents\DS_GenAI\Module_7\Documents")
MAX_FILE_SIZE = 3_000_000

def GetFileContent(fileName: str):
    target_path = os.path.abspath(os.path.join(ALLOWED_DIR, fileName))

    if not target_path.startswith(ALLOWED_DIR):
        return {"error": "Access denied: file is outside the allowed directory."}

    if not os.path.isfile(target_path):
        available = os.listdir(ALLOWED_DIR)
        return {"error": f"File not found: {fileName}", "available_files": available}

    if os.path.getsize(target_path) > MAX_FILE_SIZE:
        return {"error": f"File too large to read (max {MAX_FILE_SIZE} bytes)."}

    ext = os.path.splitext(target_path)[1].lower()

    try:
        if ext == ".pdf":
            reader = PdfReader(target_path)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
    except Exception as e:
        return {"error": f"Could not read file: {e}"}

    return {"fileName": fileName, "content": content[:20000]}


def GetPDFList():
    return {"files": os.listdir(ALLOWED_DIR)}


tools = [
    {
        "type": "function",
        "function": {
            "name": "GetFileContent",
            "description": (
                "Reads the contents of a file by its name from the Documents folder."
                "Access is restricted to files located within the permitted Documents directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fileName": {"type": "string", "description": "Name of the file to read"},
                },
                "required": ["fileName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GetPDFList",
            "description": "List all files currently available for reading within the Documents folder.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

available_functions = {"GetFileContent": GetFileContent, "GetPDFList": GetPDFList}

messages = [{
    "role": "user",
    "content": "List the available files, then read the relevant file and provide a summary of its contents."
}]

response = ollama.chat(model=MODEL, messages=messages, tools=tools)
msg = response["message"]
print("DEBUG tool_calls:", msg.get("tool_calls"))
messages.append(msg)

if msg.get("tool_calls"):
    for call in msg["tool_calls"]:
        func_name = call["function"]["name"]
        args = call["function"]["arguments"]

        if func_name not in available_functions:
            result = {"error": f"Unknown tool '{func_name}'"}
        else:
            result = available_functions[func_name](**args)

        messages.append({"role": "tool", "content": json.dumps(result)})

    final = ollama.chat(model=MODEL, messages=messages)
    print(final["message"]["content"])
else:
    print(msg["content"])