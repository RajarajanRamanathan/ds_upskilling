import ollama
import pyodbc
import json
from decimal import Decimal

MODEL = "llama3.1"

DB_CONFIG = {
    "server": "I20101",     
    "database": "my_database",
    "username": "sa",
    "password": "123456789",
    "driver": "{ODBC Driver 18 for SQL Server}",
}

def get_connection():
    conn_str = (
        f"DRIVER={DB_CONFIG['driver']};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"UID={DB_CONFIG['username']};"
        f"PWD={DB_CONFIG['password']};"
        f"TrustServerCertificate=yes;" 
    )
    return pyodbc.connect(conn_str)



def Search_Book(name: str = None, limit: int = 10):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    limit = min(limit, 20)

    conn = get_connection()
    cur = conn.cursor()

    if name:
        cur.execute(
            f"SELECT TOP {limit} ID, Name, Stock_Count, Created_On "
            f"FROM Books WHERE Name = ?",
            (name,),
        )
    else:
        cur.execute(f"SELECT TOP {limit} ID, Name, Stock_Count, Created_On FROM Books")

    columns = [col[0] for col in cur.description]
    rows = []
    for row in cur.fetchall():
        record = dict(zip(columns, row))
        for key, value in record.items():
            if isinstance(value, Decimal):
                record[key] = float(value)
        rows.append(record)

    conn.close()
    return rows


tools = [
    {
        "type": "function",
        "function": {
            "name": "Search_Book",
            "description": "Search books in the SQL Server database, optionally filtered by it's name. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Filter by book name (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 10, max 20)",
                    },
                },
                "required": [],
            },
        },
    }
]

available_functions = {"Search_Book": Search_Book}

# --- 4. Run the conversation ---
messages = [{"role": "user", "content": "Is the book named agile available?"}]

response = ollama.chat(model=MODEL, messages=messages, tools=tools)
msg = response["message"]
messages.append(msg)

if msg.get("tool_calls"):
    for call in msg["tool_calls"]:
        func_name = call["function"]["name"]
        args = call["function"]["arguments"]
        result = available_functions[func_name](**args)
        messages.append({"role": "tool", "content": json.dumps(result)})

    final = ollama.chat(model=MODEL, messages=messages)
    print(final["message"]["content"])
else:
    print(msg["content"])