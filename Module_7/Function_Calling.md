# Function Calling & Tool Use — Learning Plan

Same format as before: plain-English definition → why it matters → runnable
code. This one assumes you're comfortable with the RAG material already
(prompting, grounding) and are now extending that to **agentic** behavior —
letting the model decide to call code, not just read retrieved text.

```
pip install openai anthropic google-generativeai tavily-python duckduckgo-search \
            requests jsonschema python-dotenv
```

---

## Suggested schedule (2 weeks)

| Days | Module | Focus |
|---|---|---|
| 1–2 | Core Concepts | Mental model + 3 provider APIs |
| 3–4 | Defining Tools | JSON Schema mastery |
| 5–6 | Executing Tool Calls | The request/response loop |
| 7 | Parallel Tool Calling | Concurrency |
| 8–10 | Building Custom Tools | 8 real tools |
| 11–12 | Tool Use Patterns | Production concerns |

---

# MODULE 1 — Core Concepts

## 1.1 What Function Calling Is

**Plain English:** You describe a set of functions to the model (name,
description, parameters — but not the function *code*). The model can't
execute anything itself; instead, when it decides a function would help, it
replies with structured JSON saying "call this function with these
arguments." **Your code** then actually runs the function and sends the
result back. The model never touches your system directly — it only ever
proposes a call.

**Why it matters:** this is the mechanism underneath nearly every "AI
agent" — search, calculators, database lookups, sending emails. RAG gives
the model *read-only* access to text; function calling gives it a
structured, auditable way to *ask you* to do things (read or write).

---

## 1.2 How It Works — the four-step loop

```
1. TOOL DEFINITION   You send the model a list of available tools
                      (name, description, JSON-schema parameters)
                      alongside the user's message.
                           │
                           ▼
2. MODEL DECISION     The model either answers directly, or responds with
                      a "tool_call": {name, arguments} instead of text.
                           │
                           ▼
3. EXECUTION          Your code reads {name, arguments}, matches it to a
                      real Python function, and runs it.
                           │
                           ▼
4. RESPONSE           You send the tool's output back to the model as a
                      new message. The model reads it and produces the
                      final natural-language answer (or calls another tool).
```

```python
# Minimal, provider-agnostic pseudocode for the loop above
def run_agent_loop(user_message, tools, tool_impls, llm_call, max_turns=5):
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_turns):
        response = llm_call(messages, tools)

        if not response.tool_calls:
            return response.text  # model gave a final answer

        for call in response.tool_calls:
            fn = tool_impls[call.name]          # step 3: execution
            result = fn(**call.arguments)
            messages.append(response.raw_message)
            messages.append({             # step 4: response
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })
    return "Max turns reached without a final answer."
```

**Why it matters:** every provider's SDK is a variation on this exact loop.
Once this shape is muscle memory, switching between OpenAI/Anthropic/Gemini
is just syntax, not new concepts.

---

## 1.3 OpenAI Function Calling Specification

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Chennai'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "What's the weather in Chennai?"}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",  # "auto" | "required" | "none" | {"type": "function", "function": {"name": "..."}}
)

msg = response.choices[0].message
if msg.tool_calls:
    call = msg.tool_calls[0]
    print(call.function.name)                       # "get_weather"
    print(json.loads(call.function.arguments))       # {"city": "Chennai"}
```

**Why it matters:** OpenAI's `tool_choice` parameter is the cleanest of the
three providers for forcing/preventing tool use — useful for testing (force
a specific tool) or for turns where you know no tool is needed (save
latency by passing `"none"`).

---

## 1.4 Anthropic Tool Use Specification

```python
import anthropic

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {          # <- note: "input_schema", not "parameters"
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Chennai'"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Chennai?"}],
)

for block in response.content:
    if block.type == "tool_use":
        print(block.name, block.input, block.id)  # id needed for the result round-trip

# response.stop_reason == "tool_use" tells you the model is waiting on a tool result
```

**Sending the result back (Anthropic requires the tool result as a
`tool_result` content block, matched by `tool_use_id`):**

```python
follow_up = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in Chennai?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": block.id, "content": "31°C, humid, partly cloudy"}
        ]},
    ],
)
print(follow_up.content[0].text)
```

**Why it matters:** the field names differ (`input_schema` vs
`parameters`, `tool_use`/`tool_result` vs OpenAI's `tool_calls`/`role:
"tool"`), but the four-step loop from 1.2 is identical — write your tool
executor once, and only the adapter layer changes per provider.

---

## 1.5 Gemini Function Calling

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_KEY")

weather_tool = genai.protos.Tool(function_declarations=[
    genai.protos.FunctionDeclaration(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "city": genai.protos.Schema(type=genai.protos.Type.STRING),
                "unit": genai.protos.Schema(
                    type=genai.protos.Type.STRING, enum=["celsius", "fahrenheit"]
                ),
            },
            required=["city"],
        ),
    )
])

model = genai.GenerativeModel("gemini-2.0-flash", tools=[weather_tool])
chat = model.start_chat()
response = chat.send_message("What's the weather in Chennai?")

fn_call = response.candidates[0].content.parts[0].function_call
print(fn_call.name, dict(fn_call.args))

# Send the result back
result = chat.send_message(
    genai.protos.Content(parts=[genai.protos.Part(
        function_response=genai.protos.FunctionResponse(
            name="get_weather",
            response={"result": "31°C, humid, partly cloudy"},
        )
    )])
)
print(result.text)
```

**Why it matters:** Gemini's schema uses its own `protos.Schema` types
instead of raw JSON dicts — more verbose, but conceptually the same `type /
properties / required` shape as JSON Schema underneath.

---

## 1.6 Function Calling vs RAG vs Plain Prompting

| | Plain Prompting | RAG | Function Calling |
|---|---|---|---|
| Data source | Model's training data only | Retrieved text chunks | Live code execution (API, DB, calculator...) |
| Can take actions? | No | No (read-only) | Yes (read or write) |
| Freshness | Frozen at training cutoff | As fresh as your index | Real-time (live API call) |
| Determinism | Model "recalls" facts (can hallucinate) | Grounded in retrieved text | Grounded in actual return value of code |
| Typical use | Reasoning, writing, summarizing | Q&A over your documents | Actions: search, compute, book, send, query |
| Failure mode | Hallucination | Retrieval miss / stale index | Wrong tool chosen, bad arguments, tool errors |

**Why it matters:** these are complementary, not competing — a serious
agent often uses all three: plain reasoning to plan, RAG to look up
knowledge, and function calling to act on it (e.g. RAG retrieves the
refund policy, a tool actually issues the refund).

---

# MODULE 2 — Defining Tools

## 2.1 JSON Schema for Tool Definitions

**Plain English:** All three providers' `parameters`/`input_schema` fields
are (a dialect of) [JSON Schema](https://json-schema.org/) — the same spec
used for API request validation. Learn JSON Schema once, reuse everywhere.

```python
# The canonical shape every tool definition follows
tool_schema = {
    "type": "object",
    "properties": {
        "<param_name>": {
            "type": "string | number | integer | boolean | array | object",
            "description": "what this parameter means to the model",
        },
    },
    "required": ["<param_name>", "..."],
}
```

## 2.2 `name`, `description`, `parameters` Fields

```python
tool_definition = {
    "type": "function",
    "function": {
        "name": "search_orders",                 # snake_case, unique, no spaces
        "description": (                          # this is the MODEL's only clue
            "Search the customer's order history by date range and status. "
            "Use this when the user asks about past orders, tracking, or order status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "ISO 8601 date, e.g. 2026-01-01"},
                "status": {"type": "string", "enum": ["pending", "shipped", "delivered", "cancelled"]},
            },
            "required": ["start_date"],
        },
    },
}
```

**Why it matters:** the model chooses which tool to call, and how to fill
its arguments, based *entirely* on `name` + `description` + the parameter
descriptions — there's no other signal. A vague description is the #1
cause of the model picking the wrong tool or leaving required fields empty.

## 2.3 Writing Effective Tool Descriptions

```python
# BAD — too vague, no guidance on when to use it
{"name": "search", "description": "Searches for things."}

# GOOD — states purpose, WHEN to use it, and what NOT to use it for
{
    "name": "search_internal_wiki",
    "description": (
        "Search the internal company wiki for policy, process, and HR documents. "
        "Use this for questions about company policy, benefits, or internal procedures. "
        "Do NOT use this for general knowledge questions or current events — "
        "use search_web instead for those."
    ),
}
```

**Why it matters:** treat descriptions like docstrings written for a very
literal junior engineer who has never seen your codebase — spell out
purpose, trigger conditions, and disambiguation from similar tools.

## 2.4 Required vs Optional Parameters

```python
{
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query text."},
        "max_results": {
            "type": "integer",
            "description": "Max number of results to return.",
            "default": 5,   # only advisory to the model -- YOUR code must still apply the default
        },
    },
    "required": ["query"],   # max_results is optional
}

# Your executor must handle the optional param actually being absent:
def search(query: str, max_results: int = 5):
    ...
```

**Why it matters:** `"required"` only affects when the model is *nudged* to
fill a field before calling the tool — it does not guarantee the field will
be present (some providers still occasionally omit required fields). Always
validate/default in your execution code, never assume the schema is
enforced end-to-end.

## 2.5 Enum Types for Constrained Inputs

```python
{
    "name": "set_ticket_priority",
    "description": "Update the priority level of a support ticket.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"],  # model can ONLY pick one of these
            },
        },
        "required": ["ticket_id", "priority"],
    },
}
```

**Why it matters:** enums are your strongest lever against garbage input —
"high priority please" reliably maps to `"high"` instead of the model
inventing `"very important"` or `"Priority 1"`. Use enums anywhere the
downstream system expects a fixed set of values.

## 2.6 Nested Object Schemas

```python
{
    "name": "create_shipping_label",
    "description": "Create a shipping label for an order.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "recipient": {                      # nested object
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "type": "object",       # nested inside nested
                        "properties": {
                            "line1": {"type": "string"},
                            "city": {"type": "string"},
                            "postal_code": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["line1", "city", "postal_code", "country"],
                    },
                },
                "required": ["name", "address"],
            },
        },
        "required": ["order_id", "recipient"],
    },
}
```

**Why it matters:** nesting is fully supported and often clearer than
flattening (`recipient_name`, `recipient_address_line1`, ...) — but deeper
than 2-3 levels tends to increase the model's error rate. If you're
nesting 4+ levels deep, consider splitting into two tool calls instead.

## 2.7 Arrays as Parameter Types

```python
{
    "name": "bulk_tag_tickets",
    "description": "Apply the same tag to multiple support tickets at once.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticket_ids": {
                "type": "array",
                "items": {"type": "string"},          # array of strings
                "description": "List of ticket IDs to tag.",
            },
            "line_items": {
                "type": "array",
                "items": {                             # array of objects
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        "quantity": {"type": "integer"},
                    },
                    "required": ["sku", "quantity"],
                },
            },
        },
        "required": ["ticket_ids"],
    },
}
```

**Why it matters:** arrays let one tool call replace N separate calls
(e.g. tag 10 tickets in one call instead of 10 round-trips) — a real
latency/cost win when you're prompting the model to batch its own actions.

---

# MODULE 3 — Executing Tool Calls

## 3.1 Detecting When the Model Calls a Tool

```python
# OpenAI
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
msg = response.choices[0].message
model_wants_tool = bool(msg.tool_calls)

# Anthropic
response = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024, tools=tools, messages=messages)
model_wants_tool = response.stop_reason == "tool_use"

# Gemini
response = chat.send_message(user_input)
part = response.candidates[0].content.parts[0]
model_wants_tool = hasattr(part, "function_call") and part.function_call.name != ""
```

**Why it matters:** this check is your branch point — if false, you show
`msg.content` directly to the user; if true, you enter the execution loop
instead of treating the response as final text.

## 3.2 Extracting Tool Name and Arguments

```python
import json

def extract_calls_openai(msg):
    return [
        {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
        for tc in msg.tool_calls
    ]

def extract_calls_anthropic(response):
    return [
        {"id": b.id, "name": b.name, "arguments": b.input}
        for b in response.content if b.type == "tool_use"
    ]
```

**Why it matters:** OpenAI's arguments arrive as a **JSON string** you must
`json.loads()` yourself; Anthropic's `input` is already a parsed dict. A
classic first-timer bug is passing OpenAI's raw string straight into a
function call and getting a `TypeError`.

## 3.3 Executing the Tool in Code

```python
TOOL_REGISTRY = {}

def tool(name):
    """Decorator to register a Python function under a tool name."""
    def wrapper(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return wrapper

@tool("get_weather")
def get_weather(city: str, unit: str = "celsius") -> dict:
    # real implementation would call a weather API
    return {"city": city, "temp": 31, "unit": unit, "conditions": "partly cloudy"}

def execute_tool_call(call: dict):
    fn = TOOL_REGISTRY.get(call["name"])
    if fn is None:
        return {"error": f"Unknown tool: {call['name']}"}
    try:
        return fn(**call["arguments"])
    except TypeError as e:
        return {"error": f"Invalid arguments for {call['name']}: {e}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}
```

**Why it matters:** a registry pattern (dict of name → function) keeps
adding new tools to a one-line `@tool(...)` decoration instead of a
growing if/elif chain — scales cleanly past 5-10 tools.

## 3.4 Returning Tool Results to the Model

```python
# OpenAI: append the assistant's tool-call message, then one "tool" message per call
messages.append(msg)  # the assistant message containing tool_calls
for call in extract_calls_openai(msg):
    result = execute_tool_call(call)
    messages.append({
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps(result),   # must be a string
    })

# Anthropic: append assistant content, then a user message with tool_result blocks
messages.append({"role": "assistant", "content": response.content})
tool_results = []
for call in extract_calls_anthropic(response):
    result = execute_tool_call(call)
    tool_results.append({"type": "tool_result", "tool_use_id": call["id"], "content": json.dumps(result)})
messages.append({"role": "user", "content": tool_results})
```

**Why it matters:** the model needs the **matching id** (`tool_call_id` /
`tool_use_id`) to know which result answers which call — this matters a lot
once you're doing parallel calls (Module 4), where mismatched ids silently
corrupt the conversation.

## 3.5 Completing the Conversation with the Final Response

```python
final_response = client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, tools=tools,
)
final_msg = final_response.choices[0].message

if final_msg.tool_calls:
    # model wants to call ANOTHER tool -- loop again (see 3.6)
    ...
else:
    print(final_msg.content)  # done -- natural language answer to show the user
```

**Why it matters:** don't assume one tool call = one round trip. The model
may need the result of tool A to decide whether to call tool B — always
re-check `tool_calls` on every response, in a loop, not just once.

## 3.6 Multi-Turn Tool Use

```python
def agent_loop(user_message: str, tools, max_turns: int = 6) -> str:
    messages = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content  # final answer reached

        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool_call({"name": tc.function.name, "arguments": args})
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": json.dumps(result),
            })

    return "I wasn't able to complete this within the allowed number of steps."

# e.g. "What's the weather in the city where our biggest customer is headquartered?"
# turn 1: calls get_customer_hq_city  -> "Chennai"
# turn 2: calls get_weather(city="Chennai") -> final answer
print(agent_loop("What's the weather where our biggest customer is based?", tools))
```

**Why it matters:** `max_turns` is a non-negotiable safety valve — without
it, a model stuck in a bad reasoning loop (calling the same tool repeatedly
with slightly different args) will burn API budget indefinitely.

---

# MODULE 4 — Parallel Tool Calling

## 4.1 What Parallel Tool Calling Is

**Plain English:** instead of proposing one tool call per model turn, the
model can propose **several tool calls at once** in a single response (e.g.
"get weather for Chennai" AND "get weather for Mumbai" in the same turn),
because they don't depend on each other's results.

## 4.2 OpenAI Parallel Tool Calls Implementation

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Compare the weather in Chennai and Mumbai."}],
    tools=tools,
    parallel_tool_calls=True,  # default is True for models that support it
)

msg = response.choices[0].message
print(len(msg.tool_calls))  # likely 2 -- one call per city, same turn
for tc in msg.tool_calls:
    print(tc.function.name, tc.function.arguments)
```

**Why it matters:** without parallel calls, comparing N things costs N
separate model round-trips (each with its own latency); with parallel
calls, the model proposes all N in one round-trip, and you execute them
together.

## 4.3 Executing Multiple Tools Concurrently

```python
import asyncio
import concurrent.futures

def execute_calls_concurrently(calls: list[dict]) -> list[dict]:
    """Runs independent tool calls in a thread pool (fine for I/O-bound tools
    like HTTP requests; use asyncio directly if your tool functions are async)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(execute_tool_call, c): c for c in calls}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            call = futures[future]
            try:
                results[call["id"]] = future.result()
            except Exception as e:
                results[call["id"]] = {"error": str(e)}
    return results

# async version, if your tool implementations are already async (e.g. httpx.AsyncClient)
async def execute_calls_async(calls: list[dict]) -> dict:
    async def run_one(call):
        fn = ASYNC_TOOL_REGISTRY[call["name"]]
        return call["id"], await fn(**call["arguments"])

    pairs = await asyncio.gather(*(run_one(c) for c in calls))
    return dict(pairs)
```

**Why it matters:** the model proposing calls in parallel only helps if
*you* also execute them concurrently — running them one-by-one in a `for`
loop throws away the latency win. Match your execution strategy to your
tools: threads for blocking I/O, `asyncio` if your libraries support it
natively.

## 4.4 Aggregating Results from Parallel Calls

```python
def run_parallel_turn(messages, tools):
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content, messages

    calls = [
        {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
        for tc in msg.tool_calls
    ]
    results = execute_calls_concurrently(calls)

    messages.append(msg)
    for call in calls:
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(results[call["id"]]),  # matched back by id
        })

    final = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    return final.choices[0].message.content, messages

# "Compare weather in Chennai and Mumbai" -> both weather results feed into ONE
# synthesized answer, e.g. "Chennai is hotter and more humid than Mumbai today."
```

**Why it matters:** the model does the aggregation/synthesis for you in the
final turn — your job is just to make sure every result is correctly
matched to its `tool_call_id` before sending them all back together.

---

# MODULE 5 — Building Custom Tools

Each tool below follows the same shape: **schema** (what the model sees) +
**implementation** (what your code does) + a one-line note on production
concerns.

## 5.1 Web Search Tool — Tavily, SerpAPI, DuckDuckGo

```python
# --- Tavily (built for LLM agents; returns clean, citable snippets) ---
from tavily import TavilyClient
tavily = TavilyClient(api_key="YOUR_TAVILY_KEY")

def web_search_tavily(query: str, max_results: int = 5) -> list[dict]:
    resp = tavily.search(query=query, max_results=max_results)
    return [{"title": r["title"], "url": r["url"], "snippet": r["content"]} for r in resp["results"]]

# --- SerpAPI (scrapes real Google results; good for shopping/local/news) ---
import requests

def web_search_serpapi(query: str, num: int = 5) -> list[dict]:
    resp = requests.get("https://serpapi.com/search", params={
        "q": query, "num": num, "api_key": "YOUR_SERPAPI_KEY",
    })
    data = resp.json()
    return [{"title": r["title"], "url": r["link"], "snippet": r.get("snippet", "")}
             for r in data.get("organic_results", [])]

# --- DuckDuckGo (free, no API key, rate-limited -- good for prototyping) ---
from duckduckgo_search import DDGS

def web_search_ddg(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        return [{"title": r["title"], "url": r["href"], "snippet": r["body"]}
                 for r in ddgs.text(query, max_results=max_results)]

web_search_tool_schema = {
    "name": "web_search",
    "description": "Search the live web for current information not in the knowledge base.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
```

**Production note:** DuckDuckGo has no official API and can throttle you —
fine for a prototype, swap to Tavily/SerpAPI (paid, SLA-backed) before
production traffic.

## 5.2 Calculator / Code Execution Tool

```python
import ast
import operator

# Safe arithmetic-only evaluator -- do NOT use eval() directly on model output
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Disallowed expression")

def calculator(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)

# For general code execution, use a SANDBOXED runner (subprocess with resource
# limits, or a service like E2B/Modal/Daytona) -- never exec()/eval() raw
# model-generated code in your main process.
```

**Production note:** the difference between "calculator" (safe, restricted
grammar) and "code execution" (arbitrary Python) is a security cliff —
arbitrary code execution needs a real sandbox (container, gVisor, or a
managed service), not just a try/except.

## 5.3 Database Query Tool

```python
import sqlite3

ALLOWED_TABLES = {"orders", "customers", "products"}  # allowlist

def query_database(sql: str) -> list[dict]:
    sql_lower = sql.strip().lower()
    if not sql_lower.startswith("select"):
        return {"error": "Only SELECT statements are allowed."}
    if not any(table in sql_lower for table in ALLOWED_TABLES):
        return {"error": "Query does not reference an allowed table."}

    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchmany(100)]  # cap result size
        return rows
    finally:
        conn.close()

db_tool_schema = {
    "name": "query_database",
    "description": (
        "Run a read-only SQL SELECT query against the orders/customers/products tables. "
        "Never generate INSERT, UPDATE, DELETE, or DDL statements."
    ),
    "parameters": {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
        "required": ["sql"],
    },
}
```

**Production note:** letting a model write raw SQL is high-risk even with
the SELECT-only check above (it doesn't stop `SELECT * FROM users; DROP
TABLE ...` style tricks in some engines) — prefer parameterized "query
templates" the model fills in (e.g. `search_orders(customer_id, status)`)
over free-form SQL generation wherever possible.

## 5.4 REST API Caller Tool

```python
import requests

ALLOWED_HOSTS = {"api.yourcompany.com"}

def call_rest_api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"https://api.yourcompany.com{path}"
    if not url.startswith(tuple(f"https://{h}" for h in ALLOWED_HOSTS)):
        return {"error": "Host not allowed."}

    resp = requests.request(
        method=method.upper(), url=url, json=body, timeout=10,
        headers={"Authorization": f"Bearer {INTERNAL_SERVICE_TOKEN}"},
    )
    return {"status_code": resp.status_code, "body": resp.json() if resp.ok else resp.text}

rest_tool_schema = {
    "name": "call_internal_api",
    "description": "Call an internal company REST API endpoint.",
    "parameters": {
        "type": "object",
        "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
            "path": {"type": "string", "description": "e.g. /v1/orders/12345"},
            "body": {"type": "object", "description": "JSON body for POST/PUT requests."},
        },
        "required": ["method", "path"],
    },
}
```

**Production note:** host allowlisting + a scoped service token (not the
user's own credentials) prevents this tool from becoming an open SSRF
proxy if the model is ever tricked (via prompt injection in retrieved
content) into calling an unexpected URL.

## 5.5 File Reader / Writer Tool

```python
from pathlib import Path

SANDBOX_ROOT = Path("/safe/workspace").resolve()

def _resolve_safe(path_str: str) -> Path:
    p = (SANDBOX_ROOT / path_str).resolve()
    if SANDBOX_ROOT not in p.parents and p != SANDBOX_ROOT:
        raise ValueError("Path escapes sandbox root.")
    return p

def read_file(path: str) -> str:
    p = _resolve_safe(path)
    return p.read_text(encoding="utf-8")[:20_000]  # cap size returned to the model

def write_file(path: str, content: str) -> dict:
    p = _resolve_safe(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"written_path": str(p.relative_to(SANDBOX_ROOT)), "bytes": len(content)}
```

**Production note:** `_resolve_safe` blocks `../../etc/passwd`-style path
traversal — this check is not optional the moment file tools are exposed
to any model-generated path string.

## 5.6 Email Sender Tool

```python
import smtplib
from email.message import EmailMessage

def send_email(to: str, subject: str, body: str) -> dict:
    msg = EmailMessage()
    msg["From"] = "assistant@yourcompany.com"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP("smtp.yourcompany.com", 587) as server:
        server.starttls()
        server.login("assistant@yourcompany.com", "APP_PASSWORD")
        server.send_message(msg)
    return {"status": "sent", "to": to}

email_tool_schema = {
    "name": "send_email",
    "description": (
        "Send an email on the user's behalf. ALWAYS show the user the drafted "
        "subject and body and get explicit confirmation before calling this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address."},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
}
```

**Production note:** this is a **write/side-effect tool** — pair it with a
human-confirmation step (Module 6.5 covers access control; add an explicit
"are you sure?" turn before executing) so the model can't send email
autonomously from a single ambiguous instruction.

## 5.7 Calendar Event Creator Tool

```python
from datetime import datetime

def create_calendar_event(title: str, start_iso: str, end_iso: str, attendees: list[str] = None) -> dict:
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    if end <= start:
        return {"error": "end time must be after start time"}

    # real implementation: call Google Calendar API / Microsoft Graph here
    event_id = "evt_12345"
    return {"event_id": event_id, "title": title, "start": start_iso, "end": end_iso,
            "attendees": attendees or []}

calendar_tool_schema = {
    "name": "create_calendar_event",
    "description": "Create a calendar event. Requires ISO 8601 datetimes.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_iso": {"type": "string", "description": "e.g. 2026-08-10T14:00:00"},
            "end_iso": {"type": "string"},
            "attendees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "start_iso", "end_iso"],
    },
}
```

**Production note:** validate `end > start` and reasonable duration bounds
in code — don't trust the model to always compute times correctly,
especially across timezones it has to infer from natural language.

## 5.8 Browser / Web Scraper Tool

```python
import requests
from bs4 import BeautifulSoup

def scrape_page(url: str) -> dict:
    resp = requests.get(url, timeout=10, headers={"User-Agent": "MyAgent/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return {"url": url, "title": soup.title.string if soup.title else "", "text": text[:10_000]}

scrape_tool_schema = {
    "name": "scrape_web_page",
    "description": "Fetch and extract the readable text content of a specific URL.",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Full URL including https://"}},
        "required": ["url"],
    },
}
```

**Production note:** for JS-heavy sites, swap `requests` for a headless
browser (Playwright/Selenium); either way, respect `robots.txt` and rate
limits, and treat scraped text as **untrusted input** — never let it be
interpreted as new instructions to the model (prompt-injection risk).

---

# MODULE 6 — Tool Use Patterns

## 6.1 Tool Selection via Descriptions

```python
# When tools overlap in purpose, explicitly disambiguate in each description.
tools = [
    {"name": "search_internal_docs", "description":
        "Search internal company documents (policies, HR, internal wiki). "
        "Use for company-specific questions. NOT for general knowledge."},
    {"name": "search_web", "description":
        "Search the public internet for current events or general knowledge. "
        "NOT for company-internal information -- use search_internal_docs for that."},
]
# Cross-referencing each tool's description against its sibling tools ("NOT for X,
# use Y instead") measurably reduces wrong-tool selection in practice.
```

**Why it matters:** as your tool count grows past ~8-10, selection accuracy
degrades unless descriptions actively disambiguate against each other, not
just describe themselves in isolation.

## 6.2 Tool Chaining

```python
# Chaining happens naturally in the multi-turn loop (3.6) -- the model decides
# the sequence itself. You can also chain explicitly for deterministic pipelines:

def lookup_and_notify(customer_email: str, order_id: str):
    order = execute_tool_call({"name": "query_database",
                                "arguments": {"sql": f"SELECT * FROM orders WHERE id='{order_id}'"}})
    if "error" in order:
        return order

    summary = f"Your order {order_id} status: {order[0]['status']}"
    return execute_tool_call({"name": "send_email",
                               "arguments": {"to": customer_email, "subject": "Order update", "body": summary}})
```

**Why it matters:** model-driven chaining (letting the LLM decide the
sequence turn-by-turn) is flexible but non-deterministic; code-driven
chaining (like above) is faster, cheaper, and predictable for workflows you
already know the shape of. Reserve model-driven chaining for genuinely
open-ended tasks.

## 6.3 Error Handling When a Tool Fails

```python
def execute_tool_call_safe(call: dict) -> dict:
    fn = TOOL_REGISTRY.get(call["name"])
    if fn is None:
        return {"error": f"Tool '{call['name']}' does not exist.", "recoverable": False}
    try:
        return {"result": fn(**call["arguments"])}
    except TypeError as e:
        return {"error": f"Bad arguments: {e}", "recoverable": True}   # model can retry with fixed args
    except TimeoutError:
        return {"error": "Tool timed out.", "recoverable": True}
    except Exception as e:
        return {"error": f"Unexpected failure: {type(e).__name__}: {e}", "recoverable": False}

# Send the error back to the model AS A TOOL RESULT (not as an exception) --
# this lets the model see the failure and decide whether to retry, try a
# different tool, or apologize to the user instead of crashing your app.
```

**Why it matters:** feeding structured errors back into the conversation
(rather than crashing or silently retrying forever) lets the model make
the retry/fallback decision — often better than hardcoding retry logic for
every possible failure mode.

## 6.4 Tool Result Formatting

```python
# BAD: dumping a huge raw object wastes context and confuses the model
def get_orders_bad(customer_id: str):
    return db.query(f"SELECT * FROM orders WHERE customer_id='{customer_id}'")  # could be 500 rows, 40 columns

# GOOD: summarize / truncate / project to what the model actually needs
def get_orders_good(customer_id: str, limit: int = 10):
    rows = db.query(
        "SELECT id, status, total, created_at FROM orders WHERE customer_id=? "
        "ORDER BY created_at DESC LIMIT ?", (customer_id, limit)
    )
    return {
        "count_returned": len(rows),
        "orders": rows,
        "note": "Showing most recent orders only." if len(rows) == limit else None,
    }
```

**Why it matters:** tool results consume context-window tokens just like
everything else — an unfiltered 500-row dump both costs money and buries
the signal the model actually needs. Project, paginate, and summarize
before returning.

## 6.5 Role-Based Tool Access Control

```python
TOOL_PERMISSIONS = {
    "search_internal_docs": {"support_agent", "admin"},
    "query_database":        {"admin"},
    "send_email":            {"support_agent", "admin"},
    "issue_refund":          {"admin"},
}

def get_tools_for_role(role: str, all_tool_schemas: list[dict]) -> list[dict]:
    """Only expose tool DEFINITIONS the role is allowed to use -- the model
    can't call what it never sees."""
    allowed_names = {name for name, roles in TOOL_PERMISSIONS.items() if role in roles}
    return [t for t in all_tool_schemas if t["name"] in allowed_names]

def execute_tool_call_with_rbac(call: dict, user_role: str) -> dict:
    allowed_roles = TOOL_PERMISSIONS.get(call["name"], set())
    if user_role not in allowed_roles:
        return {"error": f"Role '{user_role}' is not permitted to use '{call['name']}'."}
    return execute_tool_call_safe(call)
```

**Why it matters:** enforce access control **twice** — once by filtering
which tool schemas you even send to the model (defense in depth, smaller
attack surface) and again in the executor (in case a stale schema list, a
prompt injection, or a bug lets an unauthorized call slip through).

---

## Capstone project

Build a small internal-support agent with:

1. Three tools minimum: `search_internal_docs` (reuse your RAG capstone!),
   `query_database` (read-only, mock SQLite orders table), `send_email`
   (mock — just log instead of actually sending).
2. A multi-turn loop (3.6) with `max_turns` and structured error handling (6.3).
3. At least one scenario requiring **tool chaining**: e.g. "look up this
   customer's last order, then email them a status update."
4. **Role-based access control** (6.5): an `"agent"` role that can search
   docs and send email, and an `"admin"` role that can additionally query
   the database directly.
5. A short write-up of one deliberate failure case you tested (bad
   arguments, tool timeout, unauthorized tool call) and how the agent
   handled it.

Want this turned into a runnable notebook scaffold like the RAG capstone?
