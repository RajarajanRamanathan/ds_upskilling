import os
import time
import concurrent.futures
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# Mock data so the project runs with no external API keys besides the LLM's.
_MOCK_WEATHER = {
    "paris": {"temp_c": 14, "condition": "light rain"},
    "tokyo": {"temp_c": 22, "condition": "clear"},
    "chennai": {"temp_c": 33, "condition": "humid, sunny"},
}
_MOCK_FX_RATES_PER_USD = {"eur": 0.92, "jpy": 151.0, "inr": 83.5}

@tool
def get_weather(city: str) -> dict:
    """Get current weather (temp in Celsius, condition) for a given city name."""
    key = city.strip().lower()
    if key not in _MOCK_WEATHER:
        raise ValueError(f"No weather data for '{city}'")
    return _MOCK_WEATHER[key]

@tool
def convert_currency(amount: float, to_currency: str) -> dict:
    """Convert a USD amount to another currency. to_currency is a 3-letter code, e.g. 'eur', 'jpy', 'inr'."""
    key = to_currency.strip().lower()
    if key not in _MOCK_FX_RATES_PER_USD:
        raise ValueError(f"No FX rate for '{to_currency}'")
    converted = round(amount * _MOCK_FX_RATES_PER_USD[key], 2)
    return {"amount_usd": amount, "converted": converted, "currency": key.upper()}

@tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression, e.g. '500 * 0.92'. Only numbers and + - * / ( )."""
    allowed = set("0123456789.+-*/() ")
    if not set(expression) <= allowed:
        raise ValueError("Expression contains disallowed characters")
    return eval(expression, {"__builtins__": {}}, {})

TOOLS = [get_weather, convert_currency, calculate]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps_taken: int        
    trace: list

MAX_ITERATIONS = 6            
TOOL_TIMEOUT_SECONDS = 10 

def _build_llm():
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are a trip planning assistant. You have tools to check weather and "
    "convert currency. Use tools as needed, then give a short final answer "
    "covering: expected weather, and what the traveller's budget converts to "
    "locally. Do not guess values you can look up with a tool."
)

def call_llm(state: AgentState) -> dict:
    """Think step: ask the LLM brain what to do next."""
    llm = _build_llm()
    response = llm.invoke(state["messages"])
    trace_entry = f"[step {state['steps_taken'] + 1}] LLM response: " + (
        f"tool_calls={[tc['name'] for tc in response.tool_calls]}"
        if response.tool_calls else f"final_answer='{response.content[:80]}...'"
    )
    return {
        "messages": [response],
        "steps_taken": state["steps_taken"] + 1,
        "trace": state["trace"] + [trace_entry],
    }

def _safe_tool_node(state: AgentState) -> dict:
    """Act + Observe step, wrapped with error handling (Section 7):
    catches tool exceptions and timeouts, turning them into observations
    instead of crashing the graph."""
    last_message = state["messages"][-1]
    tool_map = {t.name: t for t in TOOLS}
    results = []
    trace_additions = []

    for call in last_message.tool_calls:
        name, args, call_id = call["name"], call["args"], call["id"]
        tool_fn = tool_map.get(name)

        try:
            if tool_fn is None:
                raise ValueError(f"Unknown tool '{name}'. Available: {list(tool_map)}")

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(tool_fn.invoke, args)
                output = future.result(timeout=TOOL_TIMEOUT_SECONDS)
            trace_additions.append(f"  -> {name}({args}) = {output}")

        except concurrent.futures.TimeoutError:
            output = f"Error: tool '{name}' timed out after {TOOL_TIMEOUT_SECONDS}s"
            trace_additions.append(f"  -> {name}({args}) TIMEOUT")
        except Exception as e:
            output = f"Error: tool '{name}' failed: {e}"
            trace_additions.append(f"  -> {name}({args}) ERROR: {e}")

        from langchain_core.messages import ToolMessage
        results.append(ToolMessage(content=str(output), tool_call_id=call_id))

    return {"messages": results, "trace": state["trace"] + trace_additions}

tool_node = _safe_tool_node

def should_continue(state: AgentState) -> str:
    if state["steps_taken"] >= MAX_ITERATIONS:
        return END  # hard cap hit — stop even if the model wants to keep going
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()

def run(user_goal: str):
    app = build_graph()
    initial_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_goal)],
        "steps_taken": 0,
        "trace": [],
    }
    result = app.invoke(initial_state)

    print("=== TRAJECTORY (for evaluation) ===")
    for line in result["trace"]:
        print(line)
    print(f"\nTotal LLM steps: {result['steps_taken']} (cap: {MAX_ITERATIONS})")

    print("\n=== FINAL ANSWER ===")
    print(result["messages"][-1].content)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this against a real LLM, e.g.:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  python trip_assistant_agent.py")
    else:
        run("I'm travelling to Paris next week with a budget of 500 USD. "
            "What should I expect, and what's my budget worth there?")