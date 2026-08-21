"""
Mini Project #2b: Multi-Agent Trip Planning System — No API Key Needed
=========================================================================

Same graph, same concepts as trip_multi_agent.py, but every LLM call is
replaced with a small deterministic "mock brain" function. This lets you
run the WHOLE thing end-to-end right now with no OPENAI_API_KEY, and
step through exactly what each agent role produces.

Trade-off worth naming explicitly: real LLMs (Planner reading a vague goal,
Writer producing prose, Critic judging quality) are being stood in for by
hand-written logic here. That's fine for learning the *graph mechanics*
(dependency execution, retries, interrupts, audit trail) but the deterministic
planner/writer/critic below are far less capable than a real LLM would be —
this file is a mechanics sandbox, not a production planner.

To see the same concepts with a real LLM, use trip_multi_agent.py instead.

Run with:
    python trip_multi_agent_no_llm.py
"""

import re
import concurrent.futures
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Mock data — same as prior projects.
# ---------------------------------------------------------------------------

_MOCK_WEATHER = {
    "paris": {"temp_c": 14, "condition": "light rain"},
    "tokyo": {"temp_c": 22, "condition": "clear"},
}
_MOCK_FX_RATES_PER_USD = {"eur": 0.92, "jpy": 151.0}
_CITY_CURRENCY = {"paris": "eur", "tokyo": "jpy"}


def _weather_researcher(city: str) -> dict:
    key = city.strip().lower()
    if key not in _MOCK_WEATHER:                 # guardrail (Sec 5)
        raise ValueError(f"No weather data for '{city}'")
    return _MOCK_WEATHER[key]


def _currency_researcher(city: str, budget_usd: float) -> dict:
    key = city.strip().lower()
    if key not in _CITY_CURRENCY:
        raise ValueError(f"No currency mapping for '{city}'")
    currency = _CITY_CURRENCY[key]
    return {"currency": currency.upper(), "converted": round(budget_usd * _MOCK_FX_RATES_PER_USD[currency], 2)}


_AGENT_FUNCS = {"weather_researcher": _weather_researcher, "currency_researcher": _currency_researcher}


# ---------------------------------------------------------------------------
# STATE SCHEMA — the blackboard (Section 3), unchanged from the LLM version.
# ---------------------------------------------------------------------------

class Task(BaseModel):
    name: str
    agent: Literal["weather_researcher", "currency_researcher"]
    args: dict
    depends_on: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    goal: str
    plan: list[dict]
    results: dict
    draft: str
    critic_retries: int
    orchestrator_steps: int
    trace: list
    human_decision: str
    final_answer: str


MAX_CRITIC_RETRIES = 2
MAX_ORCHESTRATOR_STEPS = 8
TASK_TIMEOUT_SECONDS = 10
TASK_RETRY_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# NODE: Planner — deterministic mock brain. Extracts known city names and a
# dollar amount from the goal text with regex/keyword matching, instead of
# an LLM interpreting free-form intent. This is the "task specialization"
# and "task decomposition" logic (Sections 1 & 4), just non-LLM-driven.
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    goal_lower = state["goal"].lower()
    known_cities = set(_MOCK_WEATHER) | set(_CITY_CURRENCY)
    cities_found = [c for c in known_cities if c in goal_lower]

    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:usd|dollars|\$)?", goal_lower)
    budget = float(budget_match.group(1)) if budget_match else 0.0

    tasks: list[Task] = []
    for city in cities_found:
        tasks.append(Task(name=f"weather_{city}", agent="weather_researcher", args={"city": city}))
        tasks.append(Task(name=f"currency_{city}", agent="currency_researcher",
                           args={"city": city, "budget_usd": budget}))

    trace = state["trace"] + [
        f"[planner] (mock) parsed cities={cities_found}, budget=${budget}",
        f"[planner] (mock) produced {len(tasks)} tasks",
    ]
    return {
        "plan": [t.model_dump() for t in tasks],
        "trace": trace,
        "orchestrator_steps": state["orchestrator_steps"] + 1,
    }


# ---------------------------------------------------------------------------
# NODE: Executor stage — identical logic to the LLM version. Nothing here
# needs an LLM at all; dependency-wave execution, retries, and timeouts are
# plain orchestration code in both versions.
# ---------------------------------------------------------------------------

def execute_plan_node(state: AgentState) -> dict:
    tasks = {t["name"]: t for t in state["plan"]}
    results = dict(state["results"])
    trace = list(state["trace"])
    remaining = set(tasks)

    while remaining:
        ready = [name for name in remaining if all(d in results for d in tasks[name]["depends_on"])]
        if not ready:
            trace.append("[executor] deadlock: remaining tasks have unmet dependencies")
            break

        trace.append(f"[executor] running wave in parallel: {ready}")

        def run_one(name):
            task = tasks[name]
            fn = _AGENT_FUNCS[task["agent"]]
            last_err = None
            for attempt in range(1, TASK_RETRY_ATTEMPTS + 1):
                try:
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(fn, **task["args"])
                        return name, future.result(timeout=TASK_TIMEOUT_SECONDS), None
                except concurrent.futures.TimeoutError:
                    last_err = f"timed out after {TASK_TIMEOUT_SECONDS}s (attempt {attempt})"
                except Exception as e:
                    last_err = f"{e} (attempt {attempt})"
            return name, None, last_err

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(ready), 1)) as pool:
            for name, output, err in pool.map(run_one, ready):
                if err:
                    results[name] = {"error": err}
                    trace.append(f"  -> {name} FAILED: {err}")
                else:
                    results[name] = output
                    trace.append(f"  -> {name} = {output}")
                remaining.discard(name)

    return {"results": results, "trace": trace, "orchestrator_steps": state["orchestrator_steps"] + 1}


# ---------------------------------------------------------------------------
# NODE: Writer — deterministic mock brain. Deliberately produces an
# INCOMPLETE first draft (weather only, omits currency) so the critic loop
# below actually has something real to catch and you can see a revision
# cycle happen (Section 5: reliability / revision loop), not just succeed
# silently on the first try.
# ---------------------------------------------------------------------------

def write_report_node(state: AgentState) -> dict:
    cities = sorted({name.split("_", 1)[1] for name in state["results"]})
    lines = [f"Trip report for: {', '.join(c.title() for c in cities)}", ""]

    for city in cities:
        weather = state["results"].get(f"weather_{city}", {})
        lines.append(f"{city.title()} weather: {weather}")
        if state["critic_retries"] >= 1:
            # Only included from the 2nd draft onward, in response to critic feedback.
            currency = state["results"].get(f"currency_{city}", {})
            lines.append(f"{city.title()} budget converts to: {currency}")

    draft = "\n".join(lines)
    trace = state["trace"] + [
        f"[writer] (mock) draft #{state['critic_retries'] + 1} produced "
        f"({'includes' if state['critic_retries'] >= 1 else 'omits'} currency info)"
    ]
    return {"draft": draft, "trace": trace, "orchestrator_steps": state["orchestrator_steps"] + 1}


# ---------------------------------------------------------------------------
# NODE: Critic — deterministic mock brain. Real, meaningful checks: does the
# draft mention every planned city, and does it cover currency conversion
# for each. This is a genuine (if simple) verification pass, not a fake one.
# ---------------------------------------------------------------------------

def critique_node(state: AgentState) -> dict:
    cities = sorted({name.split("_", 1)[1] for name in state["results"]})
    missing = [c for c in cities if f"{c.title()} budget converts to" not in state["draft"]]

    if missing:
        approved = False
        feedback = f"Draft is missing currency conversion for: {', '.join(missing)}"
    else:
        approved = True
        feedback = "Draft covers weather and currency for every planned city."

    trace = state["trace"] + [f"[critic] (mock) approved={approved}", f"[critic] feedback: {feedback}"]
    return {"trace": trace, "orchestrator_steps": state["orchestrator_steps"] + 1}


def route_after_critique(state: AgentState) -> str:
    if state["orchestrator_steps"] >= MAX_ORCHESTRATOR_STEPS:
        return "finalize"
    approved = state["trace"][-2].endswith("approved=True")
    if approved or state["critic_retries"] >= MAX_CRITIC_RETRIES:
        return "finalize"
    return "revise"


def bump_retry_node(state: AgentState) -> dict:
    return {"critic_retries": state["critic_retries"] + 1}


def finalize_node(state: AgentState) -> dict:
    decision = state.get("human_decision", "approved")
    trace = state["trace"] + [f"[human] decision recorded: {decision}"]
    final = "Task cancelled by human reviewer." if decision == "rejected" else state["draft"]
    return {"final_answer": final, "trace": trace}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("execute_plan", execute_plan_node)
    graph.add_node("write_report", write_report_node)
    graph.add_node("critique", critique_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "execute_plan")
    graph.add_edge("execute_plan", "write_report")
    graph.add_edge("write_report", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"revise": "bump_retry", "finalize": "finalize"})
    graph.add_edge("bump_retry", "write_report")

    checkpointer = MemorySaver()
    return graph.compile(interrupt_before=["finalize"], checkpointer=checkpointer)


def run(goal: str, human_decision: str = "approved"):
    app = build_graph()
    config = {"configurable": {"thread_id": "trip-plan-no-llm"}}

    initial_state = {
        "goal": goal, "plan": [], "results": {}, "draft": "",
        "critic_retries": 0, "orchestrator_steps": 0, "trace": [],
        "human_decision": "", "final_answer": "",
    }

    state = app.invoke(initial_state, config)
    print("=== PAUSED FOR HUMAN REVIEW (Section 6) ===")
    print("Draft:\n" + state["draft"])
    print("\nFull trace:")
    for line in state["trace"]:
        print(" ", line)

    app.update_state(config, {"human_decision": human_decision})
    final_state = app.invoke(None, config)

    print("\n=== FINAL ANSWER ===")
    print(final_state["final_answer"])
    print(f"\nOrchestrator steps used: {final_state['orchestrator_steps']} (cap: {MAX_ORCHESTRATOR_STEPS})")
    print(f"Critic revision cycles used: {final_state['critic_retries']} (cap: {MAX_CRITIC_RETRIES})")


if __name__ == "__main__":
    run("Planning a trip to Paris and Tokyo, budget 500 USD each. "
        "Give me expected weather and what my budget converts to locally, for both cities.")