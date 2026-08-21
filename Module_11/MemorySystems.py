"""
Mini Project #3: Trip Planner with Long-Term Memory — No API Key Needed
===========================================================================

Extends trip_multi_agent_no_llm.py with a long-term memory layer, covering
as much of the Memory Systems module as fits in one runnable example:

  Section 1 (Types):          semantic memory (profile_facts table) vs.
                               episodic memory (episodes table); short-term
                               memory is still just AgentState within a run
  Section 2 (Short-Term Mgmt): save_memory_node's summary IS a "summarize
                               old turns" step — this run's whole trace gets
                               compressed into one durable episodic sentence
  Section 3 (Vector Stores):  TF-IDF + cosine similarity stands in for an
                               embedding-based vector store (no API key
                               needed) — storing facts, retrieving relevant
                               ones at query time, and a simple consolidation
                               pass that drops near-duplicate facts
  Section 4 (External Stores): SQLite for structured long-term memory
                               (profile_facts, episodes tables), separate
                               from LangGraph's SqliteSaver checkpointer,
                               which durably persists SHORT-term graph state
                               (the interrupt/resume thread) across process
                               restarts — two different stores, two different
                               jobs, both covered here
  Section 5 (In Practice):    retrieved memory injected into the planner's
                               input (user profile + project/episode memory
                               pattern); run() simulates two separate
                               "sessions" (two process-level runs) to prove
                               memory actually persists and gets reused

Run with:
    python trip_memory_agent.py           # first session — teaches it facts
    python trip_memory_agent.py --again   # second session — reuses memory
"""

import re
import sys
import sqlite3
import concurrent.futures
from datetime import datetime, timezone
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


USER_ID = "rajarajan"
MEMORY_DB = "trip_memory.db"           # long-term semantic/episodic store (Section 4)
CHECKPOINT_DB = "trip_checkpoints.db"  # short-term durable graph state (Section 4)

_MOCK_WEATHER = {
    "paris": {"temp_c": 14, "condition": "light rain"},
    "tokyo": {"temp_c": 22, "condition": "clear"},
}
_MOCK_FX_RATES_PER_USD = {"eur": 0.92, "jpy": 151.0}
_CITY_CURRENCY = {"paris": "eur", "tokyo": "jpy"}


def _weather_researcher(city: str) -> dict:
    key = city.strip().lower()
    if key not in _MOCK_WEATHER:
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
# LONG-TERM MEMORY STORE (Section 4: SQLite for structured memory)
# ---------------------------------------------------------------------------

def _db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS profile_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, fact TEXT, created_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, summary TEXT, created_at TEXT)""")
    return conn


def save_fact(user_id: str, fact: str):
    """Writes a semantic-memory fact, with a simple consolidation check
    (Section 3): skip if a near-duplicate already exists for this user."""
    conn = _db()
    existing = [r[0] for r in conn.execute("SELECT fact FROM profile_facts WHERE user_id=?", (user_id,))]
    if existing:
        vec = TfidfVectorizer().fit(existing + [fact])
        sims = cosine_similarity(vec.transform([fact]), vec.transform(existing))[0]
        if sims.max() > 0.8:                      # near-duplicate -> consolidate by skipping
            conn.close()
            return False
    conn.execute("INSERT INTO profile_facts (user_id, fact, created_at) VALUES (?, ?, ?)",
                 (user_id, fact, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    return True


def save_episode(user_id: str, summary: str):
    """Writes an episodic-memory record (Section 1) — this session, compressed
    to one sentence, the same move as short-term summarization (Section 2)
    but persisted long-term instead of just replacing old messages in-context."""
    conn = _db()
    conn.execute("INSERT INTO episodes (user_id, summary, created_at) VALUES (?, ?, ?)",
                 (user_id, summary, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def retrieve_memory(user_id: str, query: str, k: int = 3) -> list[str]:
    """Section 3: retrieval at query time. TF-IDF + cosine similarity here
    plays the role a real embedding model + vector store would play — same
    retrieval concept (embed query, rank stored items by similarity), just
    without calling an external embeddings API."""
    conn = _db()
    facts = [r[0] for r in conn.execute("SELECT fact FROM profile_facts WHERE user_id=?", (user_id,))]
    episodes = [r[0] for r in conn.execute("SELECT summary FROM episodes WHERE user_id=?", (user_id,))]
    conn.close()

    corpus = facts + episodes
    if not corpus:
        return []

    vec = TfidfVectorizer().fit(corpus + [query])
    sims = cosine_similarity(vec.transform([query]), vec.transform(corpus))[0]
    ranked = sorted(zip(corpus, sims), key=lambda x: -x[1])
    return [text for text, score in ranked[:k] if score > 0.05]  # drop irrelevant matches


# ---------------------------------------------------------------------------
# STATE SCHEMA
# ---------------------------------------------------------------------------

class Task(BaseModel):
    name: str
    agent: Literal["weather_researcher", "currency_researcher"]
    args: dict
    depends_on: list[str] = Field(default_factory=list)


class AgentState(TypedDict):
    goal: str
    memory_context: list[str]   # Section 5: retrieved memory injected before planning
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
# NODE: retrieve_memory — Section 5, runs BEFORE planning so the planner can
# use what's remembered about this user/project, same pattern as injecting
# retrieved memory into a system prompt.
# ---------------------------------------------------------------------------

def retrieve_memory_node(state: AgentState) -> dict:
    memories = retrieve_memory(USER_ID, state["goal"])
    trace = state["trace"] + [f"[memory] retrieved {len(memories)} relevant item(s): {memories}"]
    return {"memory_context": memories, "trace": trace}


# ---------------------------------------------------------------------------
# NODE: Planner — now falls back to remembered cities/budget when the goal
# is vague, instead of only parsing the current goal text. This is the part
# that makes memory actually change agent behavior, not just get logged.
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    goal_lower = state["goal"].lower()
    known_cities = set(_MOCK_WEATHER) | set(_CITY_CURRENCY)
    cities_found = [c for c in known_cities if c in goal_lower]

    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:usd|dollars|\$)?", goal_lower)
    budget = float(budget_match.group(1)) if budget_match else None

    used_memory = False
    if not cities_found or budget is None:
        memory_text = " ".join(state["memory_context"]).lower()
        if not cities_found:
            cities_found = [c for c in known_cities if c in memory_text]
            if cities_found:
                used_memory = True
        if budget is None:
            mem_budget_match = re.search(r"(\d+(?:\.\d+)?)\s*usd", memory_text)
            if mem_budget_match:
                budget = float(mem_budget_match.group(1))
                used_memory = True

    budget = budget or 0.0
    tasks: list[Task] = []
    for city in cities_found:
        tasks.append(Task(name=f"weather_{city}", agent="weather_researcher", args={"city": city}))
        tasks.append(Task(name=f"currency_{city}", agent="currency_researcher",
                           args={"city": city, "budget_usd": budget}))

    trace = state["trace"] + [
        f"[planner] cities={cities_found}, budget=${budget}"
        + (" (filled in from long-term memory)" if used_memory else ""),
        f"[planner] produced {len(tasks)} tasks",
    ]
    return {
        "plan": [t.model_dump() for t in tasks],
        "trace": trace,
        "orchestrator_steps": state["orchestrator_steps"] + 1,
    }


# ---------------------------------------------------------------------------
# NODE: Executor — unchanged from prior projects (dependency waves, retry,
# timeout, guardrails; Module 9 Sec 7 + Multi-Agent Sec 4/5).
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
# NODE: Writer / Critic — same deterministic mock-brain pattern as project 2b.
# ---------------------------------------------------------------------------

def write_report_node(state: AgentState) -> dict:
    cities = sorted({name.split("_", 1)[1] for name in state["results"]})
    lines = [f"Trip report for: {', '.join(c.title() for c in cities)}", ""]
    for city in cities:
        weather = state["results"].get(f"weather_{city}", {})
        lines.append(f"{city.title()} weather: {weather}")
        if state["critic_retries"] >= 1:
            currency = state["results"].get(f"currency_{city}", {})
            lines.append(f"{city.title()} budget converts to: {currency}")
    draft = "\n".join(lines)
    trace = state["trace"] + [f"[writer] draft #{state['critic_retries'] + 1} produced"]
    return {"draft": draft, "trace": trace, "orchestrator_steps": state["orchestrator_steps"] + 1}


def critique_node(state: AgentState) -> dict:
    cities = sorted({name.split("_", 1)[1] for name in state["results"]})
    missing = [c for c in cities if f"{c.title()} budget converts to" not in state["draft"]]
    approved = not missing
    feedback = "Draft is complete." if approved else f"Missing currency for: {', '.join(missing)}"
    trace = state["trace"] + [f"[critic] approved={approved}", f"[critic] feedback: {feedback}"]
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


# ---------------------------------------------------------------------------
# NODE: save_memory — Section 2 (summarize this session) + Section 4
# (write to durable structured store), runs AFTER finalize.
# ---------------------------------------------------------------------------

def save_memory_node(state: AgentState) -> dict:
    if state["final_answer"].startswith("Task cancelled"):
        return {"trace": state["trace"] + ["[memory] skipped saving (task was rejected)"]}

    cities = sorted({name.split("_", 1)[1] for name in state["results"]})
    budgets = [t["args"]["budget_usd"] for t in state["plan"] if t["agent"] == "currency_researcher"]
    budget = budgets[0] if budgets else 0

    saved_notes = []
    if cities:
        fact = f"User has planned trips to: {', '.join(c.title() for c in cities)}"
        if save_fact(USER_ID, fact):
            saved_notes.append(fact)
    if budget:
        fact = f"User's typical trip budget is around {budget:.0f} USD per city"
        if save_fact(USER_ID, fact):
            saved_notes.append(fact)

    episode = f"Planned a trip to {', '.join(c.title() for c in cities)} with {budget:.0f} USD budget; approved."
    save_episode(USER_ID, episode)

    trace = state["trace"] + [f"[memory] saved episode: {episode}", f"[memory] saved facts: {saved_notes or 'none (already known)'}"]
    return {"trace": trace}


# ---------------------------------------------------------------------------
# BUILD + COMPILE — durable checkpointer (Section 4) so the interrupt/resume
# survives even across separate process runs, not just MemorySaver's in-process-only persistence.
# ---------------------------------------------------------------------------

def build_graph(checkpointer):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_memory", retrieve_memory_node)
    graph.add_node("planner", planner_node)
    graph.add_node("execute_plan", execute_plan_node)
    graph.add_node("write_report", write_report_node)
    graph.add_node("critique", critique_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "planner")
    graph.add_edge("planner", "execute_plan")
    graph.add_edge("execute_plan", "write_report")
    graph.add_edge("write_report", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"revise": "bump_retry", "finalize": "finalize"})
    graph.add_edge("bump_retry", "write_report")
    graph.add_edge("finalize", "save_memory")

    return graph.compile(interrupt_before=["finalize"], checkpointer=checkpointer)


def run(goal: str, thread_id: str):
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "goal": goal, "memory_context": [], "plan": [], "results": {}, "draft": "",
            "critic_retries": 0, "orchestrator_steps": 0, "trace": [],
            "human_decision": "", "final_answer": "",
        }

        state = app.invoke(initial_state, config)
        print("=== PAUSED FOR HUMAN REVIEW ===")
        print("Draft:\n" + state["draft"])
        print("\nFull trace:")
        for line in state["trace"]:
            print(" ", line)

        app.update_state(config, {"human_decision": "approved"})
        final_state = app.invoke(None, config)

        print("\n=== FINAL ANSWER ===")
        print(final_state["final_answer"])
        print("\nPost-finalize trace (memory save step):")
        for line in final_state["trace"][len(state["trace"]):]:
            print(" ", line)


if __name__ == "__main__":
    if "--again" in sys.argv:
        # Session 2: vague goal, no city or budget mentioned — relies entirely
        # on long-term memory saved from session 1.
        run("Plan another trip like before.", thread_id="session-2")
    else:
        run("Planning a trip to Paris and Tokyo, budget 500 USD each. "
            "Give me expected weather and what my budget converts to locally, for both cities.",
            thread_id="session-1")