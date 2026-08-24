"""
Mini Project #5: Observability Layer for the Trip Planning Agent
=====================================================================

A local, self-contained observability SDK (same shape as LangSmith's
@traceable / Langfuse's @observe + spans/generations) instrumenting the
agent from earlier modules, covering as much of the Observability module
as fits in one runnable, no-API-key file:

  Section 1 (Why it matters):  every run below is non-deterministic-shaped
                                (parallel city processing, occasional
                                failures) and the trace/dashboard is what
                                makes each run's behavior reconstructable
  Section 2 (LangSmith-style): @traced decorator (~ @traceable), nested
                                spans forming a call tree (~ trace explorer),
                                run metadata (user/feature tags), and a
                                prompt-version experiment comparison
                                (~ comparing prompt versions with experiments)
  Section 3 (Langfuse-style):  self-hosted (fully local SQLite — the
                                self-hosting side of that trade-off),
                                trace()/span()/generation() objects, cost
                                tracking per "model" via a pricing table,
                                user/session tracking, a dashboard query
  Section 4 (Metrics):         TTFT vs total latency, input/output/total
                                tokens, cost per request/user/feature, error
                                rate by category, cache hit rate, retrieval
                                relevance scores, a mock satisfaction score
  Section 5 (Logging):         structured JSON logs, PII scrubbing, log
                                levels (DEBUG/INFO/ERROR), a correlation ID
                                threaded through every log line for one run

Run with:
    python ObservabilityAndMonitoring.py
"""

import json
import logging
import re
import sqlite3
import time
import random
import statistics
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from uuid import uuid4

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===========================================================================
# SECTION 5: Structured JSON logging, PII scrubbing, correlation IDs
# ===========================================================================

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "correlation_id": _correlation_id.get(),
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload)


logger = logging.getLogger("trip_agent")
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logger.handlers = [_handler]


def log(level: str, msg: str, **fields):
    """Thin wrapper so call sites read as `log("INFO", "...", key=val)`."""
    getattr(logger, level.lower())(msg, extra={"extra_fields": fields})


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")


def scrub_pii(text: str) -> str:
    """Section 5: redact common PII patterns before anything gets logged."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text


# ===========================================================================
# SECTION 2/3: Local tracing SDK — trace / span / generation, mirroring
# LangSmith's @traceable and Langfuse's @observe()/span()/generation().
# Persisted to SQLite so the "dashboard" section can query real aggregates,
# not just print numbers from the current process's memory.
# ===========================================================================

TRACE_DB = "observability.db"


def _obs_db():
    conn = sqlite3.connect(TRACE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS spans (
        span_id TEXT, trace_id TEXT, parent_id TEXT, name TEXT, kind TEXT,
        user_id TEXT, feature TEXT, prompt_version TEXT,
        model TEXT, tokens_in INTEGER, tokens_out INTEGER, cost REAL,
        status TEXT, error TEXT, ttft_ms REAL, duration_ms REAL, created_at TEXT)""")
    return conn


# Mock pricing table (Section 3: cost tracking per model) — $ per 1K tokens.
_PRICING = {
    "rule-based-v1": {"in": 0.0, "out": 0.0},        # deterministic tools: free
    "mock-writer-v1": {"in": 0.0005, "out": 0.0015},
    "mock-critic-v1": {"in": 0.0003, "out": 0.0008},
}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough char-per-token heuristic


class Span:
    """A traced unit of work. `kind="generation"` additionally tracks model/
    token/cost fields (Langfuse's distinction between a span and a generation,
    Section 3)."""

    def __init__(self, trace_id, name, kind="span", parent_id=None, model=None,
                 user_id=None, feature=None, prompt_version=None):
        self.span_id = str(uuid4())
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.name = name
        self.kind = kind
        self.model = model
        self.user_id = user_id
        self.feature = feature
        self.prompt_version = prompt_version
        self._start = time.time()
        self._first_byte = None
        self.tokens_in = 0
        self.tokens_out = 0
        self.status = "ok"
        self.error = None

    def mark_first_token(self):
        """Section 4: TTFT — call this the moment the first output is ready."""
        self._first_byte = time.time()

    def record_generation(self, input_text: str, output_text: str):
        self.tokens_in = _estimate_tokens(input_text)
        self.tokens_out = _estimate_tokens(output_text)

    def fail(self, error: str):
        self.status = "error"
        self.error = error

    def end(self):
        duration_ms = (time.time() - self._start) * 1000
        ttft_ms = ((self._first_byte - self._start) * 1000) if self._first_byte else None
        cost = 0.0
        if self.kind == "generation" and self.model in _PRICING:
            p = _PRICING[self.model]
            cost = (self.tokens_in / 1000) * p["in"] + (self.tokens_out / 1000) * p["out"]

        conn = _obs_db()
        conn.execute(
            """INSERT INTO spans (span_id, trace_id, parent_id, name, kind, user_id, feature,
               prompt_version, model, tokens_in, tokens_out, cost, status, error, ttft_ms,
               duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.span_id, self.trace_id, self.parent_id, self.name, self.kind, self.user_id,
             self.feature, self.prompt_version, self.model, self.tokens_in, self.tokens_out,
             cost, self.status, self.error, ttft_ms, duration_ms, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

        log("DEBUG" if self.status == "ok" else "ERROR",
            f"span '{self.name}' finished",
            span_id=self.span_id, trace_id=self.trace_id, kind=self.kind,
            duration_ms=round(duration_ms, 1), cost=round(cost, 6), status=self.status)


class Trace:
    """One top-level run — the parent of every span/generation it contains."""

    def __init__(self, name, user_id, feature, prompt_version="v1"):
        self.trace_id = str(uuid4())
        self.name = name
        self.user_id = user_id
        self.feature = feature
        self.prompt_version = prompt_version
        _correlation_id.set(self.trace_id)   # Section 5: correlation ID for this whole run
        log("INFO", f"trace '{name}' started", trace_id=self.trace_id, user_id=user_id, feature=feature)

    def span(self, name, **kwargs):
        return Span(self.trace_id, name, kind="span", user_id=self.user_id,
                     feature=self.feature, prompt_version=self.prompt_version, **kwargs)

    def generation(self, name, model, **kwargs):
        return Span(self.trace_id, name, kind="generation", model=model, user_id=self.user_id,
                     feature=self.feature, prompt_version=self.prompt_version, **kwargs)


def traced(feature):
    """Section 2's @traceable / Section 3's @observe() — wraps a whole run as
    one top-level Trace, tagging every span inside it with feature/user."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, user_id="anonymous", prompt_version="v1", **kwargs):
            trace = Trace(fn.__name__, user_id, feature, prompt_version)
            try:
                return fn(*args, trace=trace, **kwargs)
            finally:
                log("INFO", f"trace '{fn.__name__}' finished", trace_id=trace.trace_id)
        return wrapper
    return decorator


# ===========================================================================
# Agent logic — same mock brains as earlier mini-projects, now instrumented.
# ===========================================================================

_MOCK_WEATHER = {"paris": {"temp_c": 14, "condition": "light rain"}, "tokyo": {"temp_c": 22, "condition": "clear"}}
_MOCK_FX_RATES_PER_USD = {"eur": 0.92, "jpy": 151.0}
_CITY_CURRENCY = {"paris": "eur", "tokyo": "jpy"}

_fx_cache = {}  # Section 4: cache hit rate — cache currency lookups for a few seconds
CACHE_TTL_SECONDS = 30


def _weather(city, trace):
    span = trace.span("weather_researcher")
    try:
        key = city.lower()
        if key not in _MOCK_WEATHER:
            raise ValueError(f"No weather data for '{city}'")
        result = _MOCK_WEATHER[key]
        span.record_generation(city, json.dumps(result))
        return result
    except Exception as e:
        span.fail(str(e))
        raise
    finally:
        span.end()


def _currency(city, budget, trace):
    span = trace.span("currency_researcher")
    cache_key = (city.lower(), budget)
    try:
        cached = _fx_cache.get(cache_key)
        if cached and time.time() - cached[1] < CACHE_TTL_SECONDS:
            log("DEBUG", "cache hit", key=str(cache_key))
            span.record_generation(f"{city},{budget}", json.dumps(cached[0]))
            return cached[0]

        log("DEBUG", "cache miss", key=str(cache_key))
        key = city.lower()
        if key not in _CITY_CURRENCY:
            raise ValueError(f"No currency mapping for '{city}'")
        currency = _CITY_CURRENCY[key]
        result = {"currency": currency.upper(), "converted": round(budget * _MOCK_FX_RATES_PER_USD[currency], 2)}
        _fx_cache[cache_key] = (result, time.time())
        span.record_generation(f"{city},{budget}", json.dumps(result))
        return result
    except Exception as e:
        span.fail(str(e))
        raise
    finally:
        span.end()


def _write_report(cities, results, trace):
    """A 'generation' — Section 2/3's prompt-version experiment: v1 omits
    currency (a worse prompt), v2 includes it (a fixed prompt) — same trick
    used in earlier no-LLM mini-projects, now scored and compared via traces."""
    gen = trace.generation("write_report", model="mock-writer-v1")
    gen.mark_first_token()
    lines = [f"Trip report for: {', '.join(c.title() for c in cities)}", ""]
    for city in cities:
        lines.append(f"{city.title()} weather: {results[city]['weather']}")
        if trace.prompt_version == "v2":
            lines.append(f"{city.title()} budget converts to: {results[city]['currency']}")
    draft = "\n".join(lines)
    gen.record_generation(input_text=json.dumps(results), output_text=draft)
    gen.end()
    return draft


def _critique(draft, cities, trace):
    gen = trace.generation("critique", model="mock-critic-v1")
    gen.mark_first_token()
    missing = [c for c in cities if f"{c.title()} budget converts to" not in draft]
    approved = not missing
    feedback = "complete" if approved else f"missing currency for {missing}"
    gen.record_generation(input_text=draft, output_text=feedback)
    gen.end()
    return approved, feedback


def _retrieval_relevance(query: str, memories: list[str]) -> float:
    """Section 4: retrieval relevance score, reusing the TF-IDF approach
    from the Memory Systems mini-project."""
    if not memories:
        return 0.0
    vec = TfidfVectorizer().fit(memories + [query])
    sims = cosine_similarity(vec.transform([query]), vec.transform(memories))[0]
    return round(float(sims.max()), 3)


@traced(feature="trip_planner")
def run_goal(goal: str, budget: float, cities: list[str], trace: Trace, user_note: str = ""):
    if user_note:
        log("INFO", "user note received", note=scrub_pii(user_note))  # Section 5: scrub before logging

    fake_memories = ["User prefers window seats", "User has planned trips to Paris"]
    relevance = _retrieval_relevance(goal, fake_memories)
    log("INFO", "memory retrieval scored", relevance=relevance)

    results = {}
    for city in cities:
        weather = _weather(city, trace)
        currency = _currency(city, budget, trace)
        results[city] = {"weather": weather, "currency": currency}

    draft = _write_report(cities, results, trace)
    approved, feedback = _critique(draft, cities, trace)

    log("INFO", "run complete", approved=approved, feedback=feedback, prompt_version=trace.prompt_version)
    return {"draft": draft, "approved": approved, "trace_id": trace.trace_id}


# ===========================================================================
# SECTION 2/3: Dashboard — aggregate queries over the SQLite span store,
# the same numbers a LangSmith/Langfuse dashboard renders from traces.
# ===========================================================================

def print_dashboard():
    conn = _obs_db()
    rows = conn.execute("SELECT * FROM spans").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM spans LIMIT 1").description]
    conn.close()
    data = [dict(zip(cols, r)) for r in rows]

    print("\n" + "=" * 60)
    print("DASHBOARD (Sections 2-4: aggregated from real traced spans)")
    print("=" * 60)

    durations = [d["duration_ms"] for d in data if d["duration_ms"] is not None]
    if durations:
        durations.sort()
        p50 = statistics.median(durations)
        p95 = durations[int(len(durations) * 0.95) - 1] if len(durations) > 1 else durations[0]
        print(f"Latency  — p50: {p50:.1f}ms   p95: {p95:.1f}ms   (n={len(durations)} spans)")

    total_cost = sum(d["cost"] or 0 for d in data)
    print(f"Total cost this session: ${total_cost:.6f}")

    print("\nCost by feature:")
    by_feature = {}
    for d in data:
        by_feature[d["feature"]] = by_feature.get(d["feature"], 0) + (d["cost"] or 0)
    for feature, cost in by_feature.items():
        print(f"  {feature}: ${cost:.6f}")

    print("\nCost by prompt_version (Section 2: comparing versions with experiments):")
    by_version = {}
    approvals_by_version = {}
    for d in data:
        v = d["prompt_version"]
        by_version[v] = by_version.get(v, 0) + (d["cost"] or 0)
    for v, cost in by_version.items():
        n = sum(1 for d in data if d["prompt_version"] == v and d["kind"] == "generation")
        print(f"  {v}: ${cost:.6f} across {n} generation spans")

    errors = [d for d in data if d["status"] == "error"]
    total_tool_spans = [d for d in data if d["kind"] == "span"]
    error_rate = (len(errors) / len(total_tool_spans) * 100) if total_tool_spans else 0
    print(f"\nError rate (tool spans): {error_rate:.1f}% ({len(errors)}/{len(total_tool_spans)})")
    for e in errors:
        print(f"  - {e['name']}: {e['error']}")

    total_tokens_in = sum(d["tokens_in"] or 0 for d in data)
    total_tokens_out = sum(d["tokens_out"] or 0 for d in data)
    print(f"\nToken usage — input: {total_tokens_in}, output: {total_tokens_out}, total: {total_tokens_in + total_tokens_out}")


if __name__ == "__main__":
    print("--- Run 1: normal request, prompt v1 (buggy — omits currency) ---")
    run_goal("Plan a trip to Paris and Tokyo", 500, ["paris", "tokyo"],
              user_id="u123", prompt_version="v1", user_note="Contact me at raj@example.com if issues")

    print("\n--- Run 2: same request, prompt v2 (fixed — includes currency) ---")
    run_goal("Plan a trip to Paris and Tokyo", 500, ["paris", "tokyo"],
              user_id="u123", prompt_version="v2")

    print("\n--- Run 3: same city again shortly after -> currency cache hit ---")
    run_goal("Plan another Paris trip", 500, ["paris"], user_id="u123", prompt_version="v2")

    print("\n--- Run 4: unknown city -> deliberate tool failure (error rate) ---")
    try:
        run_goal("Plan a trip to Atlantis", 200, ["atlantis"], user_id="u456", prompt_version="v2")
    except ValueError:
        pass  # error already recorded on the span before re-raising

    print_dashboard()