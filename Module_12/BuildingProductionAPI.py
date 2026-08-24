import os
import re
import time
import uuid
import json
import sqlite3
import tempfile
import concurrent.futures
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ===========================================================================
# SECTION: Agent service (services/agent_service.py in a real project layout)
# — the mock planner/executor/writer from the earlier mini-projects, plus
# the long-term memory store (SQLite + TF-IDF) from the Memory Systems module.
# ===========================================================================

MEMORY_DB = "trip_memory_api.db"
_MOCK_WEATHER = {"paris": {"temp_c": 14, "condition": "light rain"}, "tokyo": {"temp_c": 22, "condition": "clear"}}
_MOCK_FX_RATES_PER_USD = {"eur": 0.92, "jpy": 151.0}
_CITY_CURRENCY = {"paris": "eur", "tokyo": "jpy"}


class AgentError(Exception):
    """Raised for agent-level failures (e.g. unrecognized city) — caught by
    a dedicated exception handler below instead of surfacing a raw 500."""
    def __init__(self, message: str):
        self.message = message


def _weather(city: str) -> dict:
    key = city.lower()
    if key not in _MOCK_WEATHER:
        raise AgentError(f"No weather data for '{city}'")
    return _MOCK_WEATHER[key]


def _currency(city: str, budget_usd: float) -> dict:
    key = city.lower()
    if key not in _CITY_CURRENCY:
        raise AgentError(f"No currency mapping for '{city}'")
    currency = _CITY_CURRENCY[key]
    return {"currency": currency.upper(), "converted": round(budget_usd * _MOCK_FX_RATES_PER_USD[currency], 2)}


def _memory_db():
    conn = sqlite3.connect(MEMORY_DB)
    conn.execute("CREATE TABLE IF NOT EXISTS profile_facts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, fact TEXT, created_at TEXT)")
    return conn


def save_fact(user_id: str, fact: str) -> bool:
    conn = _memory_db()
    existing = [r[0] for r in conn.execute("SELECT fact FROM profile_facts WHERE user_id=?", (user_id,))]
    if existing:
        vec = TfidfVectorizer().fit(existing + [fact])
        if cosine_similarity(vec.transform([fact]), vec.transform(existing))[0].max() > 0.8:
            conn.close()
            return False
    conn.execute("INSERT INTO profile_facts (user_id, fact, created_at) VALUES (?, ?, ?)",
                 (user_id, fact, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()
    return True


def run_goal(goal: str, user_id: str) -> dict:
    """The whole agent pipeline: parse -> execute (parallel) -> write report.
    Raises AgentError on unrecoverable failures, letting the API layer's
    exception handler turn that into a clean HTTP response."""
    goal_lower = goal.lower()
    cities = [c for c in _MOCK_WEATHER if c in goal_lower]
    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:usd|dollars|\$)?", goal_lower)
    budget = float(budget_match.group(1)) if budget_match else 0.0

    if not cities:
        raise AgentError("Could not identify a known city in the request")

    trace = [f"parsed cities={cities}, budget=${budget}"]

    def task(city):
        w = _weather(city)
        c = _currency(city, budget)
        return city, w, c

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cities)) as pool:
        for city, w, c in pool.map(task, cities):
            results[city] = {"weather": w, "currency": c}
            trace.append(f"{city}: weather={w}, currency={c}")

    lines = [f"Trip report for: {', '.join(c.title() for c in cities)}", ""]
    for city, data in results.items():
        lines.append(f"{city.title()} weather: {data['weather']}")
        lines.append(f"{city.title()} budget converts to: {data['currency']}")
    answer = "\n".join(lines)

    save_fact(user_id, f"User has planned trips to: {', '.join(c.title() for c in cities)}")

    return {"answer": answer, "trace": trace, "cities": cities, "steps_used": len(cities) + 1}


# ===========================================================================
# SECTION: Pydantic models (models/schemas.py)
# ===========================================================================

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    steps_used: int
    cities: list[str]


class StartChatResponse(BaseModel):
    conversation_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[ChatResponse] = None


# ===========================================================================
# SECTION: Redis-backed session store (Session and State Management, Sec 3)
# ===========================================================================

redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
SESSION_TTL_SECONDS = 3600


async def get_redis():
    return redis_client


def session_key(user_id: str, conversation_id: str) -> str:
    # Tenant-scoped key (Section 3: multi-tenant isolation) — a session can
    # only ever be looked up under ITS OWNER's user_id, never guessed cross-tenant.
    return f"session:{user_id}:{conversation_id}"


# ===========================================================================
# SECTION: Auth dependency (dependencies.py)
# ===========================================================================

_VALID_API_KEYS = {"demo-key-123": "rajarajan"}


async def get_current_user(x_api_key: str = Header(default=None)) -> str:
    if x_api_key not in _VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
    return _VALID_API_KEYS[x_api_key]


# ===========================================================================
# SECTION: Background task store (in-process; a real deployment would use
# Celery/RQ + Redis so it survives restarts — noted as the trade-off it is)
# ===========================================================================

_TASKS: dict[str, dict] = {}


def _run_goal_background(task_id: str, goal: str, user_id: str):
    try:
        result = run_goal(goal, user_id)
        _TASKS[task_id] = {"status": "completed", "result": result}
    except AgentError as e:
        _TASKS[task_id] = {"status": "failed", "result": None, "error": e.message}


# ===========================================================================
# SECTION: App + middleware (main.py + middleware.py)
# ===========================================================================

app = FastAPI(title="Trip Planning Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example-frontend.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory rate limiter (per API key, fixed window). A real multi-
# instance deployment would back this with Redis so limits are shared across
# server processes — same trade-off as BackgroundTasks vs. a real task queue.
_RATE_LIMIT = {}  # key -> (window_start_ts, count)
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    key = request.headers.get("x-api-key", request.client.host)
    now = time.time()
    window_start, count = _RATE_LIMIT.get(key, (now, 0))
    if now - window_start > RATE_LIMIT_WINDOW:
        window_start, count = now, 0
    count += 1
    _RATE_LIMIT[key] = (window_start, count)
    if count > RATE_LIMIT_MAX:
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded, try again shortly"})
    return await call_next(request)


@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError):
    return JSONResponse(status_code=422, content={"error": exc.message})


# ===========================================================================
# SECTION: Routes — REST (Section 1/3)
# ===========================================================================

@app.post("/chat/start", response_model=StartChatResponse)
async def start_chat(user_id: str = Depends(get_current_user), r: aioredis.Redis = Depends(get_redis)):
    conversation_id = str(uuid.uuid4())
    await r.setex(session_key(user_id, conversation_id), SESSION_TTL_SECONDS, json.dumps({"history": []}))
    return StartChatResponse(conversation_id=conversation_id)


@app.post("/chat/{conversation_id}/message", response_model=ChatResponse)
async def send_message(
    conversation_id: str,
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
    r: aioredis.Redis = Depends(get_redis),
):
    key = session_key(user_id, conversation_id)
    raw = await r.get(key)
    if raw is None:
        raise HTTPException(status_code=404, detail="Conversation not found")  # tenant-safe: same 404 whether missing or belongs to someone else

    result = run_goal(request.message, user_id)  # may raise AgentError -> handled above

    session = json.loads(raw)
    session["history"].append({"message": request.message, "answer": result["answer"]})
    await r.setex(key, SESSION_TTL_SECONDS, json.dumps(session))  # refresh TTL on activity

    return ChatResponse(answer=result["answer"], steps_used=result["steps_used"], cities=result["cities"])


# ===========================================================================
# SECTION: Streaming — SSE (Section 1/2)
# ===========================================================================

@app.post("/chat/{conversation_id}/stream")
async def stream_message(conversation_id: str, request: ChatRequest, user_id: str = Depends(get_current_user)):
    async def event_generator():
        try:
            result = run_goal(request.message, user_id)
        except AgentError as e:
            yield f"event: error\ndata: {e.message}\n\n"
            return
        for line in result["trace"]:
            yield f"data: {line}\n\n"
        yield f"event: final\ndata: {result['answer']}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ===========================================================================
# SECTION: WebSocket (Section 1/2) — persistent, bidirectional
# ===========================================================================

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, x_api_key: Optional[str] = None):
    await websocket.accept()
    user_id = _VALID_API_KEYS.get(x_api_key)
    if user_id is None:
        await websocket.send_json({"type": "error", "data": "invalid api key"})
        await websocket.close()
        return
    try:
        while True:
            goal = await websocket.receive_text()
            try:
                result = run_goal(goal, user_id)
            except AgentError as e:
                await websocket.send_json({"type": "error", "data": e.message})
                continue
            for line in result["trace"]:
                await websocket.send_json({"type": "trace", "data": line})
            await websocket.send_json({"type": "final", "data": result["answer"]})
    except WebSocketDisconnect:
        pass


# ===========================================================================
# SECTION: Background task + status polling (Section 1/2)
# ===========================================================================

@app.post("/chat/run-async", response_model=TaskStatusResponse)
async def run_async(request: ChatRequest, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user)):
    task_id = str(uuid.uuid4())
    _TASKS[task_id] = {"status": "processing", "result": None}
    background_tasks.add_task(_run_goal_background, task_id, request.message, user_id)
    return TaskStatusResponse(task_id=task_id, status="processing")


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    result = ChatResponse(**task["result"]) if task["result"] else None
    return TaskStatusResponse(task_id=task_id, status=task["status"], result=result)


# ===========================================================================
# SECTION: File upload (Section 4)
# ===========================================================================

ALLOWED_UPLOAD_TYPES = {"text/plain"}
MAX_UPLOAD_BYTES = 1 * 1024 * 1024  # 1 MB


def _process_uploaded_facts(temp_path: str, user_id: str):
    try:
        with open(temp_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            save_fact(user_id, line)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # cleanup regardless of success/failure


@app.post("/upload")
async def upload_notes(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    user_id: str = Depends(get_current_user),
):
    if file.content_type not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_BYTES} byte limit")

    file_id = str(uuid.uuid4())
    temp_path = os.path.join(tempfile.gettempdir(), f"{file_id}_{file.filename}")
    with open(temp_path, "wb") as f:
        f.write(contents)

    background_tasks.add_task(_process_uploaded_facts, temp_path, user_id)
    return {"file_id": file_id, "status": "processing"}


@app.get("/health")
async def health():
    return {"status": "ok"}