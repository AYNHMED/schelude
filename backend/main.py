import os
import re
import json
import time
import hashlib
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List

import httpx
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from groq import AsyncGroq
from dotenv import load_dotenv

# Optional: Upstash Vector SDK (gracefully skipped if not installed)
try:
    from upstash_vector import Index as VectorIndex
    _vector_sdk_available = True
except ImportError:
    _vector_sdk_available = False

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
API_TOKEN = os.getenv("API_TOKEN")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://your-cloudflare-pages-url.pages.dev")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPSTASH_VECTOR_URL = os.getenv("UPSTASH_VECTOR_URL", "")
UPSTASH_VECTOR_TOKEN = os.getenv("UPSTASH_VECTOR_TOKEN", "")

# Upstash Vector — initialise SDK client if available, fall back to REST-only
vector_index = None
if UPSTASH_VECTOR_URL and UPSTASH_VECTOR_TOKEN and _vector_sdk_available:
    try:
        vector_index = VectorIndex(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)
        logger.info("Upstash Vector index connected via SDK")
    except Exception as _ve:
        logger.warning("Vector SDK init failed (REST fallback active): %s", _ve)

PARSE_SYSTEM_PROMPT = (
    "Extract scheduling intent from user input. Return JSON only, no other text. Schema:\n"
    "{\n"
    "  type: 'event'|'task'|'habit'|'none',\n"
    "  title: string,\n"
    "  date: 'YYYY-MM-DD' or null,\n"
    "  time: 'HH:MM' or null,\n"
    "  category: 'school'|'work'|'health'|'life'|'social'|'self-improvement',\n"
    "  priority: 'urgent'|'high'|'normal'|'low',\n"
    "  recurrence: 'daily'|'weekly'|'monthly'|'yearly'|null,\n"
    "  recurrence_days: ['mon','tue','wed','thu','fri','sat','sun'] or null,\n"
    "  recurrence_end: 'YYYY-MM-DD' or null,\n"
    "  recurrence_count: number or null,\n"
    "  duration_minutes: number or null,\n"
    "  notes: string or null\n"
    "}\n"
    "Category detection:\n"
    "school = homework, test, exam, quiz, project, essay, study, class, lecture, final, assignment\n"
    "work = meeting, deadline, presentation, report, interview, client\n"
    "health = workout, gym, run, walk, doctor, medication, sleep, meal prep, diet\n"
    "life = birthday, appointment, errand, shopping, travel, bill, chores\n"
    "social = party, hangout, date, call, dinner, wedding, event\n"
    "self-improvement = habit, read, meditate, journal, practice, learn, course\n"
    "If no scheduling intent return {\"type\": \"none\"}."
)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://schelude-api.onrender.com/auth/callback")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"
GOOGLE_SCOPES = " ".join([
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
    "openid",
    "email",
])

TOKENS_FILE = "/tmp/schelude_tokens.json"

google_tokens = {"access_token": None, "refresh_token": None, "expires_at": 0}

try:
    if os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, "r") as f:
            google_tokens = json.load(f)
except Exception:
    pass

RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW_MS = 60 * 1000

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=15.0)
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class ParseRequest(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: str
    parts: str


class ChatContext(BaseModel):
    events: list = []
    tasks: list = []
    habits: list = []


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    context: ChatContext = ChatContext()


class CreateEventRequest(BaseModel):
    title: str
    date: str
    time: Optional[str] = None
    description: Optional[str] = None
    reminder_minutes: Optional[int] = None
    recurrence: Optional[str] = None
    recurrence_days: Optional[List[str]] = None
    recurrence_end: Optional[str] = None
    recurrence_count: Optional[int] = None


async def check_rate_limit(ip: str) -> bool:
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return True

    key = f"ratelimit:{ip}"
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - RATE_LIMIT_WINDOW_MS

    pipeline = [
        ["ZREMRANGEBYSCORE", key, 0, cutoff_ms],
        ["ZADD", key, now_ms, str(now_ms)],
        ["ZCARD", key],
        ["EXPIRE", key, 60],
    ]

    try:
        resp = await http_client.post(
            f"{UPSTASH_REDIS_REST_URL}/pipeline",
            headers={
                "Authorization": f"Bearer {UPSTASH_REDIS_REST_TOKEN}",
                "Content-Type": "application/json",
            },
            json=pipeline,
        )
        resp.raise_for_status()
        results = resp.json()
        count = results[2]["result"]
        return count <= RATE_LIMIT_MAX
    except Exception as exc:
        logger.warning("Upstash unavailable, failing open: %s", exc)
        return True


def save_tokens():
    try:
        with open(TOKENS_FILE, "w") as f:
            json.dump(google_tokens, f)
    except Exception as e:
        logger.warning("Could not save tokens to file: %s", e)


async def refresh_access_token():
    if not google_tokens.get("refresh_token"):
        return False
    try:
        resp = await http_client.post(GOOGLE_TOKEN_URL, data={
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "refresh_token": google_tokens["refresh_token"],
            "grant_type": "refresh_token",
        })
        data = resp.json()
        if "access_token" in data:
            google_tokens["access_token"] = data["access_token"]
            google_tokens["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
            save_tokens()
            return True
        return False
    except Exception as e:
        logger.error("Token refresh error: %s", e)
        return False


async def get_valid_access_token():
    if not google_tokens.get("access_token"):
        return None
    if time.time() >= google_tokens.get("expires_at", 0):
        success = await refresh_access_token()
        if not success:
            return None
    return google_tokens["access_token"]


# ── Second Brain: Memory helpers ─────────────────────────────────────────────

def make_memory_id(content: str) -> str:
    """Deterministic-ish ID; uses current time so repeated content gets new slots."""
    return hashlib.md5((content + str(time.time())).encode()).hexdigest()[:16]


async def store_memory(content: str, metadata: dict, namespace: str = "schelude") -> None:
    """Store a memory vector via the Upstash Vector REST API."""
    if not UPSTASH_VECTOR_URL or not UPSTASH_VECTOR_TOKEN:
        return
    try:
        mem_id = make_memory_id(content)
        url = f"{UPSTASH_VECTOR_URL}/upsert"
        headers = {
            "Authorization": f"Bearer {UPSTASH_VECTOR_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "id": mem_id,
            "data": content,
            "metadata": {
                **metadata,
                "timestamp": time.time(),
                "namespace": namespace,
            },
        }
        await http_client.post(url, json=payload, headers=headers, timeout=10.0)
    except Exception as e:
        logger.warning("Memory store failed: %s", e)


async def retrieve_memories(query: str, top_k: int = 8) -> list:
    """Semantic search over stored memories via the Upstash Vector REST API."""
    if not UPSTASH_VECTOR_URL or not UPSTASH_VECTOR_TOKEN:
        return []
    try:
        url = f"{UPSTASH_VECTOR_URL}/query"
        headers = {
            "Authorization": f"Bearer {UPSTASH_VECTOR_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "data": query,
            "topK": top_k,
            "includeMetadata": True,
            "includeData": True,
        }
        resp = await http_client.post(url, json=payload, headers=headers, timeout=10.0)
        results = resp.json()
        if isinstance(results, list):
            return [
                {
                    "content": r.get("data", ""),
                    "score": r.get("score", 0),
                    "metadata": r.get("metadata", {}),
                }
                for r in results
                if r.get("score", 0) > 0.6
            ]
        return []
    except Exception as e:
        logger.warning("Memory retrieval failed: %s", e)
        return []


async def store_event_memory(event: dict) -> None:
    content = (
        f"Event: {event.get('title', '')} on {event.get('date', 'unspecified')} "
        f"at {event.get('time', 'unspecified')}. "
        f"Category: {event.get('category', 'general')}. "
        f"Priority: {event.get('priority', 'normal')}."
    )
    await store_memory(content, {
        "type": "event",
        "title": event.get("title", ""),
        "category": event.get("category", ""),
        "date": event.get("date", ""),
    })


async def store_task_memory(task: dict) -> None:
    content = (
        f"Task: {task.get('title', '')}. "
        f"Due: {task.get('date', 'no due date')}. "
        f"Category: {task.get('category', 'general')}. "
        f"Priority: {task.get('priority', 'normal')}."
    )
    await store_memory(content, {
        "type": "task",
        "title": task.get("title", ""),
        "category": task.get("category", ""),
        "date": task.get("date", ""),
    })


async def store_conversation_memory(user_msg: str, ai_response: str) -> None:
    content = (
        f"User asked: {user_msg[:200]}. "
        f"Assistant responded: {ai_response[:300]}"
    )
    await store_memory(content, {
        "type": "conversation",
        "user_message": user_msg[:100],
    })


# ── Input sanitisation ────────────────────────────────────────────────────────

def sanitize_input(text: str) -> str:
    text = text[:500]
    text = re.sub(r"<[^>]+>", "", text)
    text = re.split(r"[\n\r\x00\x1a`]", text)[0]
    return text.strip()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/parse")
async def parse(request: Request, body: ParseRequest):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    forwarded_for = request.headers.get("x-forwarded-for", "")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
    if not await check_rate_limit(ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    text = sanitize_input(body.text)
    if not text:
        return JSONResponse(status_code=400, content={"error": "invalid input"})

    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = completion.choices[0].message.content
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        return JSONResponse(content={"success": False, "error": "Failed to reach language model"})

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return JSONResponse(content={"success": False, "error": "Model returned invalid JSON"})
        else:
            return JSONResponse(content={"success": False, "error": "Model returned invalid JSON"})

    return {"success": True, "data": parsed}


async def run_groq_parser(text: str):
    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = completion.choices[0].message.content
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        return None
    except Exception as exc:
        logger.error("Groq parser error: %s", exc)
        return None


async def call_gemini(message: str, history: list, context: dict, system_prompt: str) -> str:
    """Call Gemini 1.5 Flash via the v1 REST API. Returns only user-friendly strings on failure."""
    try:
        if not GEMINI_API_KEY:
            return "I'm thinking through that — please try again in a moment."

        url = (
            f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        )

        contents = []
        for h in history[-20:]:
            role = h.get("role", "user")
            if role not in ("user", "model"):
                role = "user"
            contents.append({"role": role, "parts": [{"text": h.get("parts", "")}]})
        contents.append({"role": "user", "parts": [{"text": message}]})

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
        }

        resp = await http_client.post(url, json=payload, timeout=30.0)
        data = resp.json()

        if "candidates" in data and data["candidates"]:
            return data["candidates"][0]["content"]["parts"][0]["text"]

        if "error" in data:
            logger.error("Gemini API error: %s", data["error"])

        return "I'm thinking through that — please try again in a moment."

    except Exception as exc:
        logger.error("call_gemini unhandled exception: %s", exc)
        return "I'm thinking through that — please try again in a moment."


@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    forwarded_for = request.headers.get("x-forwarded-for", "")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
    if not await check_rate_limit(ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    text = sanitize_input(body.message)
    if not text:
        return JSONResponse(status_code=400, content={"error": "invalid input"})

    event_titles = ', '.join([e.get('title', '') for e in body.context.events[:5] if isinstance(e, dict)]) or 'none'
    task_titles  = ', '.join([t.get('title', '') for t in body.context.tasks[:5]  if isinstance(t, dict)]) or 'none'

    context_dict = {
        "events": body.context.events,
        "tasks": body.context.tasks,
        "habits": body.context.habits,
    }
    system_prompt = f"""You are Schelude AI, a powerful intelligent assistant \
built into a personal productivity app. You function exactly like ChatGPT \
or Claude — you can answer anything, help with any topic, and you also \
have special abilities to manage the user's schedule.

USER CONTEXT:
Events: {len(context_dict.get('events', []))} this month
Tasks: {len(context_dict.get('tasks', []))} active
Habits: {len(context_dict.get('habits', []))} tracked
Recent events: {', '.join([str(e.get('title','')) for e in context_dict.get('events',[])[:5]]) or 'none'}
Current tasks: {', '.join([str(t.get('title','')) for t in context_dict.get('tasks',[])[:5]]) or 'none'}

YOUR CAPABILITIES:
- Answer any question on any topic (science, math, history, coding, etc.)
- Create detailed study plans with day-by-day breakdowns
- Schedule events and tasks (confirm when added)
- Give actionable advice on health, wellness, productivity
- Help plan projects, weeks, semesters
- Explain concepts clearly
- Motivate and support the user

RESPONSE STYLE:
- Be warm, direct, and genuinely helpful like a smart friend
- Use **bold** for important terms and action items
- Use bullet points for lists and plans
- For study plans: break into specific days/weeks with topics
- For scheduling: confirm what was added and when
- For questions: give a complete, accurate answer
- Never be vague or generic
- Never mention API errors, model names, or technical issues
- Max 300 words unless a detailed plan requires more

EXAMPLES OF GOOD RESPONSES:
User: "I need to cram physics in 2 weeks"
You: "Here's your 2-week physics cram plan:
**Week 1:**
- Days 1-2: Mechanics (Newton's laws, kinematics)
- Days 3-4: Energy, work, momentum
- Day 5: Practice problems + review
**Week 2:**
- Days 6-7: Electricity and circuits
- Days 8-9: Waves and optics
- Days 10-11: Modern physics
- Day 14: Full mock exam
I've added daily study sessions to your calendar. Good luck! 💪"

User: "What is the Pythagorean theorem"
You: "The Pythagorean theorem states that in a right triangle, \
**a² + b² = c²** where c is the hypotenuse (longest side). \
For example, a triangle with sides 3 and 4 has a hypotenuse of 5."
"""

    history_dicts = [{"role": m.role, "parts": m.parts} for m in body.history]

    # ── Second Brain: inject relevant memories into the prompt ────────────────
    memories = await retrieve_memories(text)
    if memories:
        memory_context = "\n\nRELEVANT MEMORIES FROM YOUR HISTORY:\n"
        for m in memories:
            memory_context += f"- {m['content']}\n"
        system_prompt = system_prompt + memory_context

    parsed_result, gemini_response = await asyncio.gather(
        run_groq_parser(text),
        call_gemini(text, history_dicts, context_dict, system_prompt)
    )

    # ── Second Brain: store this exchange + any created items ─────────────────
    asyncio.create_task(store_conversation_memory(text, gemini_response))
    if parsed_result:
        if parsed_result.get("type") == "event":
            asyncio.create_task(store_event_memory(parsed_result))
        elif parsed_result.get("type") == "task":
            asyncio.create_task(store_task_memory(parsed_result))

    return {
        "action": parsed_result,
        "response": gemini_response,
        "success": True
    }


class MemoryStoreRequest(BaseModel):
    type: str  # "event" | "task" | "habit" | "note"
    data: dict = {}


@app.post("/memory/store")
async def memory_store(request: Request, body: MemoryStoreRequest):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        if body.type == "event":
            await store_event_memory(body.data)
        elif body.type == "task":
            await store_task_memory(body.data)
        elif body.type in ("habit", "note"):
            title = body.data.get("title") or body.data.get("name", "")
            content = f"{body.type.capitalize()}: {title}. {json.dumps(body.data)[:200]}"
            await store_memory(content, {"type": body.type, **{k: str(v) for k, v in body.data.items() if isinstance(v, str)}})
        else:
            content = json.dumps(body.data)[:400]
            await store_memory(content, {"type": body.type})
        return {"success": True}
    except Exception as e:
        logger.error("Memory store endpoint error: %s", e)
        return {"success": False}


@app.get("/memory/search")
async def memory_search(request: Request, q: str = ""):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not q:
        return {"success": True, "memories": []}

    q = q[:300]
    memories = await retrieve_memories(q, top_k=5)
    return {"success": True, "memories": memories}


@app.get("/auth/google")
async def auth_google():
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "access_type": "offline",
        "prompt": "consent",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@app.get("/auth/callback")
async def auth_callback(request: Request):
    error = request.query_params.get("error")
    code = request.query_params.get("code")

    if error or not code:
        return RedirectResponse(f"{ALLOWED_ORIGIN}?google_auth=error")

    try:
        resp = await http_client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        })
        data = resp.json()
        if "access_token" not in data:
            return RedirectResponse(f"{ALLOWED_ORIGIN}?google_auth=error")
        google_tokens["access_token"] = data["access_token"]
        google_tokens["refresh_token"] = data.get("refresh_token")
        google_tokens["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
        save_tokens()
        return RedirectResponse(f"{ALLOWED_ORIGIN}?google_auth=success")
    except Exception:
        return RedirectResponse(f"{ALLOWED_ORIGIN}?google_auth=error")


@app.get("/auth/status")
async def auth_status(request: Request):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {
        "connected": google_tokens.get("access_token") is not None,
        "has_refresh_token": google_tokens.get("refresh_token") is not None,
    }


@app.get("/auth/disconnect")
async def auth_disconnect(request: Request):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    google_tokens["access_token"] = None
    google_tokens["refresh_token"] = None
    google_tokens["expires_at"] = 0
    try:
        if os.path.exists(TOKENS_FILE):
            os.remove(TOKENS_FILE)
    except Exception:
        pass
    return {"success": True}


@app.get("/calendar/events")
async def get_calendar_events(request: Request):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    access_token = await get_valid_access_token()
    if not access_token:
        return JSONResponse(content={"success": False, "error": "not_connected"})

    now = datetime.now(timezone.utc)
    time_min = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    if now.month == 12:
        next_month = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        next_month = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    time_max = next_month.isoformat()

    async def fetch_events(tok: str):
        return await http_client.get(
            f"{GOOGLE_CALENDAR_BASE}/calendars/primary/events",
            headers={"Authorization": f"Bearer {tok}"},
            params={
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": 100,
            },
        )

    resp = await fetch_events(access_token)

    if resp.status_code == 401:
        refreshed = await refresh_access_token()
        if not refreshed:
            return JSONResponse(content={"success": False, "error": "auth_expired"})
        access_token = google_tokens["access_token"]
        resp = await fetch_events(access_token)
        if resp.status_code != 200:
            return JSONResponse(content={"success": False, "error": "auth_expired"})

    data = resp.json()
    events = [
        {
            "id": event["id"],
            "title": event["summary"],
            "start": event["start"].get("dateTime") or event["start"].get("date"),
            "end": event["end"].get("dateTime") or event["end"].get("date"),
            "description": event.get("description", ""),
            "location": event.get("location", ""),
            "source": "google",
        }
        for event in data.get("items", [])
        if event.get("summary")
    ]
    return {"success": True, "events": events}


def build_rrule(body: CreateEventRequest) -> list:
    if not body.recurrence:
        return []
    day_map = {"mon": "MO", "tue": "TU", "wed": "WE", "thu": "TH", "fri": "FR", "sat": "SA", "sun": "SU"}
    freq_map = {"daily": "DAILY", "weekly": "WEEKLY", "monthly": "MONTHLY", "yearly": "YEARLY"}
    freq = freq_map.get(body.recurrence)
    if not freq:
        return []
    rule = f"RRULE:FREQ={freq}"
    if body.recurrence == "weekly" and body.recurrence_days:
        days = ",".join(day_map.get(d.lower(), d.upper()) for d in body.recurrence_days)
        rule += f";BYDAY={days}"
    if body.recurrence_end:
        until = body.recurrence_end.replace("-", "") + "T000000Z"
        rule += f";UNTIL={until}"
    if body.recurrence_count:
        rule += f";COUNT={body.recurrence_count}"
    return [rule]


@app.post("/calendar/events")
async def create_calendar_event(request: Request, body: CreateEventRequest):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    access_token = await get_valid_access_token()
    if not access_token:
        return JSONResponse(content={"success": False, "error": "not_connected"})

    event_body: dict = {"summary": body.title}

    if body.time:
        start_dt = datetime.fromisoformat(f"{body.date}T{body.time}:00")
        end_dt = start_dt + timedelta(hours=1)
        event_body["start"] = {"dateTime": start_dt.isoformat(), "timeZone": "UTC"}
        event_body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": "UTC"}
    else:
        event_body["start"] = {"date": body.date}
        event_body["end"] = {"date": body.date}

    if body.description:
        event_body["description"] = body.description

    if body.reminder_minutes is not None:
        event_body["reminders"] = {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": body.reminder_minutes}],
        }

    rrule = build_rrule(body)
    if rrule:
        event_body["recurrence"] = rrule

    resp = await http_client.post(
        f"{GOOGLE_CALENDAR_BASE}/calendars/primary/events",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=event_body,
    )
    created = resp.json()
    return {"success": True, "event_id": created["id"], "html_link": created.get("htmlLink")}


@app.delete("/calendar/events/{event_id}")
async def delete_calendar_event(event_id: str, request: Request):
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    access_token = await get_valid_access_token()
    if not access_token:
        return JSONResponse(content={"success": False, "error": "not_connected"})

    resp = await http_client.delete(
        f"{GOOGLE_CALENDAR_BASE}/calendars/primary/events/{event_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if resp.status_code == 204:
        return {"success": True}
    if resp.status_code == 404:
        return JSONResponse(content={"success": False, "error": "not_found"})
    return JSONResponse(content={"success": False, "error": f"calendar_error_{resp.status_code}"})
