import os
import re
import json
import time
import logging
from contextlib import asynccontextmanager

import httpx
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
API_TOKEN = os.getenv("API_TOKEN")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "https://your-cloudflare-pages-url.pages.dev")

SYSTEM_PROMPT = (
    "You are a planner agent for an app called Schelude. The user types anything\n"
    "related to their schedule, tasks, or events in plain English. Parse it and\n"
    "return ONLY a valid JSON object — no markdown, no explanation, nothing else.\n"
    "\n"
    "Return this exact shape:\n"
    "{\n"
    "  type: event | task,\n"
    "  title: short clean title max 6 words,\n"
    "  tag: test | homework | school | extracurricular | work | event,\n"
    "  day: number or null (day of current month),\n"
    "  time: H:MM AM/PM or null,\n"
    "  due: human-readable due string or null\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- If input has a specific time, type is event\n"
    "- If input is a to-do with no time, type is task\n"
    "- Infer day numbers from context relative to today\n"
    "- due is only for tasks, written like Thursday or Apr 18\n"
    "- Tag inference: test/exam/quiz=test, finish/write/read/outline=homework,\n"
    "  class/period/AP=school, club/practice/team=extracurricular\n"
    "- Never return anything except the JSON object"
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
    http_client = httpx.AsyncClient(timeout=5.0)
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ParseRequest(BaseModel):
    text: str


class CreateEventRequest(BaseModel):
    title: str
    date: str
    time: str | None = None
    description: str | None = None
    reminder_minutes: int | None = None


async def check_rate_limit(ip: str) -> bool:
    """Sliding window rate limit via Upstash Redis REST pipeline. Fail-open if Redis is unavailable."""
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        return True

    key = f"ratelimit:{ip}"
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - RATE_LIMIT_WINDOW_MS

    # Pipeline: clean expired entries → add current → count → set TTL
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
        count = results[2]["result"]  # ZCARD result (includes current request)
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


def sanitize_input(text: str) -> str:
    text = text[:500]
    text = re.sub(r"<[^>]+>", "", text)
    # Truncate at characters that could break system-prompt boundaries or inject instructions
    text = re.split(r"[\n\r\x00\x1a`]", text)[0]
    return text.strip()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/parse")
async def parse(request: Request, body: ParseRequest):
    # 1. API token check — do not proceed to Groq if this fails
    token = request.headers.get("x-api-token")
    if not token or token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2. Rate limit — sliding window per IP
    forwarded_for = request.headers.get("x-forwarded-for", "")
    ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
    if not await check_rate_limit(ip):
        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    # 3. Input sanitization
    text = sanitize_input(body.text)
    if not text:
        return JSONResponse(status_code=400, content={"error": "invalid input"})

    # 4. Groq call
    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        raw = completion.choices[0].message.content
    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        return JSONResponse(content={"success": False, "error": "Failed to reach language model"})

    # 5. JSON validation — never pass raw unparsed output to the frontend
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # One recovery attempt: find the first {...} block in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return JSONResponse(content={"success": False, "error": "Model returned invalid JSON"})
        else:
            return JSONResponse(content={"success": False, "error": "Model returned invalid JSON"})

    return {"success": True, "data": parsed}


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
