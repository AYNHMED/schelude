import os
import re
import json
import time
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
