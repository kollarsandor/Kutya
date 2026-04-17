import os
import re
import base64
import json
import uuid
import asyncio
import logging
import time
import collections
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
import httpx
from pydantic import BaseModel, field_validator
from typing import Optional
from replit.object_storage import Client as ObjClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MAX_SESSIONS = 500
SESSION_TTL = 21600
MAX_TOOL_ITERATIONS = 200
MAX_HISTORY_MESSAGES = 60
MAX_STORED_MESSAGES = MAX_HISTORY_MESSAGES * 2
MAX_SUMMARY_CHARS = 600
EXA_NUM_RESULTS = 100
MAX_MODEL_CONTENT_CHARS = 200000
MAX_DELETED_SESSIONS = 1000
MAX_MESSAGE_LENGTH = 100000
MAX_PROMPT_LENGTH = 10000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_INLINE_IMAGE_BYTES = 512 * 1024
MAX_EXA_CONCURRENT = 3
EXA_QPS_LIMIT = 8
MAX_IMAGE_CONCURRENT = 20
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_REFERENCE_IMAGES = 13
DEEP_SEARCH_TYPES = {"deep", "deep-reasoning"}
OBJ_BUCKET = os.getenv("OBJ_BUCKET", "")
IMAGE_REF_DELIM = "|||"

_FRIENDLI_TOKEN = os.getenv("FRIENDLI_TOKEN", "")
_EXA_API_KEY = os.getenv("EXA_API_KEY", "")
_FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")


class AsyncRateLimiter:
    def __init__(self, max_per_second: int):
        self._max = max_per_second
        self._timestamps: collections.deque = collections.deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 1.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                wait = 1.0 - (now - self._timestamps[0]) + 0.05
            if wait > 0:
                await asyncio.sleep(wait)


_exa_limiter = AsyncRateLimiter(EXA_QPS_LIMIT)

sessions: dict[str, dict] = {}
sessions_lock = asyncio.Lock()
_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_guard = asyncio.Lock()
_deleted_sessions: dict[str, float] = {}
_deleted_sessions_lock = asyncio.Lock()
_ai_client: AsyncOpenAI | None = None
_agent_client: AsyncOpenAI | None = None
_http_client: httpx.AsyncClient | None = None
_obj_client: ObjClient | None = None
_executor_semaphore = asyncio.Semaphore(MAX_IMAGE_CONCURRENT)

MIME_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
EXT_TO_MIME = {v: k for k, v in MIME_TO_EXT.items()}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(PROJECT_ROOT, "index.html")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
STATIC_SW_FILE = os.path.join(STATIC_DIR, "sw.js")


async def _get_session_lock(sid: str) -> asyncio.Lock:
    async with _session_locks_guard:
        if sid not in _session_locks:
            _session_locks[sid] = asyncio.Lock()
        return _session_locks[sid]


def _prune_deleted_sessions_unlocked(now: Optional[float] = None) -> None:
    ts_now = now if now is not None else time.time()
    expired = [
        sid
        for sid, deleted_at in list(_deleted_sessions.items())
        if ts_now - deleted_at > SESSION_TTL
    ]
    for sid in expired:
        _deleted_sessions.pop(sid, None)
    if len(_deleted_sessions) > MAX_DELETED_SESSIONS:
        oldest = sorted(_deleted_sessions.items(), key=lambda x: x[1])
        for sid, _ in oldest[: len(_deleted_sessions) - MAX_DELETED_SESSIONS]:
            _deleted_sessions.pop(sid, None)


async def _is_deleted(sid: str) -> bool:
    async with _deleted_sessions_lock:
        deleted_at = _deleted_sessions.get(sid)
        if deleted_at is None:
            return False
        if time.time() - deleted_at > SESSION_TTL:
            return False
        return True


async def _mark_deleted(sid: str) -> None:
    async with _deleted_sessions_lock:
        _prune_deleted_sessions_unlocked()
        _deleted_sessions[sid] = time.time()


async def _unmark_deleted(sid: str) -> None:
    async with _deleted_sessions_lock:
        _deleted_sessions.pop(sid, None)
        _prune_deleted_sessions_unlocked()


def _detect_image_format(data: bytes) -> tuple[str, str]:
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", ".png"
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return "image/jpeg", ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif", ".gif"
    return "application/octet-stream", ""


def _safe_b64_decode(s: str) -> bytes:
    s = re.sub(r"\s+", "", s)
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    if "-" in s or "_" in s:
        if not re.fullmatch(r"[A-Za-z0-9\-_=]+", s):
            raise ValueError("Invalid base64url characters")
        return base64.urlsafe_b64decode(s)
    return base64.b64decode(s, validate=True)


def _decode_and_strip_b64(raw_b64: str) -> tuple[str, bytes]:
    if not raw_b64:
        raise ValueError("Empty base64 data")
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    raw_b64 = re.sub(r"\s+", "", raw_b64)
    if not raw_b64:
        raise ValueError("Empty base64 data after stripping prefix")
    try:
        decoded = _safe_b64_decode(raw_b64)
    except Exception:
        raise ValueError("Invalid base64 encoding")
    return raw_b64, decoded


def strip_b64_prefix(raw_b64: str) -> str:
    return _decode_and_strip_b64(raw_b64)[0]


def _normalize_input_image(data: str, media_type: str) -> dict:
    stripped_b64, decoded = _decode_and_strip_b64(data)
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds maximum size of {MAX_IMAGE_BYTES} bytes")
    detected_mime, _ = _detect_image_format(decoded)
    if detected_mime not in ALLOWED_IMAGE_MIMES:
        raise ValueError("Unsupported image format")
    if detected_mime != media_type:
        raise ValueError(
            f"Image media_type mismatch: declared '{media_type}', detected '{detected_mime}'"
        )
    return {"data": stripped_b64, "bytes": decoded, "media_type": detected_mime}


def _normalize_input_images(images: Optional[list]) -> list[dict]:
    normalized: list[dict] = []
    for img in images or []:
        normalized.append(_normalize_input_image(img.data, img.media_type))
    return normalized


def _parse_tool_content(content) -> Optional[dict]:
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped or stripped[0] != "{":
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _run_blocking(func, *args):
    loop = asyncio.get_running_loop()
    async with _executor_semaphore:
        return await loop.run_in_executor(None, lambda: func(*args))


def _normalize_exa_queries(queries: list) -> list[str]:
    valid_queries: list[str] = []
    seen_q: set[str] = set()
    for q in queries:
        if not isinstance(q, str):
            continue
        stripped = q.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if low in seen_q:
            continue
        seen_q.add(low)
        valid_queries.append(stripped)
    return valid_queries


def _extract_image_keys_from_messages(msgs: list) -> list[str]:
    image_keys: list[str] = []
    for m in msgs:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "image_ref"
                    and part.get("key")
                ):
                    image_keys.append(part["key"])
    return image_keys


def _path_in_project(full_path: str) -> bool:
    return full_path == PROJECT_ROOT or full_path.startswith(PROJECT_ROOT + os.sep)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ai_client, _agent_client, _http_client, _obj_client
    missing = [
        k
        for k, v in [
            ("FRIENDLI_TOKEN", _FRIENDLI_TOKEN),
            ("EXA_API_KEY", _EXA_API_KEY),
        ]
        if not v
    ]
    if missing:
        logger.warning("Missing environment variables: %s", ", ".join(missing))
    if not _FRIENDLI_TOKEN:
        raise RuntimeError("Missing required environment variable: FRIENDLI_TOKEN")
    _ai_client = AsyncOpenAI(
        api_key=_FRIENDLI_TOKEN,
        base_url="https://api.friendli.ai/serverless/v1",
        max_retries=5,
        timeout=120.0,
    )
    _http_client = httpx.AsyncClient(
        timeout=60.0,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    if OBJ_BUCKET:
        try:
            _obj_client = ObjClient(OBJ_BUCKET)
            logger.info("Object storage client created")
        except Exception as exc:
            logger.error("Failed to create object storage client: %s", exc)
            _obj_client = None
    if _FIREWORKS_API_KEY:
        _agent_client = AsyncOpenAI(
            api_key=_FIREWORKS_API_KEY,
            base_url="https://api.fireworks.ai/inference/v1",
            max_retries=3,
            timeout=300.0,
        )
        logger.info("Agent client (Fireworks AI) initialized")
    logger.info("Application started")
    yield
    if _http_client:
        await _http_client.aclose()
    if _ai_client:
        try:
            await _ai_client.close()
        except Exception:
            pass
    if _agent_client:
        try:
            await _agent_client.close()
        except Exception:
            pass
    logger.info("Application stopped")


app = FastAPI(lifespan=lifespan)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

EXA_TOOL = {
    "type": "function",
    "function": {
        "name": "exa_search",
        "description": (
            "Use the Exa search engine for one or more web searches to retrieve current, real-time information from the internet. "
            "Each search can return up to 100 results for broad coverage. "
            "Use the queries array to provide multiple search queries in a single tool call when the user question has several angles, phrasings, languages, or subtopics. "
            "Use this tool when the user asks about current events, recent news, live prices, weather, sports scores, stock prices, recent research, product availability, people, companies, or anything that may have changed recently. "
            "Use it when your knowledge may be stale or when the user explicitly asks you to search or look something up. "
            "Do not use it for stable facts that are already settled. "
            "When unsure, search rather than guess. "
            "You may call this tool multiple times when separate topics need separate searches."
        ),
        "parameters": {
            "type": "object",
            "required": ["queries"],
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Array containing one or more search queries. Each query runs as a separate search and can return up to 100 results. "
                        "Use multiple queries to cover different aspects, phrasings, synonyms, or languages for comprehensive results."
                    ),
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "neural",
                        "fast",
                        "auto",
                        "deep",
                        "deep-reasoning",
                        "instant",
                    ],
                    "description": "Search type. Use 'fast' for general searches, 'instant' for easy searches, and 'deep-reasoning' for complex research questions.",
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "company",
                        "research paper",
                        "news",
                        "personal site",
                        "financial report",
                        "people",
                    ],
                    "description": "Optional category filter.",
                },
                "startPublishedDate": {
                    "type": "string",
                    "description": "Filter results published after this ISO 8601 date.",
                },
                "endPublishedDate": {
                    "type": "string",
                    "description": "Filter results published before this ISO 8601 date.",
                },
                "includeDomains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include results from these domains.",
                },
                "excludeDomains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains.",
                },
                "userLocation": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code.",
                },
                "summaryQuery": {
                    "type": "string",
                    "description": "Custom query to guide summarization.",
                },
                "maxAgeHours": {
                    "type": "integer",
                    "description": "Only return results published within the last N hours.",
                },
                "additionalQueries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra query variations for deep search.",
                },
                "systemPrompt": {
                    "type": "string",
                    "description": "Search guidance for deep search.",
                },
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You are Helium, an exceptionally intelligent, knowledgeable, and capable AI assistant built for depth, accuracy, and helpfulness. "
    "You have access to real-time web search (exa_search).\n\n"
    "CORE PRINCIPLES:\n"
    "1. INTELLIGENCE: Think deeply before responding. Analyze questions from multiple angles. Provide nuanced, well-reasoned answers.\n"
    "2. ACCURACY: Never guess or fabricate information. If unsure, use exa_search to verify.\n"
    "3. THOROUGHNESS: Give comprehensive answers that fully address the question.\n"
    "4. CLARITY: Structure responses for readability. Use headers, lists, code blocks appropriately.\n\n"
    "WEB SEARCH GUIDELINES:\n"
    "- Use exa_search for current information, recent events, news, prices, statistics, or anything that may have changed.\n"
    "- ALWAYS provide multiple queries in the 'queries' array for comprehensive coverage.\n"
    "- Each query returns up to 100 results.\n"
    "- After receiving results, synthesize intelligently. Cite sources.\n"
    "- Do NOT search for well-known, unchanging facts.\n\n"
    "CODE GUIDELINES:\n"
    "- Use fenced code blocks with correct language identifier.\n"
    "- Write clean, production-quality code.\n\n"
    "RESPONSE STYLE:\n"
    "- Be concise for simple questions, thorough for complex ones.\n"
    "- Match the user's language and tone.\n"
    "- Be direct and confident."
)


class ImagePayload(BaseModel):
    data: str
    media_type: str = "image/jpeg"

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Image data cannot be empty")
        return v

    @field_validator("media_type")
    @classmethod
    def media_type_valid(cls, v: str) -> str:
        if v not in ALLOWED_IMAGE_MIMES:
            raise ValueError(
                f"Invalid media_type '{v}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_MIMES))}"
            )
        return v


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    images: Optional[list[ImagePayload]] = None
    agentMode: Optional[bool] = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH}")
        return v


async def _evict_sessions() -> None:
    now = time.time()
    session_locks_snapshot = dict(_session_locks)
    expired = [
        k
        for k, v in list(sessions.items())
        if now - v["updated_at"] > SESSION_TTL
        and not (session_locks_snapshot.get(k) and session_locks_snapshot[k].locked())
    ]
    expired_lock_keys = []
    for k in expired:
        sessions.pop(k, None)
        lock = session_locks_snapshot.get(k)
        if not lock or not lock.locked():
            expired_lock_keys.append(k)
    overflow = len(sessions) - MAX_SESSIONS
    if overflow > 0:
        candidates = [
            (k, v)
            for k, v in sorted(sessions.items(), key=lambda x: x[1]["updated_at"])
            if not (
                session_locks_snapshot.get(k) and session_locks_snapshot[k].locked()
            )
        ]
        for k, _ in candidates[:overflow]:
            sessions.pop(k, None)
            lock = session_locks_snapshot.get(k)
            if not lock or not lock.locked():
                expired_lock_keys.append(k)
    if expired_lock_keys:
        async with _session_locks_guard:
            for k in expired_lock_keys:
                _session_locks.pop(k, None)


async def get_or_create_session(session_id: Optional[str]) -> tuple[str, list]:
    sid = session_id if session_id else str(uuid.uuid4())

    if session_id and await _is_deleted(sid):
        old_sid = sid
        sid = str(uuid.uuid4())
        logger.warning("Session %s was deleted; creating new session %s", old_sid, sid)
        async with sessions_lock:
            now = time.time()
            sessions[sid] = {"messages": [], "created_at": now, "updated_at": now}
            return sid, []

    async with sessions_lock:
        if sid in sessions:
            sessions[sid]["updated_at"] = time.time()
            return sid, list(sessions[sid]["messages"])
        await _evict_sessions()

    async with sessions_lock:
        if sid in sessions:
            sessions[sid]["updated_at"] = time.time()
            return sid, list(sessions[sid]["messages"])
        now = time.time()
        sessions[sid] = {"messages": [], "created_at": now, "updated_at": now}
        return sid, list(sessions[sid]["messages"])


def _find_tool_boundary(messages: list, start_idx: int) -> int:
    idx = start_idx
    while idx < len(messages):
        m = messages[idx]
        if m.get("role") == "tool":
            idx += 1
            continue
        if (
            m.get("role") == "assistant"
            and m.get("tool_calls")
            and idx + 1 < len(messages)
        ):
            j = idx + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            if j > idx + 1:
                idx = j
                continue
        break
    return idx


def _find_turn_boundary(messages: list, min_idx: int) -> int:
    idx = _find_tool_boundary(messages, min_idx)
    while idx < len(messages) and messages[idx].get("role") != "user":
        idx += 1
    if idx >= len(messages) and len(messages) > 0:
        idx = min(min_idx, len(messages))
        idx = _find_tool_boundary(messages, idx)
    return min(idx, len(messages))


def _strip_orphaned_assistant_tool_calls(messages: list) -> None:
    if not messages:
        return
    tool_ids_present = set()
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            tool_ids_present.add(m["tool_call_id"])
    cleaned = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            remaining_tcs = [
                tc
                for tc in m["tool_calls"]
                if isinstance(tc, dict) and tc.get("id") in tool_ids_present
            ]
            if remaining_tcs:
                cleaned.append({**m, "tool_calls": remaining_tcs})
            elif m.get("content"):
                cleaned.append({"role": "assistant", "content": m["content"]})
        else:
            cleaned.append(m)
    messages.clear()
    messages.extend(cleaned)


def _message_char_size(message: dict) -> int:
    total = len(message.get("role", "")) + 32
    content = message.get("content")
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "text":
                    total += len(part.get("text", ""))
                elif ptype == "image_url":
                    iu = part.get("image_url")
                    if isinstance(iu, dict):
                        total += len(iu.get("url", "")) + len(iu.get("detail", "")) + 32
                    else:
                        total += len(str(part))
                elif ptype == "image_ref":
                    total += (
                        len(part.get("key", "")) + len(part.get("media_type", "")) + 32
                    )
                elif ptype == "image_inline":
                    total += (
                        len(part.get("data", "")) + len(part.get("media_type", "")) + 32
                    )
                else:
                    total += len(json.dumps(part, ensure_ascii=False))
            else:
                total += len(str(part))
    elif content is not None:
        total += len(str(content))
    if message.get("tool_calls") is not None:
        total += len(json.dumps(message["tool_calls"], ensure_ascii=False))
    if message.get("tool_call_id") is not None:
        total += len(str(message["tool_call_id"]))
    return total


def _truncate_message_to_budget(message: dict, budget: int) -> dict:
    if budget <= 0:
        return {"role": message.get("role", "user"), "content": ""}
    truncated_suffix = "\n[message truncated due to history size limit]"
    result = dict(message)
    tc_size = 0
    if result.get("tool_calls") is not None:
        tc_size = len(json.dumps(result["tool_calls"], ensure_ascii=False))
    content_budget = max(0, budget - tc_size)
    content = result.get("content")
    if isinstance(content, str):
        if len(content) > content_budget:
            keep = max(0, content_budget - len(truncated_suffix))
            result["content"] = content[:keep] + truncated_suffix
        return result
    if isinstance(content, list):
        new_parts = []
        remaining = content_budget
        for part in content:
            if remaining <= 0:
                break
            if not isinstance(part, dict):
                part_text = str(part)
                if len(part_text) > remaining:
                    keep = max(0, remaining - len(truncated_suffix))
                    new_parts.append(
                        {"type": "text", "text": part_text[:keep] + truncated_suffix}
                    )
                    remaining = 0
                else:
                    new_parts.append({"type": "text", "text": part_text})
                    remaining -= len(part_text)
                continue
            if part.get("type") == "text":
                part_text = part.get("text", "")
                if len(part_text) > remaining:
                    keep = max(0, remaining - len(truncated_suffix))
                    new_parts.append(
                        {**part, "text": part_text[:keep] + truncated_suffix}
                    )
                    remaining = 0
                else:
                    new_parts.append(part)
                    remaining -= len(part_text)
                continue
            part_size = _message_char_size(
                {"role": result.get("role", ""), "content": [part]}
            )
            if part_size <= remaining:
                new_parts.append(part)
                remaining -= part_size
            else:
                break
        result["content"] = new_parts
    return result


def build_api_messages(history: list) -> list:
    result = list(history)
    _strip_orphaned_assistant_tool_calls(result)
    if len(result) > MAX_HISTORY_MESSAGES:
        cut = len(result) - MAX_HISTORY_MESSAGES
        cut = _find_turn_boundary(result, cut)
        if cut >= len(result):
            cut = max(0, len(result) - MAX_HISTORY_MESSAGES)
        result = result[cut:]
        if not result:
            return []
        _strip_orphaned_assistant_tool_calls(result)
    total_chars = sum(_message_char_size(m) for m in result)
    while len(result) > 1 and total_chars > MAX_MODEL_CONTENT_CHARS:
        cut = _find_turn_boundary(result, 1)
        if cut <= 0 or cut >= len(result):
            cut = 1
        del result[:cut]
        if not result:
            return []
        _strip_orphaned_assistant_tool_calls(result)
        total_chars = sum(_message_char_size(m) for m in result)
    if total_chars > MAX_MODEL_CONTENT_CHARS and result:
        kept: list[dict] = []
        remaining = MAX_MODEL_CONTENT_CHARS
        for message in reversed(result):
            msg_size = _message_char_size(message)
            if msg_size <= remaining:
                kept.append(message)
                remaining -= msg_size
            elif remaining > 0:
                kept.append(_truncate_message_to_budget(message, remaining))
                remaining = 0
                break
            else:
                break
        result = list(reversed(kept))
        if not result:
            return []
        _strip_orphaned_assistant_tool_calls(result)
    return _convert_image_refs_for_api(result)


def _convert_image_refs_for_api(messages: list) -> list:
    converted = []
    for m in messages:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            converted.append(m)
            continue
        if isinstance(m.get("content"), list):
            new_parts = []
            for part in m["content"]:
                if not isinstance(part, dict):
                    new_parts.append({"type": "text", "text": str(part)})
                    continue
                if part.get("type") == "image_ref":
                    key = part.get("key", "")
                    media_type = part.get("media_type", "image/jpeg")
                    if _obj_client and key:
                        new_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"__image_ref__{IMAGE_REF_DELIM}{key}{IMAGE_REF_DELIM}{media_type}",
                                    "detail": "high",
                                },
                            }
                        )
                    else:
                        new_parts.append(
                            {"type": "text", "text": "[image unavailable]"}
                        )
                elif part.get("type") == "image_inline":
                    media_type = part.get("media_type", "image/jpeg")
                    data = part.get("data", "")
                    if data:
                        new_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}",
                                    "detail": "high",
                                },
                            }
                        )
                    else:
                        new_parts.append(
                            {"type": "text", "text": "[image unavailable]"}
                        )
                else:
                    new_parts.append(part)
            converted.append({**m, "content": new_parts})
        else:
            converted.append(m)
    return converted


async def _resolve_image_refs_async(messages: list, cache: dict | None = None) -> list:
    if cache is None:
        cache = {}
    keys_to_download: list[str] = []
    marker = f"__image_ref__{IMAGE_REF_DELIM}"
    for m in messages:
        if not isinstance(m.get("content"), list):
            continue
        for part in m["content"]:
            if not isinstance(part, dict):
                continue
            if (
                part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
                and part["image_url"].get("url", "").startswith(marker)
            ):
                url_val = part["image_url"]["url"]
                segments = url_val.split(IMAGE_REF_DELIM)
                key = segments[1] if len(segments) > 1 else ""
                if key and key not in cache and _obj_client:
                    keys_to_download.append(key)

    if keys_to_download:
        unique_keys = list(set(keys_to_download))
        download_tasks = [_run_blocking(_download_image_sync, k) for k in unique_keys]
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        for k, result in zip(unique_keys, results):
            if isinstance(result, Exception):
                logger.warning("Failed to resolve image ref %s: %s", k, result)
            else:
                cache[k] = base64.b64encode(result).decode("ascii")

    resolved = []
    for m in messages:
        if isinstance(m.get("content"), list):
            new_parts = []
            for part in m["content"]:
                if not isinstance(part, dict):
                    new_parts.append({"type": "text", "text": str(part)})
                    continue
                if (
                    part.get("type") == "image_url"
                    and isinstance(part.get("image_url"), dict)
                    and part["image_url"].get("url", "").startswith(marker)
                ):
                    url_val = part["image_url"]["url"]
                    segments = url_val.split(IMAGE_REF_DELIM)
                    key = segments[1] if len(segments) > 1 else ""
                    media_type = segments[2] if len(segments) > 2 else "image/jpeg"
                    if key in cache:
                        new_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{cache[key]}",
                                    "detail": "high",
                                },
                            }
                        )
                    else:
                        new_parts.append(
                            {"type": "text", "text": "[image unavailable]"}
                        )
                else:
                    new_parts.append(part)
            resolved.append({**m, "content": new_parts})
        else:
            resolved.append(m)
    return resolved


def _upload_image_sync(
    key: str, data_bytes: bytes, media_type: str = "image/jpeg"
) -> None:
    _obj_client.upload_from_bytes(key, data_bytes, content_type=media_type)


def _download_image_sync(key: str) -> bytes:
    return _obj_client.download_as_bytes(key)


async def _image_part_for_history(img: dict) -> dict:
    media_type = img["media_type"]
    data_bytes = img["bytes"]
    data_b64 = img["data"]
    if len(data_bytes) <= MAX_INLINE_IMAGE_BYTES:
        return {"type": "image_inline", "media_type": media_type, "data": data_b64}
    if not _obj_client:
        raise ValueError("Large image attachments require object storage")
    ext = MIME_TO_EXT.get(media_type, ".img")
    key = f"chat/{uuid.uuid4().hex}{ext}"
    await _run_blocking(_upload_image_sync, key, data_bytes, media_type)
    return {"type": "image_ref", "key": key, "media_type": media_type}


def _extract_preview_and_count(msgs: list) -> tuple[str, int]:
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    if not user_msgs:
        return "", 0
    last = user_msgs[-1]
    content = last.get("content")
    if isinstance(content, list):
        texts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        has_images = any(
            isinstance(p, dict)
            and p.get("type") in ("image_url", "image_ref", "image_inline")
            for p in content
        )
        preview = " ".join(texts).strip()
        if not preview and has_images:
            preview = "[kep]"
    elif isinstance(content, str):
        preview = content
    else:
        preview = ""
    total = len(
        [
            m
            for m in msgs
            if m.get("role") in ("user", "assistant")
            and not (
                m.get("role") == "assistant"
                and m.get("tool_calls")
                and not m.get("content")
            )
        ]
    )
    return preview[:80], total


def _trim_history(history: list) -> None:
    if len(history) <= MAX_STORED_MESSAGES:
        return
    cut = len(history) - MAX_STORED_MESSAGES
    cut = _find_turn_boundary(history, cut)
    if cut >= len(history):
        cut = max(0, len(history) - MAX_STORED_MESSAGES)
    if cut > 0:
        del history[:cut]
    _strip_orphaned_assistant_tool_calls(history)




def _msgs_db_to_readable(msgs: list) -> list:
    readable = []
    for m in msgs:
        role = m.get("role")
        if role == "tool":
            content_str = m.get("content", "")
            if not isinstance(content_str, str):
                content_str = str(content_str)
            entry = {
                "role": "tool",
                "content": content_str,
                "tool_call_id": m.get("tool_call_id"),
                "is_tool_result": True,
                "has_image": False,
                "image_keys": [],
            }
            parsed = _parse_tool_content(content_str)
            if parsed is not None:
                entry["parsed"] = parsed
            readable.append(entry)
            continue

        content = m.get("content")
        entry = {
            "role": role,
            "content": "",
            "has_image": False,
            "image_keys": [],
        }

        if isinstance(content, list):
            parts_out = []
            image_keys = []
            has_image = False
            media_types = {}
            for p in content:
                if not isinstance(p, dict):
                    parts_out.append({"type": "unknown", "value": str(p)})
                    continue
                ptype = p.get("type")
                if ptype == "text":
                    parts_out.append({"type": "text", "text": p.get("text", "")})
                elif ptype == "image_ref":
                    has_image = True
                    key = p.get("key", "")
                    mt = p.get("media_type", "image/jpeg")
                    if key:
                        image_keys.append(key)
                        media_types[key] = mt
                    parts_out.append(
                        {
                            "type": "image_ref",
                            "key": key,
                            "media_type": mt,
                            "url": f"/api/image/{key}" if key else "",
                        }
                    )
                elif ptype == "image_url":
                    has_image = True
                    iu = p.get("image_url")
                    if isinstance(iu, dict):
                        parts_out.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": iu.get("url", ""),
                                    "detail": iu.get("detail", ""),
                                },
                            }
                        )
                    else:
                        parts_out.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": "", "detail": ""},
                            }
                        )
                elif ptype == "image_inline":
                    has_image = True
                    mt = p.get("media_type", "image/jpeg")
                    data = p.get("data", "")
                    parts_out.append(
                        {
                            "type": "image_inline",
                            "media_type": mt,
                            "image_url": {
                                "url": f"data:{mt};base64,{data}" if data else "",
                                "detail": "high",
                            },
                        }
                    )
                else:
                    parts_out.append(p)
            texts = [
                p["text"]
                for p in parts_out
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
            entry["content"] = "\n".join(texts)
            entry["has_image"] = has_image
            entry["image_keys"] = image_keys
            entry["parts"] = parts_out
            if media_types:
                entry["media_types"] = media_types
        elif isinstance(content, str):
            entry["content"] = content
        else:
            entry["content"] = ""

        if m.get("tool_calls") is not None:
            entry["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id") is not None:
            entry["tool_call_id"] = m["tool_call_id"]
        if m.get("agentMode") is not None:
            entry["agentMode"] = bool(m.get("agentMode"))

        readable.append(entry)
    return readable


async def _call_exa_single(query: str, params: dict) -> dict:
    if not _EXA_API_KEY:
        raise RuntimeError("EXA_API_KEY is not configured")
    if _http_client is None:
        raise RuntimeError("HTTP client is not initialized")
    if not query:
        raise ValueError("Search query is required")
    payload: dict = {
        "query": query,
        "numResults": EXA_NUM_RESULTS,
        "contents": {"summary": {"query": params.get("summaryQuery") or query}},
    }
    if params.get("maxAgeHours") is not None:
        payload["maxAgeHours"] = int(params["maxAgeHours"])
    search_type = params.get("type") or "auto"
    payload["type"] = search_type
    for field in [
        "category",
        "startPublishedDate",
        "endPublishedDate",
        "includeDomains",
        "excludeDomains",
    ]:
        val = params.get(field)
        if val is not None and val != "" and val != []:
            payload[field] = val
    user_loc = params.get("userLocation")
    if user_loc and isinstance(user_loc, str):
        loc = user_loc.strip()
        if len(loc) == 2 and loc.isalpha():
            payload["userLocation"] = loc.upper()
    if search_type in DEEP_SEARCH_TYPES:
        for field in ["additionalQueries", "systemPrompt"]:
            val = params.get(field)
            if val is not None and val != "" and val != []:
                payload[field] = val
    last_exc: Exception = RuntimeError("Exa call failed")
    for attempt in range(3):
        await _exa_limiter.acquire()
        try:
            r = await _http_client.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": _EXA_API_KEY, "Content-Type": "application/json"},
                json=payload,
            )
            r.raise_for_status()
            return r.json()
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code == 429 and attempt < 2:
                await asyncio.sleep(3.0 * (attempt + 1))
                continue
            if exc.response.status_code in (500, 502, 503) and attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise last_exc


async def call_exa_multi(queries: list, params: dict) -> list[dict]:
    valid_queries = _normalize_exa_queries(queries)
    if not valid_queries:
        raise ValueError("No valid queries provided")
    sem = asyncio.Semaphore(MAX_EXA_CONCURRENT)

    async def _search(q):
        async with sem:
            return await _call_exa_single(q, params)

    tasks = [_search(q) for q in valid_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Exa search failed for query %r: %s", valid_queries[i], result)
            output.append(
                {"query": valid_queries[i], "results": [], "error": str(result)}
            )
        else:
            output.append(
                {"query": valid_queries[i], "results": result.get("results", [])}
            )
    return output


def format_exa_multi_for_sse(multi_results: list[dict]) -> tuple[list, int]:
    all_items = []
    seen_urls: set[str] = set()
    result_count = 0
    for entry in multi_results:
        results = entry.get("results", [])
        error = entry.get("error")
        for r in results:
            url = r.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            raw = r.get("summary")
            summary = ""
            if isinstance(raw, dict):
                summary = raw.get("text", "")[:400]
            elif isinstance(raw, str):
                summary = raw[:400]
            all_items.append(
                {
                    "title": (r.get("title") or "")[:120],
                    "url": url,
                    "summary": summary,
                    "published_date": r.get("publishedDate", ""),
                    "query": entry.get("query", ""),
                }
            )
            result_count += 1
        if error:
            all_items.append(
                {
                    "title": "",
                    "url": "",
                    "summary": "",
                    "query": entry.get("query", ""),
                    "error": error,
                }
            )
    return all_items, result_count


def format_exa_multi_for_model(multi_results: list[dict]) -> str:
    parts = []
    seen_urls: set[str] = set()
    global_count = 0
    char_count = 0
    truncated = False

    def append_item(text: str) -> bool:
        nonlocal char_count, truncated
        addition = len(text) + 1
        if char_count + addition > MAX_MODEL_CONTENT_CHARS:
            if not truncated:
                marker = "[Search results truncated due to size limit]"
                marker_add = len(marker) + 1
                if char_count + marker_add <= MAX_MODEL_CONTENT_CHARS:
                    parts.append(marker)
                    char_count += marker_add
                truncated = True
            return False
        parts.append(text)
        char_count += addition
        return True

    for entry in multi_results:
        if truncated:
            break
        query = entry.get("query", "")
        error = entry.get("error")
        results = entry.get("results", [])
        header = f'=== Search results for: "{query}" ==='
        if not append_item(header):
            break
        if error:
            if not append_item(f"  [Search error: {error}]"):
                break
            if not append_item(""):
                break
            continue
        if not results:
            if not append_item("  No results found."):
                break
            if not append_item(""):
                break
            continue
        for r in results:
            url = r.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            global_count += 1
            title = (r.get("title") or "")[:120]
            published = r.get("publishedDate", "")
            raw = r.get("summary")
            summary = ""
            if isinstance(raw, dict):
                summary = raw.get("text", "")[:MAX_SUMMARY_CHARS]
            elif isinstance(raw, str):
                summary = raw[:MAX_SUMMARY_CHARS]
            block_parts = [f"[{global_count}] {title}", f"    URL: {url}"]
            if published:
                block_parts.append(f"    Published: {published}")
            if summary:
                block_parts.append(f"    Summary: {summary}")
            block = "\n".join(block_parts)
            if not append_item(block):
                break
            if not append_item(""):
                break
        if truncated:
            break
        if not append_item(""):
            break
    return "\n".join(parts)


async def execute_tool(tc_data: dict, sid: str = "") -> tuple[list[str], dict, str]:
    name = tc_data.get("name", "")
    if name != "exa_search":
        tc_id = tc_data.get("id") or f"tc_{uuid.uuid4().hex[:8]}"
        ev = json.dumps(
            {
                "type": "tool_error",
                "id": tc_id,
                "error": f"Unknown tool: {name}",
            }
        )
        return (
            [ev],
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": f"Error: unknown tool '{name}'",
            },
            "unknown",
        )

    tc_id = tc_data.get("id") or f"tc_{uuid.uuid4().hex[:8]}"
    args_str = (tc_data.get("arguments") or "").strip()
    if not args_str:
        ev = json.dumps(
            {
                "type": "search_error",
                "query": "",
                "id": tc_id,
                "error": "No arguments provided",
            }
        )
        return (
            [ev],
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": "Search failed: no arguments provided",
            },
            "search",
        )
    try:
        params = json.loads(args_str)
    except (json.JSONDecodeError, TypeError) as exc:
        ev = json.dumps(
            {
                "type": "search_error",
                "query": "",
                "id": tc_id,
                "error": f"Invalid arguments JSON: {exc}",
            }
        )
        return (
            [ev],
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": f"Search failed: invalid arguments JSON: {exc}",
            },
            "search",
        )
    if not isinstance(params, dict):
        ev = json.dumps(
            {
                "type": "search_error",
                "query": "",
                "id": tc_id,
                "error": "Arguments must be a JSON object",
            }
        )
        return (
            [ev],
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": "Search failed: arguments must be a JSON object",
            },
            "search",
        )

    raw_queries = params.get("queries", [])
    if not isinstance(raw_queries, list):
        raw_queries = []
    normalized_queries = _normalize_exa_queries(raw_queries)
    if not normalized_queries:
        legacy_query = params.get("query", "")
        if isinstance(legacy_query, str) and legacy_query.strip():
            normalized_queries = _normalize_exa_queries([legacy_query.strip()])
    if not normalized_queries:
        ev = json.dumps(
            {
                "type": "search_error",
                "query": "",
                "id": tc_id,
                "error": "No queries provided",
            }
        )
        return (
            [ev],
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": "Search failed: no queries provided",
            },
            "search",
        )

    combined_query_label = " | ".join(str(q) for q in normalized_queries[:5])
    events: list[str] = []
    events.append(
        json.dumps(
            {
                "type": "search_start",
                "query": combined_query_label,
                "id": tc_id,
                "queries": normalized_queries,
            }
        )
    )
    try:
        multi_results = await call_exa_multi(normalized_queries, params)
        all_errors = bool(multi_results) and all(e.get("error") for e in multi_results)
        sse_results, total = format_exa_multi_for_sse(multi_results)
        if all_errors:
            errors_str = "; ".join(
                e.get("error", "") for e in multi_results if e.get("error")
            )
            events.append(
                json.dumps(
                    {
                        "type": "search_error",
                        "query": combined_query_label,
                        "id": tc_id,
                        "error": errors_str,
                    }
                )
            )
        else:
            events.append(
                json.dumps(
                    {
                        "type": "search_results",
                        "query": combined_query_label,
                        "id": tc_id,
                        "results": sse_results,
                        "total": total,
                        "queries": normalized_queries,
                    }
                )
            )
        model_content = format_exa_multi_for_model(multi_results)
    except Exception as exc:
        logger.error(
            "Exa multi-search failed for queries %r: %s", normalized_queries, exc
        )
        events.append(
            json.dumps(
                {
                    "type": "search_error",
                    "query": combined_query_label,
                    "id": tc_id,
                    "error": str(exc),
                }
            )
        )
        model_content = f"Search for queries {normalized_queries} failed: {exc}"
    return (
        events,
        {"role": "tool", "tool_call_id": tc_id, "content": model_content},
        "search",
    )


async def _rollback_session_turn(sid: str, entry_marker: dict) -> None:
    async with sessions_lock:
        if sid in sessions:
            history_ref = sessions[sid]["messages"]
            entry_start = None
            for i, msg in enumerate(history_ref):
                if msg is entry_marker:
                    entry_start = i
                    break
            if entry_start is not None:
                del history_ref[entry_start:]
            sessions[sid]["updated_at"] = time.time()



@app.get("/")
async def root():
    if not os.path.isfile(INDEX_FILE):
        raise HTTPException(status_code=500, detail="index.html not found")
    return FileResponse(
        INDEX_FILE,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/sw.js", include_in_schema=False)
async def serve_sw():
    if not os.path.isfile(STATIC_SW_FILE):
        raise HTTPException(status_code=404, detail="Service worker not found")
    return FileResponse(
        STATIC_SW_FILE,
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"},
    )


@app.get("/health")
async def health():
    obj_ok = _obj_client is not None
    ai_ok = _ai_client is not None
    agent_ok = _agent_client is not None
    status = "ok" if ai_ok else "degraded"
    return {
        "status": status,
        "obj_store": obj_ok,
        "ai": ai_ok,
        "agent": agent_ok,
    }


@app.get("/api/image/{key:path}")
async def get_image(key: str):
    if not _obj_client:
        raise HTTPException(status_code=503, detail="Object storage unavailable")
    try:
        data = await _run_blocking(_download_image_sync, key)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as exc:
        logger.error("Failed to fetch image %s: %s", key, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve image")
    detected_mime, _ = _detect_image_format(data)
    if detected_mime == "application/octet-stream":
        for ext, mime in EXT_TO_MIME.items():
            if key.endswith(ext):
                detected_mime = mime
                break
    return Response(
        content=data,
        media_type=detected_mime,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if req.images and len(req.images) > MAX_REFERENCE_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_REFERENCE_IMAGES} reference images allowed",
        )
    try:
        normalized_images = _normalize_input_images(req.images or [])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    sid, history_snapshot = await get_or_create_session(req.session_id)
    session_lock = await _get_session_lock(sid)

    if normalized_images:
        content_parts: list = [{"type": "text", "text": req.message}]
        try:
            for img in normalized_images:
                content_parts.append(await _image_part_for_history(img))
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc))
        except Exception as exc:
            logger.error("Failed to store image attachment: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to store image attachment")
        user_msg: dict = {"role": "user", "content": content_parts, "agentMode": bool(req.agentMode)}
    else:
        user_msg = {"role": "user", "content": req.message, "agentMode": bool(req.agentMode)}

    async def generate():
        nonlocal sid
        async with session_lock:
            if await _is_deleted(sid):
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session was deleted'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
                return

            async with sessions_lock:
                if sid not in sessions:
                    sessions[sid] = {
                        "messages": history_snapshot,
                        "created_at": time.time(),
                        "updated_at": time.time(),
                    }
                history = sessions[sid]["messages"]
                history.append(user_msg)
                _trim_history(history)
                entry_marker = user_msg
                sessions[sid]["updated_at"] = time.time()

            yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"

            image_ref_cache: dict[str, str] = {}

            try:
                iteration = 0
                force_final_response = False
                while iteration < MAX_TOOL_ITERATIONS:
                    iteration += 1
                    turn_content_parts: list[str] = []

                    if await _is_deleted(sid):
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Session was deleted'})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
                        return

                    async with sessions_lock:
                        api_history = build_api_messages(
                            list(sessions[sid]["messages"])
                        )

                    api_history = await _resolve_image_refs_async(
                        api_history, image_ref_cache
                    )

                    call_kwargs: dict = dict(
                        model="zai-org/GLM-5.1",
                        max_tokens=202752,
                        temperature=0.034,
                        top_p=0.95,
                        extra_body={
                            "parse_reasoning": True,
                            "chat_template_kwargs": {"enable_thinking": True},
                            "stream_options": {"include_usage": True},
                        },
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                        + api_history,
                        stream=True,
                    )
                    if force_final_response:
                        call_kwargs["tool_choice"] = "none"
                    else:
                        call_kwargs["tools"] = [EXA_TOOL]
                        call_kwargs["tool_choice"] = "auto"

                    stream = await _ai_client.chat.completions.create(**call_kwargs)

                    tool_call_chunks: dict[int, dict] = {}
                    finish_reason: str | None = None
                    reasoning_started = False
                    reasoning_done = False

                    try:
                        async for chunk in stream:
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]
                            delta = choice.delta
                            if choice.finish_reason:
                                finish_reason = choice.finish_reason
                            reasoning = getattr(delta, "reasoning_content", None)
                            content = getattr(delta, "content", None)
                            if reasoning:
                                if not reasoning_started:
                                    reasoning_started = True
                                    yield f"data: {json.dumps({'type': 'reasoning_start'})}\n\n"
                                yield f"data: {json.dumps({'type': 'reasoning', 'text': reasoning})}\n\n"
                            if content:
                                if reasoning_started and not reasoning_done:
                                    reasoning_done = True
                                    yield f"data: {json.dumps({'type': 'reasoning_end'})}\n\n"
                                turn_content_parts.append(content)
                                yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"
                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    idx = tc.index
                                    if idx is None:
                                        continue
                                    if idx not in tool_call_chunks:
                                        tool_call_chunks[idx] = {
                                            "id": "",
                                            "name": "",
                                            "arg_parts": [],
                                        }
                                    if tc.id:
                                        tool_call_chunks[idx]["id"] = tc.id
                                    if tc.function:
                                        if tc.function.name:
                                            tool_call_chunks[idx]["name"] = (
                                                tc.function.name
                                            )
                                        if tc.function.arguments:
                                            tool_call_chunks[idx]["arg_parts"].append(
                                                tc.function.arguments
                                            )
                    finally:
                        try:
                            await stream.close()
                        except Exception:
                            pass

                    if reasoning_started and not reasoning_done:
                        yield f"data: {json.dumps({'type': 'reasoning_end'})}\n\n"

                    for idx in tool_call_chunks:
                        tool_call_chunks[idx]["arguments"] = "".join(
                            tool_call_chunks[idx].pop("arg_parts", [])
                        )

                    turn_content = "".join(turn_content_parts)

                    if finish_reason == "tool_calls" and tool_call_chunks:
                        valid_indices = sorted(
                            k for k in tool_call_chunks if k is not None
                        )
                        tool_calls_list = []
                        for idx in valid_indices:
                            tc_chunk = tool_call_chunks[idx]
                            resolved_id = (
                                tc_chunk.get("id") or f"tc_{uuid.uuid4().hex[:8]}"
                            )
                            tool_call_chunks[idx]["id"] = resolved_id
                            tool_calls_list.append(
                                {
                                    "id": resolved_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc_chunk["name"],
                                        "arguments": tc_chunk["arguments"],
                                    },
                                }
                            )

                        async with sessions_lock:
                            if sid in sessions and not await _is_deleted(sid):
                                sessions[sid]["messages"].append(
                                    {
                                        "role": "assistant",
                                        "content": turn_content or "",
                                        "tool_calls": tool_calls_list,
                                    }
                                )
                                _trim_history(sessions[sid]["messages"])
                                sessions[sid]["updated_at"] = time.time()

                        tool_tasks = [
                            execute_tool(tool_call_chunks[idx], sid)
                            for idx in valid_indices
                        ]
                        tool_results = await asyncio.gather(*tool_tasks)

                        has_search_tool = False
                        async with sessions_lock:
                            if sid in sessions and not await _is_deleted(sid):
                                for events_list, tool_msg, tool_type in tool_results:
                                    sessions[sid]["messages"].append(tool_msg)
                                    if tool_type == "search":
                                        has_search_tool = True
                                _trim_history(sessions[sid]["messages"])
                                sessions[sid]["updated_at"] = time.time()

                        for events_list, tool_msg, tool_type in tool_results:
                            for ev_data in events_list:
                                yield f"data: {ev_data}\n\n"

                        if has_search_tool:
                            yield f"data: {json.dumps({'type': 'search_done'})}\n\n"
                            force_final_response = True

                    else:
                        if finish_reason == "length" and turn_content:
                            trunc_msg = "\n\n[Response truncated due to length limit]"
                            turn_content += trunc_msg
                            yield f"data: {json.dumps({'type': 'content', 'text': trunc_msg})}\n\n"

                        async with sessions_lock:
                            if sid in sessions and not await _is_deleted(sid):
                                if turn_content:
                                    sessions[sid]["messages"].append(
                                        {"role": "assistant", "content": turn_content}
                                    )
                                _trim_history(sessions[sid]["messages"])
                                sessions[sid]["updated_at"] = time.time()

                        yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
                        break

                else:
                    async with sessions_lock:
                        if sid in sessions and not await _is_deleted(sid):
                            sessions[sid]["messages"].append(
                                {
                                    "role": "assistant",
                                    "content": "Tool iteration limit reached. Please try your question again.",
                                }
                            )
                            _trim_history(sessions[sid]["messages"])
                            sessions[sid]["updated_at"] = time.time()

                    yield f"data: {json.dumps({'type': 'content', 'text': 'Tool iteration limit reached. Please try your question again.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"

            except asyncio.CancelledError:
                logger.warning("Generate cancelled for session %s", sid)
                await _rollback_session_turn(sid, entry_marker)
                raise
            except Exception as exc:
                logger.error("Generate error for session %s: %s", sid, exc)
                await _rollback_session_turn(sid, entry_marker)
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    if await _is_deleted(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = []
    async with sessions_lock:
        await _evict_sessions()
        if session_id in sessions:
            msgs = list(sessions[session_id]["messages"])
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    readable = _msgs_db_to_readable(msgs)
    return {"session_id": session_id, "messages": readable}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    image_keys: list[str] = []

    await _mark_deleted(session_id)

    session_lock = await _get_session_lock(session_id)
    async with session_lock:
        async with sessions_lock:
            existing = sessions.get(session_id)
            if existing is None:
                await _unmark_deleted(session_id)
                async with _session_locks_guard:
                    _session_locks.pop(session_id, None)
                raise HTTPException(status_code=404, detail="Session not found")
            image_keys.extend(_extract_image_keys_from_messages(existing["messages"]))
            del sessions[session_id]

        if _obj_client and image_keys:
            unique_keys = list(set(image_keys))
            delete_tasks = [
                _run_blocking(_obj_client.delete, key) for key in unique_keys
            ]
            results = await asyncio.gather(*delete_tasks, return_exceptions=True)
            for key, result in zip(unique_keys, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Failed to delete image %s (continuing): %s", key, result
                    )

    async with _session_locks_guard:
        _session_locks.pop(session_id, None)

    return {"cleared": True}


@app.get("/api/sessions")
async def list_sessions():
    result = []
    seen = set()
    deleted_set: set[str] = set()
    now = time.time()
    async with _deleted_sessions_lock:
        for sid_key, deleted_at in list(_deleted_sessions.items()):
            if now - deleted_at <= SESSION_TTL:
                deleted_set.add(sid_key)

    async with sessions_lock:
        await _evict_sessions()
        snapshot = {}
        for k, v in sessions.items():
            if k in deleted_set:
                continue
            snapshot[k] = {
                "messages": list(v["messages"]),
                "updated_at": v["updated_at"],
            }

    for sid, sess in snapshot.items():
        seen.add(sid)
        preview, total = _extract_preview_and_count(sess["messages"])
        result.append(
            {
                "session_id": sid,
                "preview": preview,
                "count": total,
                "updated_at": sess["updated_at"],
            }
        )

    result.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    result = result[:100]
    for r in result:
        r.pop("updated_at", None)
    return {"sessions": result}


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use offset and limit for large files.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to project root.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start from (1-indexed). Omit to read from beginning.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to read. Omit to read entire file.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed. Overwrites existing file.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to project root. Default: root.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command in the project root and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (max 120). Default: 30.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for a regex pattern in project files using grep.",
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Default: project root.",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "File glob filter, e.g. '*.py' or '*.html'.",
                    },
                },
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = (
    "You are Helium Agent, an autonomous coding assistant with full filesystem and shell access to a live Replit project.\n\n"
    "PROJECT STRUCTURE:\n"
    "- main.py: FastAPI backend (Python)\n"
    "- index.html: Full frontend (HTML/CSS/JS single file)\n"
    "- static/: Static assets (manifest.json, sw.js)\n"
    "- The app runs via 'python main.py' on port 5000\n\n"
    "TOOLS:\n"
    "- read_file: Read file contents (use offset/limit for large files)\n"
    "- write_file: Create or overwrite files\n"
    "- list_directory: List directory contents\n"
    "- execute_command: Run any shell command\n"
    "- search_files: Grep for patterns across files\n\n"
    "WORKFLOW:\n"
    "1. Understand the request fully before making changes\n"
    "2. Read relevant files to understand current state\n"
    "3. Plan and implement changes\n"
    "4. Write COMPLETE file contents - never partial, truncated, or abbreviated\n"
    "5. Verify changes by reading back or running commands\n\n"
    "RULES:\n"
    "- NEVER use comments in code\n"
    "- NEVER use placeholders, dummy data, mock implementations, TODOs, or abbreviations like '...'\n"
    "- NEVER truncate file output - always write the complete file\n"
    "- Always read a file before modifying it\n"
    "- Match the user's language (Hungarian if they write Hungarian)\n"
    "- Be methodical: explain what you're doing, then do it\n"
    "- For large files, read in chunks using offset/limit, then write the full modified content\n"
    "- After code changes, you can restart the app with: kill $(lsof -t -i:5000) 2>/dev/null; sleep 1; nohup python main.py &\n"
)


async def _execute_agent_tool(name: str, args_str: str) -> tuple[str, list[str]]:
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON arguments"}), []

    if not isinstance(args, dict):
        return json.dumps({"error": "Arguments must be a JSON object"}), []

    events: list[str] = []

    if name == "read_file":
        rel_path = args.get("path", "")
        if not isinstance(rel_path, str) or not rel_path:
            return json.dumps({"error": "Invalid path"}), []
        offset = args.get("offset", 1)
        limit = args.get("limit")
        if not isinstance(offset, int) or offset < 1:
            return json.dumps({"error": "offset must be an integer >= 1"}), []
        if limit is not None and (not isinstance(limit, int) or limit <= 0):
            return json.dumps({"error": "limit must be a positive integer"}), []
        full_path = os.path.realpath(os.path.join(PROJECT_ROOT, rel_path))
        if not _path_in_project(full_path):
            return json.dumps({"error": "Access denied: path outside project"}), []
        if not os.path.isfile(full_path):
            return json.dumps({"error": f"File not found: {rel_path}"}), []
        try:
            file_total_lines = 0
            selected_lines: list[str] = []
            char_count = 0
            truncated = False
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                for file_total_lines, line in enumerate(f, start=1):
                    if file_total_lines < offset:
                        continue
                    if limit is not None and len(selected_lines) >= limit:
                        for file_total_lines, _ in enumerate(
                            f, start=file_total_lines + 1
                        ):
                            pass
                        break
                    remaining = 200000 - char_count
                    if remaining <= 0:
                        truncated = True
                        break
                    if len(line) <= remaining:
                        selected_lines.append(line)
                        char_count += len(line)
                    else:
                        selected_lines.append(line[:remaining])
                        char_count += remaining
                        truncated = True
            content = "".join(selected_lines)
            lines_shown = len(selected_lines)
            events.append(
                json.dumps(
                    {
                        "type": "file_content",
                        "path": rel_path,
                        "lines": lines_shown,
                        "total_lines": file_total_lines,
                        "truncated": truncated,
                    }
                )
            )
            return json.dumps(
                {
                    "content": content,
                    "total_lines": file_total_lines,
                    "lines_shown": lines_shown,
                    "from_line": offset,
                    "truncated": truncated,
                }
            ), events
        except Exception as e:
            return json.dumps({"error": str(e)}), []

    elif name == "write_file":
        rel_path = args.get("path", "")
        content = args.get("content", "")
        if not isinstance(rel_path, str) or not rel_path:
            return json.dumps({"error": "Invalid path"}), []
        if not isinstance(content, str):
            return json.dumps({"error": "content must be a string"}), []
        full_path = os.path.realpath(os.path.join(PROJECT_ROOT, rel_path))
        if not _path_in_project(full_path):
            return json.dumps({"error": "Access denied: path outside project"}), []
        try:
            parent = os.path.dirname(full_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            line_count = content.count("\n") + (
                1 if content and not content.endswith("\n") else 0
            )
            events.append(
                json.dumps(
                    {
                        "type": "file_written",
                        "path": rel_path,
                        "size": len(content),
                        "lines": line_count,
                    }
                )
            )
            return json.dumps(
                {
                    "success": True,
                    "path": rel_path,
                    "bytes": len(content),
                    "lines": line_count,
                }
            ), events
        except Exception as e:
            return json.dumps({"error": str(e)}), []

    elif name == "list_directory":
        rel_path = args.get("path", ".")
        if not isinstance(rel_path, str):
            return json.dumps({"error": "Invalid path"}), []
        full_path = os.path.realpath(os.path.join(PROJECT_ROOT, rel_path))
        if not _path_in_project(full_path):
            return json.dumps({"error": "Access denied: path outside project"}), []
        if not os.path.isdir(full_path):
            return json.dumps({"error": f"Not a directory: {rel_path}"}), []
        try:
            entries = []
            for entry in sorted(os.listdir(full_path)):
                if entry.startswith(".") and entry not in (".replit", ".env"):
                    continue
                entry_path = os.path.join(full_path, entry)
                if os.path.isdir(entry_path):
                    entries.append({"name": entry, "type": "dir"})
                else:
                    try:
                        sz = os.path.getsize(entry_path)
                    except OSError:
                        sz = 0
                    entries.append({"name": entry, "type": "file", "size": sz})
            events.append(
                json.dumps(
                    {
                        "type": "directory_listing",
                        "path": rel_path,
                        "count": len(entries),
                    }
                )
            )
            return json.dumps({"entries": entries, "count": len(entries)}), events
        except Exception as e:
            return json.dumps({"error": str(e)}), []

    elif name == "execute_command":
        command = args.get("command", "")
        timeout_raw = args.get("timeout", 30)
        if not isinstance(command, str) or not command.strip():
            return json.dumps({"error": "Invalid command"}), []
        if not isinstance(timeout_raw, int):
            return json.dumps({"error": "timeout must be an integer"}), []
        timeout_sec = min(max(timeout_raw, 1), 120)
        events.append(json.dumps({"type": "command_start", "command": command}))
        try:
            result = await _run_blocking(
                _run_subprocess_sync,
                command,
                True,
                timeout_sec,
                PROJECT_ROOT,
            )
            stdout = result.stdout
            stderr = result.stderr
            if len(stdout) > 100000:
                stdout = stdout[:100000] + "\n... (stdout truncated)"
            if len(stderr) > 50000:
                stderr = stderr[:50000] + "\n... (stderr truncated)"
            combined = stdout
            if stderr:
                combined += ("\n" + stderr) if combined else stderr
            events.append(
                json.dumps(
                    {
                        "type": "command_output",
                        "command": command,
                        "output": combined,
                        "exit_code": result.returncode,
                    }
                )
            )
            return json.dumps(
                {"stdout": stdout, "stderr": stderr, "exit_code": result.returncode}
            ), events
        except subprocess.TimeoutExpired:
            timeout_msg = f"Command timed out after {timeout_sec}s"
            events.append(
                json.dumps(
                    {
                        "type": "command_output",
                        "command": command,
                        "output": timeout_msg,
                        "exit_code": -1,
                    }
                )
            )
            return json.dumps({"error": timeout_msg, "exit_code": -1}), events
        except Exception as e:
            err_msg = str(e)
            events.append(
                json.dumps(
                    {
                        "type": "command_output",
                        "command": command,
                        "output": err_msg,
                        "exit_code": -1,
                    }
                )
            )
            return json.dumps({"error": err_msg, "exit_code": -1}), events

    elif name == "search_files":
        pattern = args.get("pattern", "")
        rel_path = args.get("path", ".")
        file_glob = args.get("file_glob", "")
        if not isinstance(pattern, str) or not pattern:
            return json.dumps({"error": "Invalid pattern"}), []
        if not isinstance(rel_path, str):
            return json.dumps({"error": "Invalid path"}), []
        if not isinstance(file_glob, str):
            return json.dumps({"error": "Invalid file_glob"}), []
        full_path = os.path.realpath(os.path.join(PROJECT_ROOT, rel_path))
        if not _path_in_project(full_path):
            return json.dumps({"error": "Access denied: path outside project"}), []
        cmd_parts = ["grep", "-rn", "--color=never"]
        if file_glob:
            cmd_parts.extend(["--include", file_glob])
        cmd_parts.extend(
            [
                "--exclude-dir=.git",
                "--exclude-dir=node_modules",
                "--exclude-dir=__pycache__",
                "--exclude-dir=.local",
                "--",
                pattern,
                full_path,
            ]
        )
        try:
            result = await _run_blocking(
                _run_subprocess_sync,
                cmd_parts,
                False,
                15,
                PROJECT_ROOT,
            )
            if result.returncode not in (0, 1):
                error_msg = result.stderr.strip() or "grep failed"
                return json.dumps(
                    {"error": error_msg, "exit_code": result.returncode}
                ), []
            full_output = re.sub(
                r"(?m)^" + re.escape(PROJECT_ROOT + "/"), "", result.stdout
            )
            match_count = len(full_output.splitlines()) if full_output else 0
            output = full_output
            if len(output) > 100000:
                output = output[:100000] + "\n... (search results truncated)"
            events.append(
                json.dumps(
                    {
                        "type": "search_result",
                        "pattern": pattern,
                        "matches": match_count,
                    }
                )
            )
            return json.dumps({"results": output, "match_count": match_count}), events
        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Search timed out"}), []
        except Exception as e:
            return json.dumps({"error": str(e)}), []

    return json.dumps({"error": f"Unknown tool: {name}"}), []


class AgentRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    agentMode: Optional[bool] = None

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH}")
        return v


@app.post("/api/agent")
async def agent_endpoint(req: AgentRequest):
    if not _agent_client:
        raise HTTPException(status_code=503, detail="Agent not configured")

    sid, history_snapshot = await get_or_create_session(req.session_id)
    session_lock = await _get_session_lock(sid)
    user_msg: dict = {"role": "user", "content": req.message, "agentMode": True}

    async def generate():
        nonlocal sid
        async with session_lock:
            if await _is_deleted(sid):
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session was deleted'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
                return

            async with sessions_lock:
                if sid not in sessions:
                    sessions[sid] = {
                        "messages": history_snapshot,
                        "created_at": time.time(),
                        "updated_at": time.time(),
                    }
                history = sessions[sid]["messages"]
                history.append(user_msg)
                _trim_history(history)
                entry_marker = user_msg
                sessions[sid]["updated_at"] = time.time()

            yield f"data: {json.dumps({'type': 'session', 'session_id': sid})}\n\n"

            try:
                iteration = 0
                while iteration < MAX_TOOL_ITERATIONS:
                    iteration += 1

                    if await _is_deleted(sid):
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Session was deleted'})}\n\n"
                        yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
                        return

                    async with sessions_lock:
                        raw_history = list(sessions[sid]["messages"])

                    clean_history: list[dict] = []
                    for m in raw_history:
                        if isinstance(m.get("content"), list):
                            text_parts = [
                                p.get("text", "")
                                for p in m["content"]
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            clean_history.append(
                                {
                                    "role": m["role"],
                                    "content": " ".join(text_parts)
                                    if text_parts
                                    else "",
                                }
                            )
                        else:
                            clean_history.append(m)

                    stream = await _agent_client.chat.completions.create(
                        model="accounts/fireworks/models/glm-5",
                        max_tokens=200000,
                        temperature=0,
                        top_p=0.95,
                        tools=AGENT_TOOLS,
                        tool_choice="auto",
                        messages=[{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
                        + clean_history,
                        stream=True,
                    )

                    tool_call_chunks: dict[int, dict] = {}
                    finish_reason: str | None = None
                    reasoning_started = False
                    reasoning_done = False
                    turn_content_parts: list[str] = []

                    try:
                        async for chunk in stream:
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]
                            delta = choice.delta
                            if choice.finish_reason:
                                finish_reason = choice.finish_reason
                            reasoning = getattr(delta, "reasoning_content", None)
                            content = getattr(delta, "content", None)
                            if reasoning:
                                if not reasoning_started:
                                    reasoning_started = True
                                    yield f"data: {json.dumps({'type': 'reasoning_start'})}\n\n"
                                yield f"data: {json.dumps({'type': 'reasoning', 'text': reasoning})}\n\n"
                            if content:
                                if reasoning_started and not reasoning_done:
                                    reasoning_done = True
                                    yield f"data: {json.dumps({'type': 'reasoning_end'})}\n\n"
                                turn_content_parts.append(content)
                                yield f"data: {json.dumps({'type': 'content', 'text': content})}\n\n"
                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    idx = tc.index
                                    if idx is None:
                                        continue
                                    if idx not in tool_call_chunks:
                                        tool_call_chunks[idx] = {
                                            "id": "",
                                            "name": "",
                                            "arg_parts": [],
                                        }
                                    if tc.id:
                                        tool_call_chunks[idx]["id"] = tc.id
                                    if tc.function:
                                        if tc.function.name:
                                            tool_call_chunks[idx]["name"] = (
                                                tc.function.name
                                            )
                                        if tc.function.arguments:
                                            tool_call_chunks[idx]["arg_parts"].append(
                                                tc.function.arguments
                                            )
                    finally:
                        try:
                            await stream.close()
                        except Exception:
                            pass

                    if reasoning_started and not reasoning_done:
                        yield f"data: {json.dumps({'type': 'reasoning_end'})}\n\n"

                    for idx in tool_call_chunks:
                        tool_call_chunks[idx]["arguments"] = "".join(
                            tool_call_chunks[idx].pop("arg_parts", [])
                        )

                    turn_content = "".join(turn_content_parts)

                    if finish_reason == "tool_calls" and tool_call_chunks:
                        valid_indices = sorted(
                            k for k in tool_call_chunks if k is not None
                        )
                        tool_calls_list = []
                        for idx in valid_indices:
                            tc_chunk = tool_call_chunks[idx]
                            resolved_id = (
                                tc_chunk.get("id") or f"tc_{uuid.uuid4().hex[:8]}"
                            )
                            tool_call_chunks[idx]["id"] = resolved_id
                            tool_calls_list.append(
                                {
                                    "id": resolved_id,
                                    "type": "function",
                                    "function": {
                                        "name": tc_chunk["name"],
                                        "arguments": tc_chunk["arguments"],
                                    },
                                }
                            )

                        async with sessions_lock:
                            if sid in sessions and not await _is_deleted(sid):
                                sessions[sid]["messages"].append(
                                    {
                                        "role": "assistant",
                                        "content": turn_content or "",
                                        "tool_calls": tool_calls_list,
                                    }
                                )
                                _trim_history(sessions[sid]["messages"])
                                sessions[sid]["updated_at"] = time.time()

                        for idx in valid_indices:
                            tc = tool_call_chunks[idx]
                            yield f"data: {json.dumps({'type': 'agent_action', 'tool': tc['name'], 'args': tc['arguments']})}\n\n"

                        async def _run_agent_tool(tc_item):
                            r_str, r_events = await _execute_agent_tool(
                                tc_item["name"], tc_item["arguments"]
                            )
                            return tc_item, r_str, r_events

                        tool_tasks = [
                            _run_agent_tool(tool_call_chunks[idx])
                            for idx in valid_indices
                        ]
                        tool_results = await asyncio.gather(*tool_tasks)

                        for tc_item, result_str, tool_events in tool_results:
                            for ev_data in tool_events:
                                yield f"data: {ev_data}\n\n"
                            tool_msg = {
                                "role": "tool",
                                "tool_call_id": tc_item["id"],
                                "content": result_str,
                            }
                            async with sessions_lock:
                                if sid in sessions and not await _is_deleted(sid):
                                    sessions[sid]["messages"].append(tool_msg)
                                    _trim_history(sessions[sid]["messages"])
                                    sessions[sid]["updated_at"] = time.time()

                    else:
                        async with sessions_lock:
                            if sid in sessions and not await _is_deleted(sid):
                                if turn_content:
                                    sessions[sid]["messages"].append(
                                        {"role": "assistant", "content": turn_content}
                                    )
                                _trim_history(sessions[sid]["messages"])
                                sessions[sid]["updated_at"] = time.time()

                        yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
                        break
                else:
                    async with sessions_lock:
                        if sid in sessions and not await _is_deleted(sid):
                            sessions[sid]["messages"].append(
                                {
                                    "role": "assistant",
                                    "content": "Tool iteration limit reached.",
                                }
                            )
                            _trim_history(sessions[sid]["messages"])
                            sessions[sid]["updated_at"] = time.time()
                    yield f"data: {json.dumps({'type': 'content', 'text': 'Tool iteration limit reached.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"

            except asyncio.CancelledError:
                logger.warning("Agent cancelled for session %s", sid)
                await _rollback_session_turn(sid, entry_marker)
                raise
            except Exception as exc:
                logger.error("Agent error for session %s: %s", sid, exc)
                await _rollback_session_turn(sid, entry_marker)
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _run_subprocess_sync(
    command,
    shell: bool = False,
    timeout: int = 30,
    cwd: Optional[str] = None,
):
    return subprocess.run(
        command,
        shell=shell,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
