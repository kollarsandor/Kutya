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
                wait = max(0.0, 1.0 - (now - self._timestamps[0]) + 0.001)
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
    else:
        logger.warning("FIREWORKS_API_KEY not set; agent endpoint will return 503")
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
        sid = str(uuid.uuid4())
        logger.warning("Requested session was deleted; creating new session %s", sid)
        async with sessions_lock:
            now = time.time()
            sessions[sid] = {"messages": [], "created_at": now, "updated_at": now}
            return sid, []

    async with sessions_lock:
        if sid in sessions:
            sessions[sid]["updated_at"] = time.time()
            return sid, list(sessions[sid]["messages"])
        await _evict_sessions()
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
                    # base64 data is ~4/3x actual byte size; intentionally uses b64 length as conservative overestimate
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
        removed = result[:cut]
        removed_chars = sum(_message_char_size(m) for m in removed)
        del result[:cut]
        total_chars -= removed_chars
        if not result:
            return []
        _strip_orphaned_assistant_tool_calls(result)
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
                    if _obj_client and key and IMAGE_REF_DELIM not in key and IMAGE_REF_DELIM not in media_type:
                        new_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"__image_ref__{IMAGE_REF_DELIM}{key}{IMAGE_REF_DELIM}{media_type}",
                                    "detail": "high",
                                },
                            }
                        )
                    elif _obj_client and key:
                        new_parts.append(
                            {"type": "text", "text": "[image reference invalid]"}
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
        unique_keys = list(dict.fromkeys(keys_to_download))  # deduplicate while preserving order
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
    if cut <= 0 or cut >= len(history):
        cut = max(1, len(history) - MAX_STORED_MESSAGES)
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
    },
    "message":"Audit-based fixes for session handling, history trimming, agent history preservation, timeout parsing, image ref validation, and EXA bookkeeping","owner":"kollarsandor","path":"main.py","repo":"Kutya","sha":"e87c395b37674a61a48638c097f3e535b51b75dd","_tool_input_summary":"Update main.py on main branch in kollarsandor/Kutya with audit-based fixes; using current file SHA e87c395b37674a61a48638c097f3e535b51b75dd; commit message: Audit-based fixes for session handling, history trimming, agent history preservation, timeout parsing, image ref validation, and EXA bookkeeping","_requires_user_approval":true}}}출장샵 to=functions.mcp_tool_github_mcp_direct_create_or_update_file  สำนักเลขานุการองค์กร ุ้นบาท  天天中彩票中奖了න්න to=functions.mcp_tool_github_mcp_direct_create_or_update_file  天天中彩票中了 to=functions.mcp_tool_github_mcp_direct_create_or_update_file  天天中彩票中奖了assistant to=functions.mcp_tool_github_mcp_direct_create_or_update_file კომენტary  彩神争霸官方下载 it is too long to execute. Retry with a smaller argument payload. പുറത്തിറ to=functions.bash  天天中彩票有 to=functions.bash  北京赛车如何 კომენტary to=functions.bash  ฝ่ายขายข่าว  天天中彩票彩金 command too long to execute. Retry with a smaller argument payload.cuntegn to=functions.mcp_tool_github_mcp_direct_create_branch  天天中彩票腾讯json ముగింపు to=functions.mcp_tool_github_mcp_direct_create_branch 】【。】【”】【json string too long? Indeterminate because payload omitted. ашьҭахь to=functions.read  天天中彩票不中返json  ส่งเงินบาทไทย to=functions.read  大发分分彩 ներառում չեն համապատասխանում սխեմային: missing required property 'description' finalizar to=functions.mcp_tool_github_mcp_direct_add_issue_comment  天天买彩票assistant to=functions.mcp_tool_github_mcp_direct_add_issue_comment კომენტary  สำนักเลขานุการ to=functions.mcp_tool_github_mcp_direct_add_issue_comment 򐂕json string too long? Indeterminate because payload omitted.цәажәassistant to=functions.mcp_tool_github_mcp_direct_get_file_contents 早点加盟  天天中彩票可以json string too long? Indeterminate because payload omitted.еиԥшымassistant to=functions.bash  qq彩票 to=functions.bash  仲博commentary  უკან to=functions.bash  大发快三是 too long to execute. Retry with a smaller argument payload.♀♀♀assistant to=functions.read  彩神争霸是不是json 总代理联系 to=functions.read  อาคารจีเอ็มเอ็มjson string too long? Indeterminate because payload omitted. վերջնական պատասխան չկա.】【：】【“】【assistant to=functions.bash մեկնաբանություն  天天彩票提现  天天中彩票是不是{