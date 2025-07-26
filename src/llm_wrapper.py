import aiohttp, asyncio
from typing import Optional
import atexit

API_URL = "http://localhost:11434/api/generate"
_TIMEOUT = aiohttp.ClientTimeout(total=180)

# ---------- change this line ----------
_session: Optional[aiohttp.ClientSession] = None
# --------------------------------------

async def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=_TIMEOUT)
    return _session

def _close_session():
    global _session
    if _session and not _session.closed:
        asyncio.run(_session.close())

async def call_llama_async(prompt: str, model: str = "llama3:8b") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "raw":   True,
        "stream": False
    }
    session = await _get_session()
    async with session.post(API_URL, json=payload) as resp:
        data = await resp.json()
        return data["response"]
