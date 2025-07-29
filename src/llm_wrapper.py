import aiohttp, asyncio, atexit
from typing import Dict

API_URL  = "http://localhost:11434/api/generate"
_TIMEOUT = aiohttp.ClientTimeout(total=180)

_sessions: Dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}

async def _get_session() -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    sess = _sessions.get(loop)

    if sess is None or sess.closed:
        sess = aiohttp.ClientSession(timeout=_TIMEOUT)
        _sessions[loop] = sess
    return sess


async def call_llama_async(
        prompt: str,
        *,
        model: str = "mistral:7b-instruct",
        backend: str = "ollama",
) -> str:
    """Generate text from *prompt* using the chosen backend.

    Parameters
    ----------
    prompt : str
        The user prompt.
    model : str, optional
        Model name / ID.  For Ollama this is the local model tag;
        for OpenAI it is the chat model (e.g. ``gpt-4o``).
    backend : {"ollama", "openai"}
        Which provider to use.  Defaults to the local Ollama server.
    """

    backend = backend.lower()

    if backend == "ollama":
        payload = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {"temperature": 0.15, "num_predict": 512},
        }
        session = await _get_session()
        async with session.post(API_URL, json=payload) as resp:
            data = await resp.json()
            return data["response"]

    elif backend == "openai":
        try:
            import os, openai
            # lazily create a client – it’s cheap
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError as e:
            raise RuntimeError("openai package not installed – `pip install openai`") from e

        chat_model = model or "gpt-3.5-turbo"
        response = await client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported backend '{backend}'. Use 'ollama' or 'openai'.")


@atexit.register
def _close_sessions() -> None:
    """
    Close every cached ClientSession.  We try to run the close coroutine
    in the *original* loop when it is still alive; otherwise we fall back
    to asyncio.run() so the interpreter can shut down cleanly.
    """
    for sess in list(_sessions.values()):
        if sess.closed:
            continue

        loop = sess._loop
        if loop.is_running():
            loop.call_soon_threadsafe(asyncio.create_task, sess.close())
        else:
            try:
                asyncio.run(sess.close())
            except RuntimeError:
                pass
