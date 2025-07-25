import subprocess, json, re, tiktoken

def call_llama(prompt: str, timeout=30) -> str:
    cmd = ["ollama", "run", "llama3:8b", prompt]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return out.stdout.strip()

def token_count(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")  # cheap fallback
    return len(enc.encode(text))