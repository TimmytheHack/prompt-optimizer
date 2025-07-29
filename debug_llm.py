# debug_llm.py
import asyncio
from src.llm_wrapper import call_llama_async

async def main():
    prompt = "Hello, world!"
    out = await call_llama_async(prompt, model="mistral:7b-instruct", backend="ollama")
    print("LLM output:", repr(out))

if __name__ == "__main__":
    asyncio.run(main())