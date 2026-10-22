import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# Support both local execution from the repository root and Render's
# "server" root directory. Never commit either .env file.
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_FILES = (
    _PROJECT_ROOT / "server" / ".env",
    _PROJECT_ROOT / ".env",
    Path.cwd() / ".env",
)
for _env_file in _ENV_FILES:
    if _env_file.is_file():
        load_dotenv(_env_file, override=False)

SYSTEM_PROMPT = (
    "You are Maya, a sharp and helpful AI assistant. "
    "Answer clearly and concisely in 2 to 3 sentences unless the user asks for more."
)
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def is_groq_configured() -> bool:
    """Return whether a non-placeholder Groq key is available."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(api_key and not api_key.lower().startswith("your_"))


def ask_ai(prompt: str) -> str:
    if not is_groq_configured():
        return "Groq is not configured yet. Add GROQ_API_KEY to the server environment and restart the backend."

    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=os.getenv("GROQ_MODEL", DEFAULT_MODEL),
        )
        reply = chat_completion.choices[0].message.content
        return reply.strip() if reply else "I could not find an answer for that."
    except Exception as exc:
        # Keep the API response safe while preserving the real cause in server logs.
        print(f"Groq Error: {type(exc).__name__}: {exc}")
        return "I could not reach Groq right now. Check the server logs and Groq configuration."


ask_gemini = ask_ai
