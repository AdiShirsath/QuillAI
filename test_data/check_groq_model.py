"""List the models your Groq key can use"""

import os
import sys

import httpx

# from dotenv import load_dotenv
#
# load_dotenv()
key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("Set GROQ_API_KEY first")
H = {"Authorization": f"Bearer {key}"}
models = sorted(
    m["id"] for m in httpx.get("https://api.groq.com/openai/v1/models", headers=H, timeout=30).json()["data"]
)
print("Available:", *models, sep="\n  ")
want = os.environ.get("EXECUTOR_MODEL", "openai/gpt-oss-20b")
r = httpx.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers=H,
    timeout=60,
    json={"model": want, "messages": [{"role": "user", "content": "Reply with the word OK"}]},
)
print(f"\nConfigured model {want}: HTTP {r.status_code}")

if r.status_code == 200:
    print("Model test: OK")
else:
    print(r.text[:500])

print("\nRate limits reported by Groq:")
for name in [
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
]:
    if name in r.headers:
        print(f"{name}: {r.headers[name]}")
