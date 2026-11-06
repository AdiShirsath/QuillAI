"""End-to-end check against a running QuillAI (local or hosted) using real Groq key.

    # http://localhost:8000
    BASE_URL=https://quillai.onrender.com python scripts/smoke_test.py
"""

import io
import os
import sys

import httpx

BASE = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
CSV = "region,sales,visits\n" + "\n".join(
    f"{r},{s},{v}"
    for r, s, v in [("N", 100, 10), ("S", 150, 14), ("E", 90, 9), ("W", 210, 21), ("N", 120, 12), ("S", 160, 15)]
)


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), name, detail)
    if not ok:
        sys.exit(1)


with httpx.Client(timeout=300) as c:
    h = c.get(f"{BASE}/health", timeout=120)  # first hit may wake a sleeping free host
    check("health", h.status_code == 200 and h.json()["agent_ready"], h.text[:120])
    up = c.post(f"{BASE}/upload", files={"file": ("t.csv", io.BytesIO(CSV.encode()), "text/csv")})
    check("upload", up.status_code == 200, up.text[:120])
    r = c.post(
        f"{BASE}/analyze/sync",
        json={
            "goal": "Find the relationship between visits and sales and summarize it",
            "file_key": up.json()["file_key"],
        },
    )
    check("analyze/sync status", r.status_code == 200, r.text[:200])
    j = r.json()
    check("steps ran", j["steps_taken"] >= 2, f"steps={j['steps_taken']} corrections={j['self_corrections']}")
    check("has summary", len(j["executive_summary"]) > 20, j["executive_summary"][:120])
    print("limitations:", j["limitations"])
print("ALL OK")
